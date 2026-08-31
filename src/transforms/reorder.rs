//! Document reordering by recursive graph bisection ("BP").
//!
//! Implements the algorithm from Dhulipala, Kraska et al., "Compressing
//! Graphs and Indexes with Recursive Graph Bisection" (KDD 2016), applied
//! to the doc-term bipartite graph as described for IR by Mackenzie,
//! Petri, Moffat, "Compressing Inverted Indexes with Recursive Graph
//! Bisection" (2019/2021). See `optimizations.md`, section P2.
//!
//! Renumbering document IDs so that documents sharing many terms get
//! nearby IDs has two effects, both purely from smaller/skewed gaps in
//! posting lists:
//! - smaller docid deltas -> better PFOR/bitpacking compression, faster
//!   decode;
//! - per-block impact maxima and (P1a) per-block minimum document lengths
//!   become much more skewed, so block-max pruning (WAND/MaxScore/BMP)
//!   and the P1a `min_dl` bound both discard far more blocks.
//!
//! # Algorithm
//!
//! Recursively bisect the current document order in half (split points
//! are always the midpoint of the *current* order -- deterministic, no
//! randomness). At each internal node:
//!
//! 1. Build per-term occurrence counts within the left/right halves
//!    ("degrees"), restricted to terms with `2 <= df <= n_docs / 2`
//!    (terms outside that range carry no useful clustering signal and are
//!    dropped, standard practice for BP).
//! 2. Run up to `max_iters` "swap iterations": compute, for every document
//!    in each half, the cost delta if it moved to the other half (the
//!    log-gap cost approximation `cost(d1, d2) = d1*log2((d1+d2)/d1) +
//!    d2*log2((d1+d2)/d2)` from the BP paper), sort each half by
//!    descending gain, and swap the highest-gain pairs across halves
//!    while the combined gain is positive. Stop early once no swap has
//!    positive combined gain.
//! 3. Recurse into the (now-updated) left and right halves in parallel
//!    (`rayon::join`), each targeting its own half of the final id range.
//! 4. Once a subtree's size is `<= leaf_size` (default 64), stop: the
//!    documents keep whatever relative order the swaps above left them
//!    in.
//!
//! The whole process is deterministic: the initial order is the
//! caller-supplied document order (typically the existing docid order),
//! every split is a plain midpoint split of the current order, and every
//! sort/tie-break is by an explicit total order (gain, then docid) so
//! floating-point ties never depend on iteration/thread scheduling. Which
//! *thread* computes which subtree is decided by rayon's work-stealing,
//! but each subtree's result depends only on its own (fixed) input slice,
//! so the overall permutation is identical across runs.
//!
//! # Memory
//!
//! The per-document term adjacency needed to drive bisection is stored as
//! a flat CSR structure (`Vec<u64>` offsets + `Vec<u32>` term ids), not as
//! `Vec<Vec<u32>>` per document -- at MS MARCO scale (~8.8M docs, ~350M
//! postings) a nested-Vec representation would multiply allocator
//! overhead and pointer-chasing by the document count.

use std::cell::RefCell;
use std::path::Path;
use std::sync::atomic::{AtomicU32, Ordering};

use byteorder::{LittleEndian, ReadBytesExt, WriteBytesExt};
use rayon::prelude::*;

use crate::base::{DocId, ImpactValue, Len, TermImpact, TermIndex};
use crate::docmeta::DocMetadata;
use crate::index::{SparseIndexInformation, SparseIndexView};
use crate::transforms::IndexTransform;

/// Options controlling the BP recursive-bisection algorithm.
#[derive(Clone, Debug)]
pub struct BpOptions {
    /// Recursion stops (leaves documents in whatever order the last swap
    /// pass produced) once a subtree has at most this many documents.
    pub leaf_size: usize,
    /// Maximum number of swap iterations run per internal node.
    pub max_iters: usize,
    /// Terms with document frequency strictly below this are dropped
    /// (they carry no clustering signal -- a term in <2 docs never causes
    /// a swap).
    pub min_df: usize,
    /// Terms with document frequency above `max_df_ratio * n_docs` are
    /// dropped (very common terms are roughly uniform across any split
    /// and dominate the cost purely by volume without adding signal).
    pub max_df_ratio: f64,
}

impl Default for BpOptions {
    fn default() -> Self {
        Self {
            leaf_size: 64,
            max_iters: 20,
            min_df: 2,
            max_df_ratio: 0.5,
        }
    }
}

// ---------------------------------------------------------------------
// Doc -> term adjacency (flat CSR)
// ---------------------------------------------------------------------

/// Flat CSR adjacency: document `d`'s (filtered) term ids live in
/// `terms[offsets[d]..offsets[d+1]]`, sorted ascending by term id.
struct DocTermCsr {
    offsets: Vec<u64>,
    terms: Vec<u32>,
}

impl DocTermCsr {
    #[inline]
    fn terms_of(&self, doc: u32) -> &[u32] {
        let lo = self.offsets[doc as usize] as usize;
        let hi = self.offsets[doc as usize + 1] as usize;
        &self.terms[lo..hi]
    }
}

/// Builds the doc -> term CSR adjacency from a term -> postings view.
///
/// Pass 1 (parallel over terms): for every term, materialize its posting
/// list once, drop it if its document frequency falls outside
/// `[min_df, max_df_ratio * n_docs]`, and otherwise atomically bump every
/// referenced document's degree counter.
///
/// Pass 2 (sequential, in ascending term-id order): prefix-sum the
/// degrees into offsets, then scatter each retained term's docids into
/// the flat `terms` array. Iterating terms in ascending order makes the
/// per-document term list end up sorted (needed for determinism -- it
/// must not depend on how pass 1's parallel work was scheduled).
fn build_doc_term_csr(index: &dyn SparseIndexView, n_docs: usize, opts: &BpOptions) -> DocTermCsr {
    let num_terms = index.len();
    let max_df = ((n_docs as f64) * opts.max_df_ratio) as usize;

    let degree: Vec<AtomicU32> = (0..n_docs).map(|_| AtomicU32::new(0)).collect();

    // Pass 1: filtered postings per term (empty Vec for dropped terms).
    let filtered: Vec<Vec<u32>> = (0..num_terms)
        .into_par_iter()
        .map(|term_ix| {
            let docids: Vec<u32> = index.iterator(term_ix).map(|p| p.docid as u32).collect();
            if docids.len() < opts.min_df || docids.len() > max_df {
                return Vec::new();
            }
            for &d in &docids {
                degree[d as usize].fetch_add(1, Ordering::Relaxed);
            }
            docids
        })
        .collect();

    let degree: Vec<u32> = degree.into_iter().map(|a| a.into_inner()).collect();

    let mut offsets = vec![0u64; n_docs + 1];
    for i in 0..n_docs {
        offsets[i + 1] = offsets[i] + degree[i] as u64;
    }
    let total = offsets[n_docs] as usize;

    let mut terms = vec![0u32; total];
    let mut cursor: Vec<u64> = offsets[..n_docs].to_vec();
    for (term_ix, docids) in filtered.iter().enumerate() {
        for &d in docids {
            let pos = cursor[d as usize] as usize;
            terms[pos] = term_ix as u32;
            cursor[d as usize] += 1;
        }
    }

    DocTermCsr { offsets, terms }
}

// ---------------------------------------------------------------------
// Recursive bisection
// ---------------------------------------------------------------------

struct BpContext {
    csr: DocTermCsr,
    num_terms: usize,
    max_iters: usize,
    leaf_size: usize,
}

/// Log-gap cost approximation for a term split `d1`/`d2` documents
/// between the two halves (Dhulipala et al. 2016). Zero when the term is
/// entirely on one side -- moving other documents around doesn't change
/// its (already minimal) contribution.
#[inline]
fn split_cost(d1: u32, d2: u32) -> f64 {
    if d1 == 0 || d2 == 0 {
        return 0.0;
    }
    let (d1, d2) = (d1 as f64, d2 as f64);
    let total = d1 + d2;
    d1 * (total / d1).log2() + d2 * (total / d2).log2()
}

/// Per-thread scratch buffers, reused across the (many) recursion nodes a
/// given worker thread ends up executing, to avoid a `num_terms`-sized
/// allocation per node. Safe to reuse: a node fully finishes with the
/// scratch (and clears the touched entries back to zero) before recursing,
/// so there is never a re-entrant borrow on the same thread.
struct Scratch {
    left: Vec<u32>,
    right: Vec<u32>,
    touched: Vec<u32>,
}

impl Scratch {
    fn new() -> Self {
        Self {
            left: Vec::new(),
            right: Vec::new(),
            touched: Vec::new(),
        }
    }

    fn ensure(&mut self, n: usize) {
        if self.left.len() < n {
            self.left.resize(n, 0);
            self.right.resize(n, 0);
        }
    }
}

thread_local! {
    static SCRATCH: RefCell<Scratch> = RefCell::new(Scratch::new());
}

/// Recursively bisects `order` in place. On return, `order` holds the
/// same set of document ids it started with, permuted so that documents
/// sharing many (filtered) terms are adjacent.
fn bisect(ctx: &BpContext, order: &mut [u32]) {
    if order.len() <= ctx.leaf_size {
        return;
    }
    let mid = order.len() / 2;
    let (left, right) = order.split_at_mut(mid);

    SCRATCH.with(|cell| {
        let mut s = cell.borrow_mut();
        s.ensure(ctx.num_terms);
        s.touched.clear();

        for &d in left.iter() {
            for &t in ctx.csr.terms_of(d) {
                let idx = t as usize;
                if s.left[idx] == 0 && s.right[idx] == 0 {
                    s.touched.push(t);
                }
                s.left[idx] += 1;
            }
        }
        for &d in right.iter() {
            for &t in ctx.csr.terms_of(d) {
                let idx = t as usize;
                if s.left[idx] == 0 && s.right[idx] == 0 {
                    s.touched.push(t);
                }
                s.right[idx] += 1;
            }
        }

        for _pass in 0..ctx.max_iters {
            // Plain slices, computed sequentially: nothing here may call
            // into rayon (`par_iter`/`join`) while `s` is borrowed --
            // rayon's work-stealing can run an unrelated, already-queued
            // `bisect` continuation (from an ancestor's `rayon::join`) on
            // *this* thread while it waits on a nested parallel op, which
            // would try to borrow this same thread's `SCRATCH` again and
            // panic ("RefCell already borrowed"). Gain computation is
            // O(local nnz) per pass, and cross-subtree parallelism still
            // comes from `rayon::join` below, once this borrow is dropped.
            let left_counts: &[u32] = &s.left;
            let right_counts: &[u32] = &s.right;

            let left_gain: Vec<(f64, u32, usize)> = left
                .iter()
                .enumerate()
                .map(|(i, &d)| {
                    let mut g = 0.0f64;
                    for &t in ctx.csr.terms_of(d) {
                        let idx = t as usize;
                        let (cl, cr) = (left_counts[idx], right_counts[idx]);
                        g += split_cost(cl - 1, cr + 1) - split_cost(cl, cr);
                    }
                    (g, d, i)
                })
                .collect();
            let right_gain: Vec<(f64, u32, usize)> = right
                .iter()
                .enumerate()
                .map(|(i, &d)| {
                    let mut g = 0.0f64;
                    for &t in ctx.csr.terms_of(d) {
                        let idx = t as usize;
                        let (cl, cr) = (left_counts[idx], right_counts[idx]);
                        g += split_cost(cl + 1, cr - 1) - split_cost(cl, cr);
                    }
                    (g, d, i)
                })
                .collect();

            // Descending gain, ascending docid tie-break -- keeps the
            // result independent of parallel scheduling.
            let mut left_gain = left_gain;
            let mut right_gain = right_gain;
            left_gain.sort_unstable_by(|a, b| b.0.total_cmp(&a.0).then(a.1.cmp(&b.1)));
            right_gain.sort_unstable_by(|a, b| b.0.total_cmp(&a.0).then(a.1.cmp(&b.1)));

            let mut any_swap = false;
            let pairs = left_gain.len().min(right_gain.len());
            for k in 0..pairs {
                let (gl, dl, il) = left_gain[k];
                let (gr, dr, ir) = right_gain[k];
                if gl + gr <= 0.0 {
                    break;
                }

                for &t in ctx.csr.terms_of(dl) {
                    let idx = t as usize;
                    s.left[idx] -= 1;
                    s.right[idx] += 1;
                }
                for &t in ctx.csr.terms_of(dr) {
                    let idx = t as usize;
                    s.right[idx] -= 1;
                    s.left[idx] += 1;
                }
                left[il] = dr;
                right[ir] = dl;
                any_swap = true;
            }

            if !any_swap {
                break;
            }
        }

        for i in 0..s.touched.len() {
            let idx = s.touched[i] as usize;
            s.left[idx] = 0;
            s.right[idx] = 0;
        }
    });

    rayon::join(|| bisect(ctx, left), || bisect(ctx, right));
}

/// Computes the BP permutation for `index`: `result[new_docid] =
/// original_docid`.
///
/// Deterministic: the same index and options always produce the same
/// permutation, regardless of thread count/scheduling.
pub fn compute_permutation(index: &dyn SparseIndexView, opts: &BpOptions) -> Vec<DocId> {
    let n_docs = (index.max_doc_id() + 1) as usize;
    if n_docs <= 1 {
        return (0..n_docs as DocId).collect();
    }

    let csr = build_doc_term_csr(index, n_docs, opts);
    let mut order: Vec<u32> = (0..n_docs as u32).collect();

    let ctx = BpContext {
        csr,
        num_terms: index.len(),
        max_iters: opts.max_iters.max(1),
        leaf_size: opts.leaf_size.max(1),
    };
    bisect(&ctx, &mut order);

    order.into_iter().map(|d| d as DocId).collect()
}

/// Sum, over every (filtered-or-not, this is diagnostic-only) term, of
/// `sum(log2(gap))` between consecutive docids in its posting list --
/// the quantity BP's cost function approximates and PFOR/bitpacking
/// compression tracks. Exposed for tests/benchmarks that want to verify
/// BP actually reduces it; not used by the transform itself.
pub fn log_gap_cost(index: &dyn SparseIndexView) -> f64 {
    (0..index.len())
        .into_par_iter()
        .map(|term_ix| {
            let mut prev: Option<DocId> = None;
            let mut cost = 0.0f64;
            for p in index.iterator(term_ix) {
                let gap = match prev {
                    Some(prev_id) => p.docid - prev_id,
                    None => p.docid + 1,
                };
                cost += ((gap as f64) + 1.0).log2();
                prev = Some(p.docid);
            }
            cost
        })
        .sum()
}

// ---------------------------------------------------------------------
// new-docid -> original-docid map, persisted alongside the index
// ---------------------------------------------------------------------

/// Persists the BP permutation (`new_docid -> original_docid`) so callers
/// can translate a reordered index's results back to whatever external
/// identity they associated with the original document ids.
///
/// Needed because [`DocMetadata`] only stores lengths, not external
/// names/ids -- there is nowhere else in an index directory that records
/// pre-reorder document identity.
pub struct ReorderMap;

impl ReorderMap {
    pub const FILENAME: &'static str = "reorder_map.dat";

    /// Writes `new_to_old` (indexed by new docid) to `dir/reorder_map.dat`.
    pub fn save(new_to_old: &[DocId], dir: &Path) -> std::io::Result<()> {
        let mut w = std::io::BufWriter::new(
            std::fs::File::options()
                .write(true)
                .create(true)
                .truncate(true)
                .open(dir.join(Self::FILENAME))?,
        );
        w.write_u64::<LittleEndian>(new_to_old.len() as u64)?;
        for &old in new_to_old {
            w.write_u64::<LittleEndian>(old)?;
        }
        Ok(())
    }

    /// Loads a previously-saved `new_to_old` map, if `dir` has one.
    pub fn load(dir: &Path) -> std::io::Result<Vec<DocId>> {
        let mut r = std::io::BufReader::new(std::fs::File::open(dir.join(Self::FILENAME))?);
        let n = r.read_u64::<LittleEndian>()? as usize;
        let mut v = Vec::with_capacity(n);
        for _ in 0..n {
            v.push(r.read_u64::<LittleEndian>()?);
        }
        Ok(v)
    }

    /// Whether `dir` has a reorder map (i.e. was produced by
    /// [`ReorderTransform`]).
    pub fn exists(dir: &Path) -> bool {
        dir.join(Self::FILENAME).exists()
    }
}

// ---------------------------------------------------------------------
// SparseIndexView adapter: remaps doc ids, keeps docmeta consistent
// ---------------------------------------------------------------------

/// Wraps a source [`SparseIndexView`], remapping every posting's document
/// id through a BP permutation and re-sorting each term's postings by new
/// docid (required by downstream transforms such as
/// [`crate::compress::CompressionTransform`], which assume ascending
/// docid order). Also permutes [`DocMetadata`] (lengths) so a downstream
/// compression transform computes correct per-block `min_doc_length`
/// (P1a) and so the reordered index's own docmeta stays internally
/// consistent (docid -> length) after reordering.
pub struct ReorderedIndexView<'a> {
    source: &'a dyn SparseIndexView,
    old_to_new: Vec<DocId>,
    max_doc_id: DocId,
    doc_meta: Option<DocMetadata>,
}

impl<'a> ReorderedIndexView<'a> {
    pub fn new(source: &'a dyn SparseIndexView, new_to_old: &[DocId]) -> Self {
        let n = new_to_old.len();
        let mut old_to_new = vec![0 as DocId; n];
        for (new_id, &old_id) in new_to_old.iter().enumerate() {
            old_to_new[old_id as usize] = new_id as DocId;
        }

        let doc_meta = source.doc_meta().map(|dm| {
            let mut lengths = vec![0u32; n];
            for (new_id, &old_id) in new_to_old.iter().enumerate() {
                lengths[new_id] = dm.doc_lengths.get(old_id as usize).copied().unwrap_or(0);
            }
            DocMetadata::from_lengths(lengths)
        });

        Self {
            source,
            old_to_new,
            max_doc_id: source.max_doc_id(),
            doc_meta,
        }
    }
}

impl<'a> Len for ReorderedIndexView<'a> {
    fn len(&self) -> usize {
        self.source.len()
    }
}

impl<'a> SparseIndexInformation for ReorderedIndexView<'a> {
    fn value_range(&self, term_ix: TermIndex) -> (ImpactValue, ImpactValue) {
        self.source.value_range(term_ix)
    }
}

impl<'a> SparseIndexView for ReorderedIndexView<'a> {
    fn iterator<'b>(&'b self, term_ix: TermIndex) -> Box<dyn Iterator<Item = TermImpact> + 'b> {
        let mut postings: Vec<TermImpact> = self
            .source
            .iterator(term_ix)
            .map(|p| TermImpact {
                docid: self.old_to_new[p.docid as usize],
                value: p.value,
            })
            .collect();
        postings.sort_unstable_by_key(|p| p.docid);
        Box::new(postings.into_iter())
    }

    fn max_doc_id(&self) -> DocId {
        self.max_doc_id
    }

    fn doc_meta(&self) -> Option<&DocMetadata> {
        self.doc_meta.as_ref()
    }
}

// ---------------------------------------------------------------------
// IndexTransform
// ---------------------------------------------------------------------

/// Reorders document ids by recursive graph bisection, then delegates to
/// a downstream [`IndexTransform`] (typically
/// [`crate::compress::CompressionTransform`]) to write the reordered
/// postings out -- so the output directory is a completely normal index
/// of whatever kind `sink` produces (same manifest kind/version as
/// running `sink` directly, no format change).
///
/// Also writes:
/// - permuted `docmeta.{dat,cbor}` directly into the output directory
///   (the sink itself never touches docmeta, so this is the only place a
///   reordered index's lengths get persisted correctly -- do **not**
///   call a doc-metadata-saving helper with the *original*, unpermuted
///   index afterwards, that would silently overwrite it with wrong
///   lengths);
/// - `reorder_map.dat` (new docid -> original docid), so callers can
///   translate reordered search results back to the identity space of
///   the original (un-reordered) index -- the only place that mapping is
///   recoverable, since [`DocMetadata`] carries no external
///   names/ids.
pub struct ReorderTransform {
    /// Downstream transform that writes the reordered postings.
    pub sink: Box<dyn IndexTransform>,
    /// BP algorithm parameters.
    pub options: BpOptions,
}

impl IndexTransform for ReorderTransform {
    fn process(&self, path: &Path, index: &dyn SparseIndexView) -> Result<(), std::io::Error> {
        if !path.is_dir() {
            std::fs::create_dir(path)?;
        }

        let new_to_old = compute_permutation(index, &self.options);
        let view = ReorderedIndexView::new(index, &new_to_old);

        self.sink.process(path, &view)?;

        if let Some(meta) = view.doc_meta() {
            meta.save(path)?;
        }

        ReorderMap::save(&new_to_old, path)?;

        Ok(())
    }
}
