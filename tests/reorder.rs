//! Integration tests for P2: document reordering by recursive graph
//! bisection (`optimizations.md`, section P2;
//! `src/transforms/reorder.rs`).
//!
//! Covers the validation plan from the task:
//! 1. **Correctness**: reordering then searching must reproduce the exact
//!    same top-k result set/scores as searching the un-reordered index,
//!    for many queries — search results are translated back to original
//!    document ids transparently by the search functions.
//! 2. **Determinism**: the same input always yields the same permutation.
//! 3. **Effectiveness**: total log-gap cost (the quantity BP's objective
//!    approximates, and which drives PFOR/bitpacking bit-widths) strictly
//!    decreases on a clustered synthetic collection.
//! 4. **Scale sanity**: ~200k docs complete in a few minutes (rayon
//!    parallel recursion), and compressed `docids.dat` shrinks.

use std::collections::{BTreeSet, HashMap, HashSet};
use std::sync::Arc;
use std::time::Instant;

use ndarray::Array1;
use temp_dir::TempDir;

use impact_index::base::{load_index, DocId, ImpactValue, TermIndex};
use impact_index::builder::{BuilderOptions, Indexer, SparseBuilderIndex};
use impact_index::compress::docid::PForCompressor;
use impact_index::compress::impact::Identity;
use impact_index::compress::CompressionTransform;
use impact_index::docmeta::DocMetadata;
use impact_index::index::SparseIndex;
use impact_index::scoring::bm25::BM25Scoring;
use impact_index::scoring::ScoredIndex;
use impact_index::search::maxscore::{search_maxscore, MaxScoreOptions};
use impact_index::search::wand::search_wand;
use impact_index::search::ScoredDocument;
use impact_index::transforms::reorder::{
    compute_permutation, log_gap_cost, BpOptions, ReorderTransform, ReorderedIndexView,
};
use impact_index::transforms::IndexTransform;

/// Deterministic xorshift-ish PRNG (mirrors `tests/p1a_min_dl.rs`'s
/// helper) -- no external RNG dependency needed for reproducible
/// synthetic data, and it makes seeds trivially portable across test
/// runs/platforms.
struct Rng(u64);
impl Rng {
    fn next_u64(&mut self) -> u64 {
        self.0 = self.0.wrapping_mul(6364136223846793005).wrapping_add(1);
        self.0
    }
    fn next_f32(&mut self) -> f32 {
        ((self.next_u64() >> 33) as f32) / (u32::MAX as f32)
    }
    fn next_range(&mut self, bound: usize) -> usize {
        (self.next_u64() % (bound.max(1) as u64)) as usize
    }
}

/// Generates a "clustered" synthetic collection: `num_clusters` disjoint
/// pools of `terms_per_cluster` "signature" term ids, plus a shared pool
/// of `noise_vocab` noise terms. Each document draws
/// `doc_terms_from_cluster` distinct signature terms from its cluster's
/// pool and `noise_terms` distinct terms from the noise pool -- so
/// documents in the same cluster share most of their vocabulary, and BP
/// reordering has real structure to exploit.
///
/// Cluster assignment is round-robin over creation order, but documents
/// are then written into a *randomly shuffled* docid space, so the
/// initial (pre-reorder) docid order carries no cluster locality --
/// exactly the scenario reordering is meant to fix, and a fair baseline
/// (a real un-reordered collection has no reason to already be clustered
/// by docid).
///
/// Returns, indexed by (post-shuffle) docid, each document's
/// `(term, value)` list, plus the total vocabulary size.
fn generate_clustered_docs(
    seed: u64,
    num_docs: usize,
    num_clusters: usize,
    terms_per_cluster: usize,
    doc_terms_from_cluster: usize,
    noise_terms: usize,
    noise_vocab: usize,
) -> (Vec<Vec<(TermIndex, f32)>>, usize) {
    let vocab_size = num_clusters * terms_per_cluster + noise_vocab;
    let mut rng = Rng(seed);

    // Fisher-Yates: creation order -> shuffled docid.
    let mut shuffled_docid: Vec<u32> = (0..num_docs as u32).collect();
    for i in (1..num_docs).rev() {
        let j = rng.next_range(i + 1);
        shuffled_docid.swap(i, j);
    }

    let target_cluster = doc_terms_from_cluster.min(terms_per_cluster).max(1);
    let target_noise = noise_terms.min(noise_vocab);

    let mut docs: Vec<Vec<(TermIndex, f32)>> = vec![Vec::new(); num_docs];
    for creation_ix in 0..num_docs {
        let cluster = creation_ix % num_clusters;
        let cluster_base = cluster * terms_per_cluster;

        let mut chosen = BTreeSet::new();
        while chosen.len() < target_cluster {
            chosen.insert(cluster_base + rng.next_range(terms_per_cluster));
        }
        let mut noise_chosen = BTreeSet::new();
        while noise_chosen.len() < target_noise {
            noise_chosen.insert(num_clusters * terms_per_cluster + rng.next_range(noise_vocab));
        }
        chosen.extend(noise_chosen);

        let docid = shuffled_docid[creation_ix] as usize;
        docs[docid] = chosen
            .into_iter()
            .map(|t| (t, 0.5 + rng.next_f32()))
            .collect();
    }

    (docs, vocab_size)
}

/// Builds a raw (forward) index from a per-docid term/value list,
/// inserting docs in ascending docid order (required: postings are
/// appended in insertion order, not sorted at build time), and attaches
/// doc lengths as `doc_meta` for P1a `min_dl` / BM25 use downstream.
fn build_raw_index(
    dir: &std::path::Path,
    docs: &[Vec<(TermIndex, f32)>],
) -> (SparseBuilderIndex<f32>, Vec<u32>) {
    let mut indexer = Indexer::<f32>::new(
        dir,
        &BuilderOptions {
            in_memory_threshold: 256,
            checkpoint_frequency: 0,
            checkpoint_flush_ratio: 0.5,
        },
    );

    let mut lengths = Vec::with_capacity(docs.len());
    for (doc_id, terms) in docs.iter().enumerate() {
        let ids: Array1<TermIndex> = Array1::from_iter(terms.iter().map(|(t, _)| *t));
        let vals: Array1<f32> = Array1::from_iter(terms.iter().map(|(_, v)| *v));
        indexer.add(doc_id as DocId, &ids, &vals).unwrap();
        lengths.push(terms.len().max(1) as u32);
    }
    indexer.build().unwrap();
    let mut raw = indexer.to_index(true);
    raw.doc_meta = Some(DocMetadata::from_lengths(lengths.clone()));
    (raw, lengths)
}

fn compression_sink(max_block_size: usize) -> CompressionTransform {
    CompressionTransform {
        max_block_size,
        doc_ids_compressor_factory: Box::new(PForCompressor {}),
        impacts_compressor_factory: Box::new(Identity {}),
    }
}

/// Asserts that `reordered` (searched against the reordered index)
/// reproduces the exact same docid/score set as `baseline` (searched
/// against the original index). Search results on a reordered index are
/// translated back to ORIGINAL document ids automatically by the search
/// functions (`remap_to_original_ids`), so no mapping is applied here —
/// `_reorder_map` remains only for auxiliary sanity checks in callers.
fn assert_same_after_mapping(
    label: &str,
    baseline: &[ScoredDocument],
    reordered: &[ScoredDocument],
    _reorder_map: &[DocId],
) {
    assert_eq!(
        baseline.len(),
        reordered.len(),
        "{label}: result count mismatch (baseline={}, reordered={})",
        baseline.len(),
        reordered.len()
    );

    let mapped: HashMap<DocId, f32> = reordered.iter().map(|d| (d.docid, d.score)).collect();
    let baseline_map: HashMap<DocId, f32> = baseline.iter().map(|d| (d.docid, d.score)).collect();

    let mapped_ids: HashSet<DocId> = mapped.keys().copied().collect();
    let baseline_ids: HashSet<DocId> = baseline_map.keys().copied().collect();
    assert_eq!(
        mapped_ids, baseline_ids,
        "{label}: docid set mismatch after mapping reordered results back to original ids"
    );

    for (docid, &score) in &baseline_map {
        let reordered_score = mapped[docid];
        let tol = 1e-3 * score.abs().max(1.0);
        assert!(
            (score - reordered_score).abs() <= tol,
            "{label}: score mismatch for original doc {docid}: baseline={score} \
             reordered(mapped)={reordered_score} (tol={tol})"
        );
    }
}

/// **Critical correctness test**: for many queries and top-k values,
/// searching the reordered+compressed index and mapping results back via
/// `reorder_map()` must reproduce exactly the same result set and scores
/// as searching the original (un-reordered) compressed index. This is the
/// property that makes reordering safe to ship: whatever external
/// identity a caller associated with the original docids stays
/// recoverable.
#[test]
fn test_reorder_preserves_search_results() {
    let (docs, vocab) = generate_clustered_docs(0xABCD_1234, 3000, 15, 40, 10, 4, 200);
    let dir = TempDir::new().expect("tmpdir");
    let (raw, lengths) = build_raw_index(dir.path(), &docs);

    // Baseline: compress without reordering.
    let baseline_path = dir.path().join("baseline");
    compression_sink(16).process(&baseline_path, &raw).unwrap();
    DocMetadata::from_lengths(lengths.clone())
        .save(&baseline_path)
        .unwrap();

    // Reordered + compressed, via the composable transform.
    let reordered_path = dir.path().join("reordered");
    let transform = ReorderTransform {
        sink: Box::new(compression_sink(16)),
        options: BpOptions {
            leaf_size: 32,
            max_iters: 20,
            ..Default::default()
        },
    };
    transform.process(&reordered_path, &raw).unwrap();

    let baseline_index = load_index(&baseline_path, true);
    let baseline_meta = Arc::new(DocMetadata::load(&baseline_path).unwrap());
    let baseline_scored = ScoredIndex::new(
        Arc::new(baseline_index),
        baseline_meta,
        Box::new(BM25Scoring::new()),
    );

    let reordered_index = load_index(&reordered_path, true);
    let reorder_map = SparseIndex::reorder_map(&*reordered_index)
        .expect("a reordered index must expose a reorder map")
        .clone();
    assert_eq!(reorder_map.len(), docs.len());
    let reordered_meta = Arc::new(DocMetadata::load(&reordered_path).unwrap());
    let reordered_scored = ScoredIndex::new(
        Arc::new(reordered_index),
        reordered_meta,
        Box::new(BM25Scoring::new()),
    );

    let mut rng = Rng(0xF00D_BEEF);
    for qi in 0..30 {
        let n_terms = 2 + rng.next_range(6);
        let mut query: HashMap<TermIndex, ImpactValue> = HashMap::new();
        for _ in 0..n_terms {
            query.insert(rng.next_range(vocab), 0.5 + rng.next_f32() * 2.0);
        }
        if query.is_empty() {
            continue;
        }

        for &top_k in &[1usize, 5, 10, 25] {
            let base_wand = search_wand(&baseline_scored, &query, top_k);
            let reord_wand = search_wand(&reordered_scored, &query, top_k);
            assert_same_after_mapping(
                &format!("query {qi} top_k {top_k} WAND"),
                &base_wand,
                &reord_wand,
                &reorder_map,
            );

            let base_ms =
                search_maxscore(&baseline_scored, &query, top_k, MaxScoreOptions::default());
            let reord_ms =
                search_maxscore(&reordered_scored, &query, top_k, MaxScoreOptions::default());
            assert_same_after_mapping(
                &format!("query {qi} top_k {top_k} MaxScore"),
                &base_ms,
                &reord_ms,
                &reorder_map,
            );
        }
    }
}

/// The BP permutation must be a deterministic function of the index
/// content: computing it twice on the same input must yield identical
/// results (result must also actually be a permutation of `0..n_docs`).
#[test]
fn test_reorder_permutation_is_deterministic() {
    let (docs, _vocab) = generate_clustered_docs(0x1234_5678, 4000, 10, 30, 8, 3, 150);
    let dir = TempDir::new().expect("tmpdir");
    let (raw, _lengths) = build_raw_index(dir.path(), &docs);

    let opts = BpOptions {
        leaf_size: 32,
        max_iters: 20,
        ..Default::default()
    };
    let perm1 = compute_permutation(&raw, &opts);
    let perm2 = compute_permutation(&raw, &opts);
    assert_eq!(
        perm1, perm2,
        "BP permutation must be deterministic across runs"
    );

    let mut sorted = perm1.clone();
    sorted.sort_unstable();
    let expected: Vec<DocId> = (0..docs.len() as DocId).collect();
    assert_eq!(
        sorted, expected,
        "result must be a permutation of 0..n_docs"
    );
}

/// **Effectiveness test**: on a clustered synthetic collection whose
/// initial docid order is deliberately shuffled (see
/// `generate_clustered_docs`), BP reordering must strictly reduce the
/// total log-gap cost (`sum over terms of sum log2(gap+1)`) -- the
/// quantity the BP objective approximates, and what drives smaller
/// PFOR/bitpacking bit-widths.
#[test]
fn test_reorder_reduces_log_gap_cost() {
    let (docs, _vocab) = generate_clustered_docs(0x5678_9ABC, 6000, 20, 25, 8, 3, 150);
    let dir = TempDir::new().expect("tmpdir");
    let (raw, _lengths) = build_raw_index(dir.path(), &docs);

    let opts = BpOptions::default();
    let original_cost = log_gap_cost(&raw);

    let new_to_old = compute_permutation(&raw, &opts);
    let reordered_view = ReorderedIndexView::new(&raw, &new_to_old);
    let reordered_cost = log_gap_cost(&reordered_view);

    eprintln!(
        "[reorder] log-gap cost: original={:.1} reordered={:.1} ({:.1}% reduction)",
        original_cost,
        reordered_cost,
        100.0 * (1.0 - reordered_cost / original_cost)
    );
    assert!(
        reordered_cost < original_cost,
        "BP reordering should strictly reduce total log-gap cost on a clustered collection \
         (original={original_cost}, reordered={reordered_cost})"
    );
}

/// **Scale sanity**: ~200k documents (the task's stand-in for the 8.8M
/// MS MARCO target) complete in a few minutes thanks to rayon-parallel
/// recursion, and report the compressed `docids.dat` size before/after
/// on the same clustered collection.
///
/// Marked `#[ignore]` so the default `cargo test` run stays fast; run
/// explicitly with `cargo test --release -- --ignored --nocapture
/// reorder_scale`.
#[test]
#[ignore = "slow (~200k docs) -- run explicitly with --release --ignored"]
fn test_reorder_scale_200k_docs_and_compression() {
    let num_docs = 200_000usize;
    let (docs, vocab) = generate_clustered_docs(0x9E37_79B9, num_docs, 200, 60, 12, 5, 2000);
    let dir = TempDir::new().expect("tmpdir");
    let (raw, _lengths) = build_raw_index(dir.path(), &docs);

    let opts = BpOptions::default();

    let start = Instant::now();
    let new_to_old = compute_permutation(&raw, &opts);
    let elapsed = start.elapsed();
    eprintln!("[reorder] BP permutation over {num_docs} docs (vocab={vocab}) took {elapsed:?}");
    assert!(
        elapsed.as_secs() < 180,
        "BP reordering of {num_docs} docs took too long: {elapsed:?}"
    );

    assert_eq!(new_to_old.len(), num_docs);
    let mut sorted = new_to_old.clone();
    sorted.sort_unstable();
    assert_eq!(sorted, (0..num_docs as DocId).collect::<Vec<_>>());

    let baseline_path = dir.path().join("baseline");
    compression_sink(128).process(&baseline_path, &raw).unwrap();

    let reordered_path = dir.path().join("reordered");
    ReorderTransform {
        sink: Box::new(compression_sink(128)),
        options: opts,
    }
    .process(&reordered_path, &raw)
    .unwrap();

    let baseline_size = std::fs::metadata(baseline_path.join("docids.dat"))
        .unwrap()
        .len();
    let reordered_size = std::fs::metadata(reordered_path.join("docids.dat"))
        .unwrap()
        .len();
    eprintln!(
        "[reorder] docids.dat size: baseline={baseline_size} bytes, reordered={reordered_size} \
         bytes ({:.1}% reduction)",
        100.0 * (1.0 - reordered_size as f64 / baseline_size as f64)
    );
    assert!(
        reordered_size < baseline_size,
        "expected reordering to shrink compressed docids.dat (baseline={baseline_size}, \
         reordered={reordered_size})"
    );
}
