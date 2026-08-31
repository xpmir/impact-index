//! Implementation for WAND and Block-Max WAND algorithms

use std::collections::HashMap;

use log::debug;

use crate::{
    base::{DocId, ImpactValue, Len},
    search::{
        cursor::{as_bm25_compressed, TermCursor},
        ScoredDocument, TopScoredDocuments,
    },
};

use crate::base::TermIndex;

use crate::index::SparseIndex;

/**
 * WAND algorithm
 *
 *  Broder, A. Z., Carmel, D., Herscovici, M., Soffer, A. & Zien, J.
 * Efficient query evaluation using a two-level retrieval process.
 * Proceedings of the twelfth international conference on Information and knowledge management 426–434
 * (Association for Computing Machinery, 2003).
 * DOI 10.1145/956863.956944.
*/

/// Wraps a cursor with a query weight and cached docid for cheap sorting.
/// Generic over the cursor type `C` (P3): monomorphized per (index, scorer)
/// combination, same as [`crate::search::maxscore::MaxScoreTermIterator`].
struct BlockTermImpactIteratorWrapper<C: TermCursor> {
    iterator: C,
    query_weight: f32,
    /// Cached docid to avoid calling current().docid through the vtable during sort
    cached_docid: DocId,
}

impl<C: TermCursor> BlockTermImpactIteratorWrapper<C> {
    /// Advance the iterator and update the cached docid.
    /// Returns false if the iterator is exhausted.
    fn advance_to(&mut self, min_doc_id: DocId) -> bool {
        if let Some(_) = self.iterator.next_min_doc_id(min_doc_id) {
            self.cached_docid = self.iterator.current().docid;
            true
        } else {
            false
        }
    }
}

struct WandSearch<C: TermCursor> {
    cur_doc: Option<DocId>,
    iterators: Vec<BlockTermImpactIteratorWrapper<C>>,
}

impl<C: TermCursor> WandSearch<C> {
    /// Phase 1: Find pivot using global max scores.
    ///
    /// Sorts iterators by cached docid and accumulates global `max_value()` until the
    /// sum exceeds `theta`. Returns the pivot index extended to include all
    /// cursors at the same docid.
    fn find_pivot_term(&mut self, theta: f32) -> Option<usize> {
        // Sort iterators by cached docid (no vtable calls)
        self.iterators
            .sort_by(|a, b| a.cached_docid.cmp(&b.cached_docid));

        // Accumulate global max scores until we exceed theta
        let mut upper_bound = 0.;
        for (ix, iterator) in self.iterators.iter().enumerate() {
            upper_bound += iterator.iterator.max_value() * iterator.query_weight;
            if upper_bound > theta {
                // Extend pivot to include all cursors at the same docid
                let pivot_doc = self.iterators[ix].cached_docid;
                let mut pivot = ix;
                while pivot + 1 < self.iterators.len()
                    && self.iterators[pivot + 1].cached_docid == pivot_doc
                {
                    pivot += 1;
                }
                return Some(pivot);
            }
        }

        None
    }

    fn advance(&mut self, _ix: usize, pivot: DocId) {
        // Pick term 0: smallest docid after sort, makes the most alignment progress
        if !self.iterators[0].advance_to(pivot) {
            self.iterators.remove(0);
        }
    }

    fn next(&mut self, theta: ImpactValue) -> Option<DocId> {
        loop {
            if let Some(ix) = self.find_pivot_term(theta) {
                let pivot = self.iterators[ix].cached_docid;

                if match self.cur_doc {
                    Some(cur) => pivot <= cur,
                    None => false,
                } {
                    // Pivot has already been considered, advance one iterator
                    debug!(
                        "Pivot {} has already been considered [{}], advancing",
                        pivot, ix
                    );
                    self.advance(ix, pivot);
                } else if self.iterators[0].cached_docid == pivot {
                    // Phase 2: All cursors 0..=ix are at pivot. Check block-max bound.
                    let block_ub: f32 = self.iterators[..=ix]
                        .iter()
                        .map(|t| t.iterator.max_block_value() * t.query_weight)
                        .sum();

                    if block_ub > theta {
                        // Block-max confirms: this document could be competitive
                        self.cur_doc = Some(pivot);
                        debug!("Computing score of {}", pivot);
                        return self.cur_doc;
                    } else {
                        // Phase 3b: Block-max prunes! Skip past these blocks.
                        // Pick the cursor with highest max_score (most expensive)
                        let best_ix = self.iterators[..=ix]
                            .iter()
                            .enumerate()
                            .max_by(|(_, a), (_, b)| {
                                (a.iterator.max_value() * a.query_weight)
                                    .partial_cmp(&(b.iterator.max_value() * b.query_weight))
                                    .unwrap()
                            })
                            .map(|(i, _)| i)
                            .unwrap();

                        // Skip target: end of the earliest block across pivoted cursors
                        let mut next = self.iterators[..=ix]
                            .iter()
                            .map(|t| t.iterator.max_block_doc_id())
                            .min()
                            .unwrap()
                            + 1;

                        // Don't overshoot past the next cursor after pivot
                        if ix + 1 < self.iterators.len() {
                            next = next.min(self.iterators[ix + 1].cached_docid);
                        }

                        // Never go backwards
                        next = next.max(pivot + 1);

                        debug!(
                            "BMW prune: block_ub={} <= theta={}, advancing term {} to {}",
                            block_ub, theta, best_ix, next
                        );

                        if !self.iterators[best_ix].advance_to(next) {
                            self.iterators.remove(best_ix);
                        }
                    }
                } else {
                    /* not enough mass — advance behind term to pivot */
                    self.advance(ix, pivot);
                }
            } else {
                return None;
            }
        }
    }
}

/// Searches the index using the WAND (Weak AND) algorithm.
///
/// Returns the top-k documents by score for the given query.
///
/// # Arguments
///
/// * `index` - The sparse index to search
/// * `query` - Map from term index to query weight
/// * `top_k` - Number of top documents to return
///
/// Dispatches once per query (P3), same as [`crate::search::maxscore::search_maxscore`]:
/// a `ScoredIndex` wrapping a `CompressedIndex` with `BM25Scoring` gets a
/// fully statically-typed cursor loop; anything else falls back to the
/// generic `dyn BlockTermImpactIterator` path.
pub fn search_wand<'a>(
    index: &'a dyn SparseIndex,
    query: &HashMap<TermIndex, ImpactValue>,
    top_k: usize,
) -> Vec<ScoredDocument> {
    let mut results = if let Some((compressed, bm25)) = as_bm25_compressed(index) {
        search_wand_bm25_compressed(compressed, bm25, query, top_k)
    } else {
        search_wand_dyn(index, query, top_k)
    };
    crate::search::remap_to_original_ids(index, &mut results);
    results
}

/// Generic (dyn-dispatched) WAND: works for any index/scorer combination
/// via `Box<dyn BlockTermImpactIterator>`.
fn search_wand_dyn<'a>(
    index: &'a dyn SparseIndex,
    query: &HashMap<TermIndex, ImpactValue>,
    top_k: usize,
) -> Vec<ScoredDocument> {
    let mut iterators = Vec::new();

    for (&ix, &weight) in query.iter() {
        // Discard a term if the index does not match
        if ix >= index.len() {
            debug!("Discarding term with index {}", ix);
            continue;
        }

        let iterator = index.block_iterator(ix);

        let mut wrapper = BlockTermImpactIteratorWrapper {
            iterator,
            query_weight: weight,
            cached_docid: 0,
        };
        if wrapper.iterator.next_min_doc_id(0).is_some() {
            wrapper.cached_docid = wrapper.iterator.current().docid;
            iterators.push(wrapper)
        }
    }

    search_wand_core(iterators, top_k)
}

/// Monomorphized WAND for `CompressedIndex` + `BM25Scoring` (P3): builds one
/// `CompressedScoringCursor<BM25TermScorer>` per query term (concrete type,
/// no `Box<dyn _>`) and runs the same loop, instantiated for that concrete
/// cursor type instead.
fn search_wand_bm25_compressed(
    index: &crate::compress::CompressedIndex,
    model: &crate::scoring::bm25::BM25Scoring,
    query: &HashMap<TermIndex, ImpactValue>,
    top_k: usize,
) -> Vec<ScoredDocument> {
    let mut iterators = Vec::new();

    for (&ix, &weight) in query.iter() {
        if ix >= index.len() {
            debug!("Discarding term with index {}", ix);
            continue;
        }

        let df = index.term_length(ix);
        let scorer = model.term_scorer_typed(df);
        let mut cursor = index.typed_cursor(ix, scorer);

        if cursor.next_min_doc_id(0).is_some() {
            let cached_docid = cursor.current().docid;
            iterators.push(BlockTermImpactIteratorWrapper {
                iterator: cursor,
                query_weight: weight,
                cached_docid,
            });
        }
    }

    search_wand_core(iterators, top_k)
}

/// Shared WAND loop, generic over the cursor type `C` (P3): monomorphized
/// once per (index, scorer) combination that reaches it.
fn search_wand_core<C: TermCursor>(
    iterators: Vec<BlockTermImpactIteratorWrapper<C>>,
    top_k: usize,
) -> Vec<ScoredDocument> {
    let mut search = WandSearch {
        cur_doc: None,
        iterators,
    };

    let mut results = TopScoredDocuments::new(top_k);
    let mut theta: ImpactValue = 0.;

    // Loop until there are no more candidates
    while let Some(candidate) = search.next(theta) {
        // Score the candidate with early termination:
        // Start with the block-max upper bound and tighten it as we score.
        // If the tightened bound drops below theta, stop early (PISA trick).
        let mut score: f64 = 0.;
        let mut block_ub: f64 = 0.;
        // First pass: compute block upper bound for all terms at candidate
        for x in search.iterators.iter() {
            if x.cached_docid != candidate {
                break;
            }
            block_ub += (x.iterator.max_block_value() * x.query_weight) as f64;
        }

        let mut dominated = false;
        // `iter_mut`: resolving `current()` is lazy (P3/P4 cursors, and the
        // dyn fallback's `Cell`-based iterators alike may need `&mut self`
        // to compute/cache the exact posting).
        for x in search.iterators.iter_mut() {
            if x.cached_docid != candidate {
                break;
            }
            let c = x.iterator.current();
            let actual = (x.query_weight * c.value) as f64;
            let block_max = (x.iterator.max_block_value() * x.query_weight) as f64;
            score += actual;
            block_ub -= block_max - actual; // tighten bound
            if block_ub <= theta as f64 {
                dominated = true;
                break;
            }
        }

        if !dominated {
            // Update the heap
            theta = results.add(candidate, score as f32).max(0.);
        }
    }

    results.into_sorted_vec()
}

#[cfg(test)]
mod tests {
    //! Result-identity check (P3 validation), mirroring
    //! `search::maxscore::tests`: the monomorphized `CompressedIndex` +
    //! `BM25Scoring` fast path must return exactly the same top-k docids
    //! and (bit-identical) scores as the generic `dyn
    //! BlockTermImpactIterator` path it bypasses.

    use std::collections::HashMap;
    use std::sync::Arc;

    use ndarray::Array1;

    use crate::{
        base::{load_index, ImpactValue, TermIndex},
        builder::{BuilderOptions, Indexer},
        compress::{
            docid::BitPackingCompressor, impact::GlobalQuantizerFactory, CompressionTransform,
        },
        docmeta::DocMetadata,
        scoring::{bm25::BM25Scoring, ScoredIndex},
        search::wand::search_wand_dyn,
        transforms::IndexTransform,
    };

    use super::search_wand;

    fn build_scored_index() -> ScoredIndex {
        let tmpdir = std::env::temp_dir().join(format!(
            "wand_identity_test_{}_{}",
            std::process::id(),
            rand_seed()
        ));
        std::fs::create_dir_all(&tmpdir).unwrap();

        let mut indexer = Indexer::<f32>::new(
            &tmpdir,
            &BuilderOptions {
                in_memory_threshold: 128,
                checkpoint_frequency: 0,
                checkpoint_flush_ratio: 0.5,
            },
        );

        const NUM_DOCS: u64 = 2_000;
        const VOCABULARY_SIZE: usize = 300;
        let mut doc_lengths = Vec::with_capacity(NUM_DOCS as usize);

        let mut seed: u64 = 987654321;
        let mut next_rand = || -> f32 {
            seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1);
            ((seed >> 33) as f32) / (u32::MAX as f32) * 2.0
        };

        for doc_id in 0..NUM_DOCS {
            let num_terms = 3 + ((next_rand() * 20.0) as usize).min(40);
            let mut term_set = std::collections::BTreeSet::new();
            let mut term_vals = Vec::new();
            for _ in 0..num_terms {
                let t = (next_rand() * VOCABULARY_SIZE as f32) as TermIndex % VOCABULARY_SIZE;
                let v = next_rand().abs() + 0.1;
                if term_set.insert(t) {
                    term_vals.push((t, v));
                }
            }
            doc_lengths.push(term_vals.len() as u32);
            let terms: Array1<TermIndex> = Array1::from_iter(term_vals.iter().map(|(t, _)| *t));
            let values: Array1<f32> = Array1::from_iter(term_vals.iter().map(|(_, v)| *v));
            indexer.add(doc_id, &terms, &values).unwrap();
        }
        indexer.build().unwrap();
        let raw_index = indexer.to_index(true);

        let transform = CompressionTransform {
            max_block_size: 128,
            doc_ids_compressor_factory: Box::new(BitPackingCompressor {}),
            impacts_compressor_factory: Box::new(GlobalQuantizerFactory { nbits: 16 }),
        };
        let compressed_path = tmpdir.join("compressed");
        transform.process(&compressed_path, &raw_index).unwrap();
        let index = load_index(&compressed_path, true);

        let doc_meta = Arc::new(DocMetadata::from_lengths(doc_lengths));
        ScoredIndex::new(Arc::new(index), doc_meta, Box::new(BM25Scoring::new()))
    }

    fn rand_seed() -> u64 {
        use std::time::{SystemTime, UNIX_EPOCH};
        SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_nanos() as u64
    }

    #[test]
    fn test_typed_fast_path_matches_dyn_path() {
        let scored = build_scored_index();

        let queries: Vec<HashMap<TermIndex, ImpactValue>> = (0..15)
            .map(|i| {
                let mut q = HashMap::new();
                for j in 0..6 {
                    q.insert((i * 7 + j * 13) % 300, 1.0 + (j as f32) * 0.37);
                }
                q
            })
            .collect();

        for (qi, query) in queries.iter().enumerate() {
            for &top_k in &[1usize, 5, 10, 50] {
                let dyn_results = search_wand_dyn(&scored, query, top_k);
                // `search_wand` dispatches to the typed fast path since
                // `scored` is a `ScoredIndex<CompressedIndex, BM25Scoring>`.
                let typed_results = search_wand(&scored, query, top_k);

                assert_eq!(
                    dyn_results.len(),
                    typed_results.len(),
                    "query {qi}, top_k {top_k}: result count mismatch"
                );
                for (a, b) in dyn_results.iter().zip(typed_results.iter()) {
                    assert_eq!(
                        a.docid, b.docid,
                        "query {qi}, top_k {top_k}: docid mismatch"
                    );
                    assert_eq!(
                        a.score.to_bits(),
                        b.score.to_bits(),
                        "query {qi}, top_k {top_k}, docid {}: score mismatch ({} vs {})",
                        a.docid,
                        a.score,
                        b.score
                    );
                }
            }
        }
    }
}
