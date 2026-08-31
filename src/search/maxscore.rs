//! MaxScore algorithm for efficient top-k retrieval.
//!
//! Based on Algorithm 1 in "Accelerating Learned Sparse Indexes Via Term Impact
//! Decomposition" (Mackenzie et al., 2022). The algorithm partitions term iterators
//! into active and passive sets based on their maximum contribution to skip
//! unnecessary score computations.

use std::collections::HashMap;

use derivative::Derivative;
use log::debug;

use crate::{
    base::{DocId, ImpactValue, Len, TermImpact},
    index::SparseIndex,
    search::{
        cursor::{as_bm25_compressed, TermCursor},
        ScoredDocument, TopScoredDocuments,
    },
};

use crate::base::TermIndex;

/// Wraps a [`TermCursor`] with a query weight and the term's contribution
/// to the current candidate, generic over the concrete cursor type `C` so
/// that the whole MaxScore loop monomorphizes (P3): `C = Box<dyn
/// BlockTermImpactIterator>` for the generic fallback, or a concrete
/// compressed+BM25 cursor for the fast path.
struct MaxScoreTermIterator<C: TermCursor> {
    iterator: C,
    term_index: usize,
    query_weight: f32,
    max_value: f64,

    // Impact with value query weight taken into account
    impact: TermImpact,
}

impl<C: TermCursor> MaxScoreTermIterator<C> {
    /// Call iterator's next
    fn next(&mut self) -> bool {
        if let Some(mut impact) = self.iterator.advance() {
            impact.value *= self.query_weight;
            self.impact = impact;
            true
        } else {
            false
        }
    }

    fn seek_gek(&'_ mut self, doc_id: DocId) -> Option<&'_ TermImpact> {
        debug!(
            "[term {}] Searching for doc id >= {}",
            self.term_index, doc_id
        );
        if doc_id <= self.impact.docid {
            return Some(&self.impact);
        }

        let min_doc_id = self.iterator.next_min_doc_id(doc_id);
        if min_doc_id.is_none() {
            return None;
        }
        let mut impact = self.iterator.current();
        impact.value *= self.query_weight;
        self.impact = impact;
        debug!(
            "[term {}] Current impact is {} / {}",
            self.term_index, self.impact, doc_id
        );
        Some(&self.impact)
    }

    /// Block-max value scaled by query weight
    fn max_block_value(&self) -> f64 {
        (self.iterator.max_block_value() * self.query_weight) as f64
    }

    /// Maximum doc ID in the current block
    fn max_block_doc_id(&self) -> DocId {
        self.iterator.max_block_doc_id()
    }
}

/// Options for the MaxScore search algorithm.
#[derive(Derivative)]
#[derivative(Default)]
pub struct MaxScoreOptions {
    /// If `true`, orders term iterators by posting list length (increasing),
    /// so the longest lists become passive first. If `false`, orders by
    /// max impact value (decreasing).
    #[derivative(Default(value = "true"))]
    pub length_based_ordering: bool,
}

/// Searches the index using the MaxScore algorithm.
///
/// Returns the top-k documents by score for the given query.
///
/// # Arguments
///
/// * `index` - The sparse index to search
/// * `query` - Map from term index to query weight
/// * `top_k` - Number of top documents to return
/// * `options` - Algorithm configuration
///
/// Dispatches once per query (P3): if `index` is a `ScoredIndex` wrapping a
/// `CompressedIndex` with `BM25Scoring` — the combination the benchmark
/// exercises — the whole per-posting path (cursor advance, batched BM25
/// scoring) is statically typed and inlined, with zero vtable calls inside
/// the search loop. Any other (index, scorer) combination falls back to the
/// generic `dyn BlockTermImpactIterator` path below, unchanged.
pub fn search_maxscore<'a>(
    index: &'a dyn SparseIndex,
    query: &HashMap<TermIndex, ImpactValue>,
    top_k: usize,
    options: MaxScoreOptions,
) -> Vec<ScoredDocument> {
    let mut results = if let Some((compressed, bm25)) = as_bm25_compressed(index) {
        search_maxscore_bm25_compressed(compressed, bm25, query, top_k, &options)
    } else {
        search_maxscore_dyn(index, query, top_k, options)
    };
    crate::search::remap_to_original_ids(index, &mut results);
    results
}

/// Generic (dyn-dispatched) MaxScore: works for any index/scorer
/// combination via `Box<dyn BlockTermImpactIterator>`.
fn search_maxscore_dyn<'a>(
    index: &'a dyn SparseIndex,
    query: &HashMap<TermIndex, ImpactValue>,
    top_k: usize,
    options: MaxScoreOptions,
) -> Vec<ScoredDocument> {
    let mut active = Vec::new();

    for (&ix, &weight) in query.iter() {
        // Discard a term if the index does not match
        if ix >= index.len() {
            debug!("Discarding term with index {}", ix);
            continue;
        }

        // Adds the iterators for this term
        for iterator in index.block_iterators(ix) {
            let max_value = ((&iterator).max_value() * weight) as f64;

            let mut wrapper = MaxScoreTermIterator {
                iterator: iterator,
                query_weight: weight,
                term_index: ix,
                impact: TermImpact {
                    value: 0.,
                    docid: 0,
                },
                max_value: max_value,
            };

            if wrapper.next() {
                active.push(wrapper);
            }
        }
    }

    search_maxscore_core(active, top_k, options.length_based_ordering)
}

/// Monomorphized MaxScore for `CompressedIndex` + `BM25Scoring` (P3):
/// builds one `CompressedScoringCursor<BM25TermScorer>` per query term
/// (concrete type, no `Box<dyn _>`) and runs the same loop as the generic
/// path, instantiated for that concrete cursor type instead.
fn search_maxscore_bm25_compressed(
    index: &crate::compress::CompressedIndex,
    model: &crate::scoring::bm25::BM25Scoring,
    query: &HashMap<TermIndex, ImpactValue>,
    top_k: usize,
    options: &MaxScoreOptions,
) -> Vec<ScoredDocument> {
    let mut active = Vec::new();

    for (&ix, &weight) in query.iter() {
        if ix >= index.len() {
            debug!("Discarding term with index {}", ix);
            continue;
        }

        let df = index.term_length(ix);
        let scorer = model.term_scorer_typed(df);
        let cursor = index.typed_cursor(ix, scorer);
        let max_value = (cursor.max_value() * weight) as f64;

        let mut wrapper = MaxScoreTermIterator {
            iterator: cursor,
            query_weight: weight,
            term_index: ix,
            impact: TermImpact {
                value: 0.,
                docid: 0,
            },
            max_value,
        };

        if wrapper.next() {
            active.push(wrapper);
        }
    }

    search_maxscore_core(active, top_k, options.length_based_ordering)
}

/// Shared MaxScore loop, generic over the cursor type `C` (P3): monomorphized
/// once per (index, scorer) combination that reaches it, so the body below
/// compiles to a fully statically-dispatched loop for each instantiation.
fn search_maxscore_core<C: TermCursor>(
    mut active: Vec<MaxScoreTermIterator<C>>,
    top_k: usize,
    length_based_ordering: bool,
) -> Vec<ScoredDocument> {
    let mut results = TopScoredDocuments::new(top_k);
    let mut theta: f64 = 0.;

    if length_based_ordering {
        // Sort by posting list length (increasing, so that the longest will be passive first)
        // Note: this is what Mackenzie does
        active.sort_by(|a, b| a.iterator.length().cmp(&b.iterator.length()));
    } else {
        // other option: sort by max values (decreasing)
        active.sort_by(|a, b| b.max_value.total_cmp(&a.max_value));
    }

    let mut passive = Vec::<MaxScoreTermIterator<C>>::new();
    let mut sum_pass = 0.;

    while !&active.is_empty() {
        // select next document, match all cursors
        let candidate: DocId = (&active)
            .iter()
            .fold(DocId::MAX as DocId, |cur, t| cur.min(t.impact.docid));

        // Block-max pruning: compute block-level upper bound for the candidate
        let block_ub: f64 = active
            .iter()
            .filter(|t| t.impact.docid == candidate)
            .map(|t| t.max_block_value())
            .sum::<f64>()
            + sum_pass;

        if block_ub <= theta {
            // The block containing the candidate cannot beat theta.
            // Compute safe skip target: min of (block_end+1) and the smallest
            // docid among active terms NOT at the candidate (they may produce
            // candidates in the skipped range that need the skipped terms).
            let block_end = active
                .iter()
                .filter(|t| t.impact.docid == candidate)
                .map(|t| t.max_block_doc_id())
                .min()
                .unwrap();

            let skip_to = active
                .iter()
                .filter(|t| t.impact.docid != candidate)
                .map(|t| t.impact.docid)
                .fold(block_end + 1, |acc, d| acc.min(d));

            debug!(
                "Block-max pruning: block_ub={} <= theta={}, skipping to {}",
                block_ub, theta, skip_to
            );

            // Advance active terms at the candidate to skip_to
            active.retain_mut(|t| {
                if t.impact.docid == candidate {
                    t.seek_gek(skip_to).is_some()
                } else {
                    true
                }
            });
        } else {
            // Score active terms first (they contribute the most)
            let mut score = 0f64;
            active.retain_mut(|t| {
                if t.impact.docid == candidate {
                    score += t.impact.value as f64;
                    if !t.next() {
                        return false;
                    }
                }
                true
            });

            // Score passive terms with early termination:
            // passive is ordered by increasing max_value (from when they were
            // moved from active). We iterate in reverse (highest max_value first)
            // so we can exit early when remaining terms can't push score above theta.
            let mut remaining_pass_ub = sum_pass;
            let mut i = passive.len();
            while i > 0 {
                i -= 1;
                // Check if score + remaining passive UB can beat theta
                if score + remaining_pass_ub <= theta {
                    // Early exit: remaining passive terms can't help
                    break;
                }
                remaining_pass_ub -= passive[i].max_value;

                if let Some(impact) = passive[i].seek_gek(candidate) {
                    if candidate == impact.docid {
                        score += impact.value as f64;
                    }
                } else {
                    // Iterator exhausted — remove it and adjust sum_pass
                    let removed = passive.remove(i);
                    sum_pass -= removed.max_value;
                }
            }

            // check against heap, update if needed
            theta = results.add(candidate, score as f32).max(0.) as f64;
        }

        // try to expand passive set
        if let Some(t) = active.last() {
            if t.max_value + sum_pass < theta {
                sum_pass += t.max_value;
                passive.push(active.pop().expect("Cannot be none"));
            }
        }
    }

    results.into_sorted_vec()
}

#[cfg(test)]
mod tests {
    //! Result-identity check (P3 validation): the monomorphized
    //! `CompressedIndex` + `BM25Scoring` fast path
    //! (`search_maxscore_bm25_compressed`) must return exactly the same
    //! top-k docids and scores as the generic `dyn BlockTermImpactIterator`
    //! path (`search_maxscore_dyn`) it bypasses — bit-identical, since
    //! `BM25TermScorer::score_chunk` (P1b) is required to compute the same
    //! scalar expression as `score`.

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
        search::maxscore::{search_maxscore_dyn, MaxScoreOptions},
        transforms::IndexTransform,
    };

    use super::search_maxscore;

    /// Builds a small synthetic compressed index wrapped with BM25 scoring
    /// (the exact (index, scorer) shape the fast path targets).
    fn build_scored_index() -> ScoredIndex {
        let tmpdir = std::env::temp_dir().join(format!(
            "maxscore_identity_test_{}_{}",
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

        let mut seed: u64 = 12345;
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

    /// Cheap process-local pseudo-uniqueness for the tmpdir name (avoids
    /// collisions when tests run concurrently in the same process).
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
                let dyn_results =
                    search_maxscore_dyn(&scored, query, top_k, MaxScoreOptions::default());
                // `search_maxscore` dispatches to the typed fast path since
                // `scored` is a `ScoredIndex<CompressedIndex, BM25Scoring>`.
                let typed_results =
                    search_maxscore(&scored, query, top_k, MaxScoreOptions::default());

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
