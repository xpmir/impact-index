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
    base::{DocId, ImpactValue, TermImpact},
    index::SparseIndex,
    search::{ScoredDocument, TopScoredDocuments},
};

use crate::base::TermIndex;

use crate::index::BlockTermImpactIterator;

struct MaxScoreTermIterator<'a> {
    iterator: Box<dyn BlockTermImpactIterator + 'a>,
    term_index: usize,
    query_weight: f32,
    max_value: f64,

    // Impact with value query weight taken into account
    impact: TermImpact,
}

impl MaxScoreTermIterator<'_> {
    /// Call iterator's next
    fn next(&mut self) -> bool {
        if let Some(mut impact) = self.iterator.next() {
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
pub fn search_maxscore<'a>(
    index: &'a dyn SparseIndex,
    query: &HashMap<TermIndex, ImpactValue>,
    top_k: usize,
    options: MaxScoreOptions,
) -> Vec<ScoredDocument> {
    // --- Initialize the structures

    let mut results = TopScoredDocuments::new(top_k);
    let mut active = Vec::new();
    let mut theta: f64 = 0.;

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

    if options.length_based_ordering {
        // Sort by posting list length (increasing, so that the longest will be passive first)
        // Note: this is what Mackenzie does
        active.sort_by(|a, b| a.iterator.length().cmp(&b.iterator.length()));
    } else {
        // other option: sort by max values (decreasing)
        active.sort_by(|a, b| b.max_value.total_cmp(&a.max_value));
    }

    let mut passive = Vec::<MaxScoreTermIterator>::new();
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
