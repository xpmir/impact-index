//! Implementation for WAND and Block-Max WAND algorithms

use std::collections::HashMap;

use log::debug;

use crate::{
    base::{DocId, ImpactValue},
    search::{ScoredDocument, TopScoredDocuments},
};

use crate::base::TermIndex;

use crate::index::{BlockTermImpactIterator, SparseIndex};

/**
 * WAND algorithm
 *
 *  Broder, A. Z., Carmel, D., Herscovici, M., Soffer, A. & Zien, J.
 * Efficient query evaluation using a two-level retrieval process.
 * Proceedings of the twelfth international conference on Information and knowledge management 426–434
 * (Association for Computing Machinery, 2003).
 * DOI 10.1145/956863.956944.
*/

/// Wraps an iterator with a query weight and cached docid for cheap sorting
struct BlockTermImpactIteratorWrapper<'a> {
    iterator: Box<dyn BlockTermImpactIterator + 'a>,
    query_weight: f32,
    /// Cached docid to avoid calling current().docid through the vtable during sort
    cached_docid: DocId,
}

impl BlockTermImpactIteratorWrapper<'_> {
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

impl std::fmt::Display for BlockTermImpactIteratorWrapper<'_> {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        write!(
            f,
            "({}; max={})",
            self.iterator.current(),
            self.iterator.max_value() * self.query_weight
        )
    }
}

struct WandSearch<'a> {
    cur_doc: Option<DocId>,
    iterators: Vec<BlockTermImpactIteratorWrapper<'a>>,
}

impl<'a> WandSearch<'a> {
    fn new<'b: 'a>(index: &'b dyn SparseIndex, query: &HashMap<TermIndex, ImpactValue>) -> Self {
        let mut iterators = Vec::new();

        for (&ix, &weight) in query.iter() {
            // Discard a term if the index does not match
            if ix >= index.len() {
                debug!("Discarding term with index {}", ix);
                continue;
            }

            let iterator = index.block_iterator(ix);

            let mut wrapper = BlockTermImpactIteratorWrapper {
                iterator: iterator,
                query_weight: weight,
                cached_docid: 0,
            };
            if wrapper.iterator.next_min_doc_id(0).is_some() {
                wrapper.cached_docid = wrapper.iterator.current().docid;
                iterators.push(wrapper)
            }
        }

        Self {
            cur_doc: None,
            iterators: iterators,
        }
    }

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
pub fn search_wand<'a>(
    index: &'a dyn SparseIndex,
    query: &HashMap<TermIndex, ImpactValue>,
    top_k: usize,
) -> Vec<ScoredDocument> {
    let mut search = WandSearch::new(index, query);

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
        for x in search.iterators.iter() {
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
