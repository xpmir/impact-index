//! Search algorithms for sparse index retrieval.
//!
//! Provides two main algorithms:
//! - [`wand::search_wand`]: WAND (Weak AND) algorithm for top-k retrieval
//! - [`maxscore::search_maxscore`]: MaxScore algorithm with optional term impact decomposition

pub mod cursor;
pub mod maxscore;
pub mod wand;

use std::{cmp::Ordering, collections::BinaryHeap};

use crate::base::{DocId, ImpactValue};

/// A document with its retrieval score, returned by search algorithms.
pub struct ScoredDocument {
    /// The document identifier.
    pub docid: DocId,
    /// The computed relevance score.
    pub score: ImpactValue,
}

impl Clone for ScoredDocument {
    fn clone(&self) -> Self {
        Self {
            docid: self.docid.clone(),
            score: self.score.clone(),
        }
    }
}

impl std::fmt::Display for ScoredDocument {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "({},{})", self.docid, self.score)
    }
}

impl PartialEq for ScoredDocument {
    fn eq(&self, other: &Self) -> bool {
        self.score == other.score
    }
}

impl Eq for ScoredDocument {}

impl PartialOrd for ScoredDocument {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        other.score.partial_cmp(&self.score)
    }
}

impl Ord for ScoredDocument {
    fn cmp(&self, other: &Self) -> Ordering {
        other.score.total_cmp(&self.score)
    }
}

/// Bounded min-heap that retains only the top-k highest-scoring documents.
pub struct TopScoredDocuments {
    heap: BinaryHeap<ScoredDocument>,
    top_k: usize,
}

impl TopScoredDocuments {
    /// Creates a new heap that will retain at most `top_k` documents.
    pub fn new(top_k: usize) -> Self {
        Self {
            heap: BinaryHeap::new(),
            top_k: top_k,
        }
    }

    /// Add a new candidate, and returns the new lower bound on scores
    pub fn add(&mut self, candidate: DocId, score: ImpactValue) -> ImpactValue {
        if self.heap.len() < self.top_k {
            self.heap.push(ScoredDocument {
                docid: candidate,
                score: score,
            });
        } else if self.heap.peek().expect("should not happen").score < score {
            self.heap.pop();
            self.heap.push(ScoredDocument {
                docid: candidate,
                score: score,
            });
        }

        // Returns the minimum score
        if self.heap.len() >= self.top_k {
            self.heap.peek().unwrap().score
        } else {
            // If the heap is not full, returns -infinity
            ImpactValue::NEG_INFINITY
        }
    }

    /// Consumes the heap and returns documents sorted by decreasing score.
    pub fn into_sorted_vec(self) -> Vec<ScoredDocument> {
        self.heap.into_sorted_vec()
    }
}
