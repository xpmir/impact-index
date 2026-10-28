//! Search over a set of segments: immutable indices that together hold one
//! changing collection (see the incremental-indexing section of the README).
//!
//! Each segment is searched with its own top-k, through its own
//! [`DocFilter`] (typically the tombstones of documents deleted or
//! re-indexed since the segment was built), and the results are merged into
//! a global top-k. Scores must be comparable across segments: true of
//! learned impacts quantised with a common range (see
//! [`crate::compress::impact::QuantizationRange`]), not of BM25 segments
//! with per-segment statistics.
//!
//! Which segments exist, and their tombstones, stay the caller's concern.

use std::collections::{BinaryHeap, HashMap};

use crate::{
    base::{DocId, ImpactValue, TermIndex},
    index::SparseIndex,
    search::{
        filter::DocFilter, maxscore::search_maxscore_with, maxscore::MaxScoreOptions,
        wand::search_wand_with, ScoredDocument,
    },
};

/// Algorithm used on each segment.
#[derive(Clone, Copy, Debug)]
pub enum SegmentAlgorithm {
    /// [`crate::search::maxscore::search_maxscore`].
    MaxScore(MaxScoreOptions),
    /// [`crate::search::wand::search_wand`].
    Wand,
}

impl Default for SegmentAlgorithm {
    fn default() -> Self {
        SegmentAlgorithm::MaxScore(MaxScoreOptions::default())
    }
}

/// A document of a segmented search result.
#[derive(Clone, Debug, PartialEq)]
pub struct SegmentScoredDocument {
    /// Position of the segment in the list given to [`SegmentedSearch::new`].
    pub segment: usize,
    /// Document id within that segment.
    pub docid: DocId,
    /// Score of the document.
    pub score: ImpactValue,
}

/// Total number of postings of `index`, which orders segment searches.
pub fn segment_size(index: &dyn SparseIndex) -> usize {
    (0..index.len())
        .map(|t| index.block_iterator(t).length())
        .sum()
}

/// One segment: an index and the filter applied to it.
pub struct Segment<'a, F> {
    pub index: &'a dyn SparseIndex,
    pub filter: F,
}

/// Searches several segments as one index.
pub struct SegmentedSearch<'a, F> {
    segments: Vec<Segment<'a, F>>,
    /// Segment positions, largest first (searched in this order).
    order: Vec<usize>,
    /// Algorithm run on each segment.
    pub algorithm: SegmentAlgorithm,
    /// If `true` (default), the global k-th score found so far is passed to
    /// the next segment as its initial threshold, so smaller segments
    /// (searched last) prune harder. Results are the same either way, up
    /// to ties at the k-th score.
    pub share_threshold: bool,
}

impl<'a, F: DocFilter> SegmentedSearch<'a, F> {
    /// Searches `segments` (in any order: they are searched largest first,
    /// size being the total number of postings, see [`segment_size`]).
    pub fn new(segments: Vec<Segment<'a, F>>) -> Self {
        let sizes: Vec<usize> = segments.iter().map(|s| segment_size(s.index)).collect();
        Self::with_sizes(segments, &sizes)
    }

    /// Like [`SegmentedSearch::new`], with sizes computed beforehand (by
    /// [`segment_size`] or otherwise; only their order matters), for
    /// callers that build a search per query.
    pub fn with_sizes(segments: Vec<Segment<'a, F>>, sizes: &[usize]) -> Self {
        assert_eq!(segments.len(), sizes.len(), "one size per segment");
        let mut order: Vec<usize> = (0..segments.len()).collect();
        order.sort_by_key(|&i| std::cmp::Reverse(sizes[i]));
        Self {
            segments,
            order,
            algorithm: SegmentAlgorithm::default(),
            share_threshold: true,
        }
    }

    /// The segments, in the order given to [`SegmentedSearch::new`].
    pub fn segments(&self) -> &[Segment<'a, F>] {
        &self.segments
    }

    /// Top-k documents over all segments, by decreasing score.
    pub fn search(
        &self,
        query: &HashMap<TermIndex, ImpactValue>,
        top_k: usize,
    ) -> Vec<SegmentScoredDocument> {
        // Min-heap (on score) of the global top-k
        let mut heap: BinaryHeap<HeapEntry> = BinaryHeap::with_capacity(top_k + 1);

        for &segment in &self.order {
            let Segment { index, filter } = &self.segments[segment];
            let min_score = if self.share_threshold && heap.len() >= top_k {
                heap.peek().map_or(0., |e| e.0.score)
            } else {
                0.
            };
            let results = match &self.algorithm {
                SegmentAlgorithm::MaxScore(options) => {
                    search_maxscore_with(*index, query, top_k, options, filter, min_score)
                }
                SegmentAlgorithm::Wand => search_wand_with(*index, query, top_k, filter, min_score),
            };

            for ScoredDocument { docid, score } in results {
                if heap.len() < top_k {
                    heap.push(HeapEntry(SegmentScoredDocument {
                        segment,
                        docid,
                        score,
                    }));
                } else if heap.peek().is_some_and(|e| e.0.score < score) {
                    heap.pop();
                    heap.push(HeapEntry(SegmentScoredDocument {
                        segment,
                        docid,
                        score,
                    }));
                } else {
                    // Results are sorted by decreasing score
                    break;
                }
            }
        }

        // `HeapEntry` orders by decreasing score, so ascending = best first
        heap.into_sorted_vec().into_iter().map(|e| e.0).collect()
    }
}

/// Reversed score order, making `BinaryHeap` a min-heap on score.
struct HeapEntry(SegmentScoredDocument);

impl PartialEq for HeapEntry {
    fn eq(&self, other: &Self) -> bool {
        self.cmp(other).is_eq()
    }
}

impl Eq for HeapEntry {}

impl PartialOrd for HeapEntry {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for HeapEntry {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        other.0.score.total_cmp(&self.0.score)
    }
}
