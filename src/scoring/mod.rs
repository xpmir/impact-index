//! Scoring module for applying scoring functions (e.g., BM25) to sparse indices.
//!
//! Provides [`ScoredIndex`] which wraps any [`SparseIndex`] and transforms raw
//! posting values into scores at query time. WAND and MaxScore need no changes
//! — they operate through the standard [`BlockTermImpactIterator`] interface.

pub mod bm25;

use std::cell::Cell;
use std::sync::Arc;

use crate::base::{DocId, ImpactValue, Len, TermImpact, TermIndex};
use crate::docmeta::DocMetadata;
use crate::index::{
    AsSparseIndexView, BlockTermImpactIterator, SparseIndex, SparseIndexInformation,
    SparseIndexView,
};

/// A per-term scoring function that transforms raw posting values into scores.
pub trait ScoringFunction: Send + Sync {
    /// Transform a raw posting value into a score.
    fn score(&self, raw_value: f32, docid: DocId) -> f32;

    /// Safe upper bound on score for the given max raw value.
    fn max_score(&self, max_raw_value: f32) -> f32;
}

/// A scoring model that creates per-term scoring functions.
///
/// Implementations (e.g., BM25) configure collection-level parameters
/// and produce per-term scorers that incorporate term-level statistics.
pub trait ScoringModel: Send + Sync {
    /// Initialize with collection-level statistics.
    fn initialize(&mut self, doc_lengths: Arc<Vec<u32>>, num_docs: u64);

    /// Create a per-term scorer given term-level statistics.
    ///
    /// - `df`: document frequency (number of documents containing the term)
    /// - `max_value`: maximum raw value for this term
    fn term_scorer(&self, df: u64, max_value: f32) -> Box<dyn ScoringFunction>;
}

/// Wraps a [`BlockTermImpactIterator`], applying a [`ScoringFunction`] to each posting.
struct ScoringBlockIterator<'a> {
    inner: Box<dyn BlockTermImpactIterator + 'a>,
    scorer: Box<dyn ScoringFunction>,
    /// Cached current impact after scoring (Cell: zero-cost for Copy types)
    current_impact: Cell<Option<TermImpact>>,
    /// The scored max_value for this term
    max_value: f32,
    /// Cached block metadata (refreshed once per next_min_doc_id, avoids inner vtable calls)
    cached_block_min_doc_id: DocId,
    cached_block_max_doc_id: DocId,
    cached_scored_block_max: ImpactValue,
    /// Global constants (cached once at creation)
    cached_max_doc_id: DocId,
    cached_length: usize,
}

impl<'a> ScoringBlockIterator<'a> {
    /// Refresh cached block metadata from the inner iterator.
    #[inline]
    fn refresh_block_cache(&mut self) {
        let raw_block_max = self.inner.max_block_value();
        self.cached_scored_block_max = self.scorer.max_score(raw_block_max);
        self.cached_block_min_doc_id = self.inner.min_block_doc_id();
        self.cached_block_max_doc_id = self.inner.max_block_doc_id();
    }
}

impl<'a> BlockTermImpactIterator for ScoringBlockIterator<'a> {
    fn next_min_doc_id(&mut self, doc_id: DocId) -> Option<DocId> {
        self.current_impact.set(None);
        let result = self.inner.next_min_doc_id(doc_id);
        if result.is_some() {
            self.refresh_block_cache();
        }
        result
    }

    fn current(&self) -> TermImpact {
        if let Some(impact) = self.current_impact.get() {
            return impact;
        }

        let raw = self.inner.current();
        let scored = TermImpact {
            docid: raw.docid,
            value: self.scorer.score(raw.value, raw.docid),
        };
        self.current_impact.set(Some(scored));
        scored
    }

    #[inline]
    fn max_value(&self) -> ImpactValue {
        self.max_value
    }

    #[inline]
    fn max_doc_id(&self) -> DocId {
        self.cached_max_doc_id
    }

    #[inline]
    fn max_block_value(&self) -> ImpactValue {
        self.cached_scored_block_max
    }

    #[inline]
    fn max_block_doc_id(&self) -> DocId {
        self.cached_block_max_doc_id
    }

    #[inline]
    fn min_block_doc_id(&self) -> DocId {
        self.cached_block_min_doc_id
    }

    #[inline]
    fn length(&self) -> usize {
        self.cached_length
    }
}

/// A wrapper around a [`SparseIndex`] that applies scoring functions to iterators.
///
/// Created via [`ScoredIndex::new`]. The resulting index can be searched with
/// WAND or MaxScore without any changes to the search algorithms.
pub struct ScoredIndex {
    inner: Arc<Box<dyn SparseIndex>>,
    doc_meta: Arc<DocMetadata>,
    model: Box<dyn ScoringModel>,
}

impl ScoredIndex {
    /// Create a new scored index.
    ///
    /// Collection statistics (N, avgdl, min_dl) are computed from the doc metadata
    /// and the inner index at creation time.
    pub fn new(
        inner: Arc<Box<dyn SparseIndex>>,
        doc_meta: Arc<DocMetadata>,
        mut model: Box<dyn ScoringModel>,
    ) -> Self {
        let num_docs = doc_meta.num_docs();
        let doc_lengths = Arc::new(doc_meta.doc_lengths.clone());
        model.initialize(doc_lengths, num_docs);
        Self {
            inner,
            doc_meta,
            model,
        }
    }
}

impl SparseIndex for ScoredIndex {
    fn block_iterator(&self, term_ix: TermIndex) -> Box<dyn BlockTermImpactIterator + '_> {
        if term_ix >= self.inner.len() {
            let inner_iter = self.inner.block_iterator(term_ix);
            let scorer = self.model.term_scorer(0, 0.0);
            return Box::new(ScoringBlockIterator {
                cached_max_doc_id: inner_iter.max_doc_id(),
                cached_length: inner_iter.length(),
                inner: inner_iter,
                max_value: 0.0,
                current_impact: Cell::new(None),
                cached_block_min_doc_id: 0,
                cached_block_max_doc_id: 0,
                cached_scored_block_max: 0.0,
                scorer,
            });
        }

        let inner_iter = self.inner.block_iterator(term_ix);
        let (_, max_raw_value) = SparseIndexInformation::value_range(&**self.inner, term_ix);
        let df = inner_iter.length() as u64;
        let scorer = self.model.term_scorer(df, max_raw_value);
        let max_value = scorer.max_score(max_raw_value);

        Box::new(ScoringBlockIterator {
            cached_max_doc_id: inner_iter.max_doc_id(),
            cached_length: inner_iter.length(),
            inner: inner_iter,
            scorer,
            current_impact: Cell::new(None),
            cached_block_min_doc_id: 0,
            cached_block_max_doc_id: 0,
            cached_scored_block_max: 0.0,
            max_value,
        })
    }

    fn max_doc_id(&self) -> DocId {
        SparseIndex::max_doc_id(&**self.inner)
    }
}

impl Len for ScoredIndex {
    fn len(&self) -> usize {
        self.inner.len()
    }
}

impl SparseIndexInformation for ScoredIndex {
    fn value_range(&self, term_ix: TermIndex) -> (ImpactValue, ImpactValue) {
        let (_, max_raw) = SparseIndexInformation::value_range(&**self.inner, term_ix);
        let df = if term_ix < self.inner.len() {
            self.inner.block_iterator(term_ix).length() as u64
        } else {
            0
        };
        let scorer = self.model.term_scorer(df, max_raw);
        (0.0, scorer.max_score(max_raw))
    }
}
