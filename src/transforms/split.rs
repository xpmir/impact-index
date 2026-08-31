//! Splits posting lists by impact value quantiles.
//!
//! Given quantiles (e.g., `[0.9]`), each term's postings are partitioned into
//! ranges (e.g., low 90% and top 10% by impact value). This enables the
//! MaxScore algorithm to skip low-impact postings more aggressively.

use std::cell::RefCell;
use std::cmp::Ordering;
use std::collections::BinaryHeap;
use std::fs::create_dir;
use std::path::Path;
use std::sync::Mutex;

use crate::base::{
    load_index, save_index, DocId, ImpactValue, IndexLoader, Len, TermImpact, TermIndex,
};
use crate::index::{SparseIndexInformation, SparseIndexView};
use crate::{
    index::{BlockTermImpactIterator, SparseIndex},
    transforms::IndexTransform,
};

use serde::{Deserialize, Serialize};

/// Splits each term's posting list by impact value quantiles, then delegates
/// to a downstream [`IndexTransform`] (typically compression).
///
/// For example, with `quantiles = [0.9]`, each term gets two sub-lists:
/// one with the bottom 90% of impacts and one with the top 10%.
pub struct SplitIndexTransform {
    /// The downstream transform applied to the split index.
    pub sink: Box<dyn IndexTransform>,
    /// Quantile boundaries for splitting (values in `[0, 1]`).
    pub quantiles: Vec<f64>,
}

impl IndexTransform for SplitIndexTransform {
    fn process(
        &self,
        path: &std::path::Path,
        index: &dyn SparseIndexView,
    ) -> Result<(), std::io::Error> {
        if !path.is_dir() {
            create_dir(path)?;
        }
        let inner_path = path.join("inner");

        let split_view = SplitIndexView::new(index, &self.quantiles);
        self.sink.process(inner_path.as_path(), &split_view)?;

        let index = SplitIndexLoader {
            splits: self.quantiles.len() + 1,
        };

        // The inner (compressed) index writes its own manifest inside
        // `inner/`; this one describes the split wrapper itself.
        let builder_info = crate::manifest::BuilderInfo::new()
            .with_codecs(format!("quantiles={:?}", self.quantiles));
        crate::manifest::write_manifest(path, crate::manifest::IndexKind::Split, builder_info)?;

        save_index(Box::new(index), path)
    }
}

struct SplitIndex {
    /// Inner index that contains the postings
    inner: Box<dyn SparseIndex>,

    /// Number of split per term
    splits: usize,
}

#[derive(Serialize, Deserialize)]
struct SplitIndexLoader {
    splits: usize,
}

#[typetag::serde]
impl IndexLoader for SplitIndexLoader {
    fn into_index(self: Box<Self>, path: &Path, in_memory: bool) -> Box<dyn SparseIndex> {
        let inner = load_index(&path.join("inner"), in_memory);
        Box::new(SplitIndex {
            inner,
            splits: self.splits,
        })
    }
}

struct SplitIndexTermIteratorHeapValue {
    /// Index of the iterator
    index: usize,

    /// Term impact if loaded
    term_impact: Option<TermImpact>,

    /// Minimum document ID (>= term_impact doc ID)
    min_doc_id: DocId,
}

impl Eq for SplitIndexTermIteratorHeapValue {}

impl PartialEq for SplitIndexTermIteratorHeapValue {
    fn eq(&self, other: &Self) -> bool {
        self.min_doc_id == other.min_doc_id
    }
}

impl PartialOrd for SplitIndexTermIteratorHeapValue {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        other.min_doc_id.partial_cmp(&self.min_doc_id)
    }
}
impl Ord for SplitIndexTermIteratorHeapValue {
    fn cmp(&self, other: &Self) -> Ordering {
        other.min_doc_id.partial_cmp(&self.min_doc_id).expect("")
    }
}

struct SplitIndexTermIteratorState<'a> {
    /// List of iterators
    iterators: Vec<Box<dyn BlockTermImpactIterator + 'a>>,

    /// Current posting
    current: BinaryHeap<SplitIndexTermIteratorHeapValue>,

    /// Current requested minimum doc ID
    min_doc_id: DocId,
}

/// Merge the iterators
struct SplitIndexTermIterator<'a> {
    /// Maximum impact value
    max_value: ImpactValue,

    /// Delta
    delta_value: ImpactValue,

    /// State
    state: RefCell<SplitIndexTermIteratorState<'a>>,
}

impl<'a> SplitIndexTermIterator<'a> {}

impl<'a> BlockTermImpactIterator for SplitIndexTermIterator<'a> {
    fn next_min_doc_id(&mut self, doc_id: DocId) -> Option<DocId> {
        let state = &mut *self.state.borrow_mut();

        // No next document
        if (state.current.len() == 0) && (state.iterators.len() == 0) {
            return None;
        }

        if state.current.len() == 0 {
            // First run, we initialize all the iterators
            for (ix, it) in state.iterators.iter_mut().enumerate() {
                if let Some(block_min_doc_id) = it.next_min_doc_id(doc_id) {
                    state.current.push(SplitIndexTermIteratorHeapValue {
                        index: ix,
                        term_impact: None,
                        min_doc_id: block_min_doc_id,
                    })
                }
            }
            assert!(state.min_doc_id == 0);
            state.min_doc_id = doc_id;
            if state.current.is_empty() {
                return None;
            }
            return Some(doc_id);
        } else {
            // Try to get at least one iterator
            state.min_doc_id = doc_id.max(state.min_doc_id + 1);
            while !state.current.is_empty() {
                if state.current.peek().expect("").min_doc_id >= state.min_doc_id {
                    // Found one
                    return Some(state.min_doc_id);
                }

                // Try to move
                let mut value = state.current.pop().expect("");
                if let Some(block_min_doc_id) =
                    state.iterators[value.index].next_min_doc_id(state.min_doc_id)
                {
                    value.min_doc_id = block_min_doc_id;
                    value.term_impact = None;
                    state.current.push(value);
                    return Some(state.min_doc_id);
                }
            }
        }

        None
    }

    fn current(&self) -> TermImpact {
        let state = &mut *self.state.borrow_mut();

        loop {
            // If the top of the heap is valid, return it
            if let Some(term_impact) = state
                .current
                .peek()
                .expect("No current element")
                .term_impact
            {
                state.min_doc_id = term_impact.docid;
                return term_impact;
            }

            // Otherwise, let's retrieve the element
            let mut element = state.current.pop().expect("No current element");
            if element.min_doc_id < state.min_doc_id {
                if state.iterators[element.index]
                    .next_min_doc_id(state.min_doc_id)
                    .is_none()
                {
                    // Remove this iterator
                    continue;
                }
            }

            // Get the current element
            let term_impact = state.iterators[element.index].current();
            element.min_doc_id = term_impact.docid;
            let mut posting = state.iterators[element.index].current();
            posting.value = posting.value.min(self.max_value) + self.delta_value;
            element.term_impact = Some(posting);
            state.current.push(element);
        }
    }

    fn max_value(&self) -> ImpactValue {
        self.max_value + self.delta_value
    }

    fn max_doc_id(&self) -> crate::base::DocId {
        self.state
            .borrow()
            .iterators
            .iter()
            .fold(0, |p, it| p.max(it.max_doc_id()))
    }

    fn length(&self) -> usize {
        self.state
            .borrow()
            .iterators
            .iter()
            .fold(0, |p, it| p + it.length())
    }
}

impl SparseIndex for SplitIndex {
    fn block_iterator(
        &self,
        term_ix: crate::base::TermIndex,
    ) -> Box<dyn BlockTermImpactIterator + '_> {
        // Creates an iterator that merge all the posting lists
        let mut iterators = Vec::new();
        for j in 0..self.splits {
            iterators.push(self.inner.block_iterator(term_ix * self.splits + j));
        }

        let max_value = iterators.last().expect("").max_value();
        Box::new(SplitIndexTermIterator {
            state: RefCell::new(SplitIndexTermIteratorState {
                iterators: iterators,
                current: BinaryHeap::new(),
                min_doc_id: 0,
            }),
            max_value,
            delta_value: 0.,
        })
    }

    fn block_iterators(&self, term_ix: TermIndex) -> Vec<Box<dyn BlockTermImpactIterator + '_>> {
        let mut v: Vec<Box<dyn BlockTermImpactIterator>> = Vec::new();

        // Create the different iterators
        let mut delta_value = 0.;
        for i in 0..self.splits {
            // Iterators {i, ..., n-1}
            let mut iterators = Vec::new();
            for j in i..self.splits {
                iterators.push(self.inner.block_iterator(term_ix * self.splits + j));
            }

            // The maximum impact is bounded by the first iterator
            let max_value = iterators[0].max_value();

            v.push(Box::new(SplitIndexTermIterator {
                state: RefCell::new(SplitIndexTermIteratorState {
                    iterators: iterators,
                    current: BinaryHeap::new(),
                    min_doc_id: 0,
                }),
                max_value,
                delta_value,
            }));

            delta_value = -max_value;
        }
        v
    }

    fn max_doc_id(&self) -> DocId {
        SparseIndex::max_doc_id(&*self.inner)
    }

    fn as_any(&self) -> &dyn std::any::Any {
        self
    }
}

impl Len for SplitIndex {
    fn len(&self) -> usize {
        self.inner.len() / self.splits
    }
}

impl SparseIndexInformation for SplitIndex {
    fn value_range(&self, term_ix: TermIndex) -> (ImpactValue, ImpactValue) {
        return (
            self.inner.value_range(term_ix * self.splits - 1).0,
            self.inner.value_range((term_ix + 1) * self.splits - 1).1,
        );
    }
}

/// View on the index
struct SplitIndexView<'a> {
    /// Inner index that contains the postings
    source: &'a dyn SparseIndexView,

    /// Split quantiles (just one value if using two posting lists per term)
    quantiles: &'a Vec<f64>,

    /// The (cached) split values
    thresholds: Mutex<Vec<Vec<ImpactValue>>>,
}

impl<'a> SplitIndexView<'a> {
    pub fn new(source: &'a dyn SparseIndexView, quantiles: &'a Vec<f64>) -> Self {
        let mut thresholds = Vec::new();
        for _ in 0..source.len() {
            thresholds.push(Vec::new());
        }

        Self {
            source: source,
            quantiles: quantiles,
            thresholds: Mutex::new(thresholds),
        }
    }
}

struct SplitIndexViewIterator<'a> {
    iterator: Box<dyn Iterator<Item = TermImpact> + 'a>,
    min: ImpactValue,
    max: ImpactValue,
}

impl<'a> Iterator for SplitIndexViewIterator<'a> {
    type Item = TermImpact;

    fn next(&mut self) -> Option<Self::Item> {
        while let Some(posting) = self.iterator.next() {
            if (posting.value >= self.min) && (posting.value < self.max) {
                return Some(TermImpact {
                    docid: posting.docid,
                    value: posting.value,
                });
            }
        }
        None
    }
}

impl<'a> SplitIndexView<'a> {
    fn compute_threshold(&self, term_ix: TermIndex) {
        // Source term and quantile indices
        let source_term_ix = term_ix / (self.quantiles.len() + 1);

        let thresholds = &mut self.thresholds.lock().unwrap();
        let term_thresholds = &mut thresholds[source_term_ix];

        // Computes the term threshold if not in cache
        if term_thresholds.len() == 0 {
            let mut values: Vec<ImpactValue> = self
                .source
                .iterator(source_term_ix)
                .map(|posting| posting.value)
                .collect();
            values.sort_by(|a, b| a.partial_cmp(b).unwrap());

            if values.is_empty() {
                term_thresholds.push(0.);
            } else {
                term_thresholds.push(values[0]);
            }
            for q in self.quantiles {
                let ix = (q * values.len() as f64).trunc() as usize;
                let mut threshold = 0.;
                if ix < values.len() {
                    threshold = values[ix];
                }
                term_thresholds.push(threshold);
            }
            term_thresholds.push(ImpactValue::INFINITY);
        }
    }
}

impl<'a> SparseIndexView for SplitIndexView<'a> {
    fn iterator<'b>(&'b self, term_ix: TermIndex) -> Box<dyn Iterator<Item = TermImpact> + 'b> {
        // Source term and quantile indices
        self.compute_threshold(term_ix);
        let source_term_ix = term_ix / (self.quantiles.len() + 1);
        let quantile_ix = term_ix % (self.quantiles.len() + 1);

        // Returns the iterator
        let thresholds = &mut self.thresholds.lock().unwrap();
        let term_thresholds = &mut thresholds[source_term_ix];
        Box::new(SplitIndexViewIterator {
            iterator: self.source.iterator(source_term_ix),
            min: term_thresholds[quantile_ix],
            max: term_thresholds[quantile_ix + 1],
        })
    }

    fn max_doc_id(&self) -> DocId {
        self.source.max_doc_id()
    }
}

impl<'a> Len for SplitIndexView<'a> {
    fn len(&self) -> usize {
        self.source.len() * (self.quantiles.len() + 1)
    }
}

impl<'a> SparseIndexInformation for SplitIndexView<'a> {
    fn value_range(&self, term_ix: TermIndex) -> (ImpactValue, ImpactValue) {
        // Computes the
        self.compute_threshold(term_ix);
        let source_term_ix = term_ix / (self.quantiles.len() + 1);
        let quantile_ix = term_ix % (self.quantiles.len() + 1);

        let thresholds = &mut self.thresholds.lock().unwrap();
        let term_thresholds = &mut thresholds[source_term_ix];
        (term_thresholds[0], term_thresholds[quantile_ix])
    }
}
