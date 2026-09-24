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

/// Manifest feature marking split indices whose inner index was built with
/// correct per-split value ranges. Older split indices built with a
/// quantizing impact compressor had their top impacts clamped (the
/// quantizer's upper bound was the highest split *threshold*, not the
/// highest impact) and must be rebuilt.
pub const SPLIT_VALUE_RANGE_FEATURE: &str = "split-value-range";

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

// Split indices drop positions by design (v1, positions-plan.md Phase 2
// §2.4): `SplitIndexView` below doesn't override `SparseIndexView::has_positions`/
// `positions_iterator`, so it inherits the trait's `false`/`None` defaults
// regardless of whether the source has positions -- no explicit guard
// needed, `has_positions()` on the resulting index is just always `false`.

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
        crate::manifest::write_manifest_with_features(
            path,
            crate::manifest::IndexKind::Split,
            builder_info,
            vec![SPLIT_VALUE_RANGE_FEATURE.to_string()],
        )?;

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

/// Warns when loading a split index built before per-split value ranges
/// were fixed, with a quantizing impact compressor: its highest impacts
/// were clamped at build time, so search results are degraded.
fn warn_if_clamped_split_index(path: &Path) {
    let has_feature = match crate::manifest::read_manifest(path) {
        Ok(Some(m)) => m.features.iter().any(|f| f == SPLIT_VALUE_RANGE_FEATURE),
        _ => false,
    };
    if has_feature {
        return;
    }
    let quantized = match crate::manifest::read_manifest(&path.join("inner")) {
        Ok(Some(m)) => m.builder.codecs.map_or(false, |c| c.contains("Quantize")),
        _ => false,
    };
    if quantized {
        log::warn!(
            "Split index at {} was built with a quantizing impact compressor by a \
             version with a bug that clamped the highest impacts: search results \
             are degraded. Please rebuild the index (SplitIndexTransform.process).",
            path.display()
        );
    }
}

#[typetag::serde]
impl IndexLoader for SplitIndexLoader {
    fn into_index(self: Box<Self>, path: &Path, in_memory: bool) -> Box<dyn SparseIndex> {
        warn_if_clamped_split_index(path);
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

    /// Posting (value already transformed) if loaded
    term_impact: Option<TermImpact>,

    /// Lower bound on the doc ID of the sub-iterator's next posting; this
    /// is the posting's doc ID once `term_impact` is loaded
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
        Some(self.cmp(other))
    }
}

impl Ord for SplitIndexTermIteratorHeapValue {
    fn cmp(&self, other: &Self) -> Ordering {
        // Reversed: BinaryHeap is a max-heap, we want the smallest doc ID
        other.min_doc_id.cmp(&self.min_doc_id)
    }
}

struct SplitIndexTermIteratorState<'a> {
    /// List of iterators
    iterators: Vec<Box<dyn BlockTermImpactIterator + 'a>>,

    /// Non-exhausted sub-iterators, ordered by (a lower bound of) their
    /// next doc ID.
    ///
    /// Invariant after `next_min_doc_id`: every entry has
    /// `min_doc_id >= target`, so no entry holds a stale posting.
    current: BinaryHeap<SplitIndexTermIteratorHeapValue>,

    /// Whether the sub-iterators have been positioned once
    initialized: bool,

    /// Current requested minimum doc ID
    target: DocId,

    /// Doc ID of the last posting returned by `current()`
    last_doc_id: Option<DocId>,
}

/// Merges the (doc ID disjoint) posting lists of several splits of a term.
///
/// Returned values are `min(value, max_value) + delta_value`, which is how
/// [`SplitIndex::block_iterators`] decomposes a term impact into a sum of
/// per-split contributions.
struct SplitIndexTermIterator<'a> {
    /// Maximum impact value
    max_value: ImpactValue,

    /// Delta
    delta_value: ImpactValue,

    /// State
    state: RefCell<SplitIndexTermIteratorState<'a>>,
}

impl<'a> SplitIndexTermIterator<'a> {
    fn new(
        iterators: Vec<Box<dyn BlockTermImpactIterator + 'a>>,
        max_value: ImpactValue,
        delta_value: ImpactValue,
    ) -> Self {
        Self {
            max_value,
            delta_value,
            state: RefCell::new(SplitIndexTermIteratorState {
                iterators,
                current: BinaryHeap::new(),
                initialized: false,
                target: 0,
                last_doc_id: None,
            }),
        }
    }
}

impl<'a> BlockTermImpactIterator for SplitIndexTermIterator<'a> {
    fn next_min_doc_id(&mut self, doc_id: DocId) -> Option<DocId> {
        let state = &mut *self.state.borrow_mut();

        // Never go back, and move past the last returned posting
        let mut target = doc_id.max(state.target);
        if let Some(last) = state.last_doc_id {
            target = target.max(last + 1);
        }
        state.target = target;

        if !state.initialized {
            state.initialized = true;
            for (ix, it) in state.iterators.iter_mut().enumerate() {
                if let Some(block_min_doc_id) = it.next_min_doc_id(target) {
                    state.current.push(SplitIndexTermIteratorHeapValue {
                        index: ix,
                        term_impact: None,
                        min_doc_id: block_min_doc_id.max(target),
                    })
                }
            }
        } else {
            // Move *every* sub-iterator lagging behind the target, not only
            // the top one: otherwise a stale posting (doc ID < target) held
            // by another sub-iterator could be returned by `current()`.
            while let Some(top) = state.current.peek() {
                if top.min_doc_id >= target {
                    break;
                }
                let mut value = state.current.pop().expect("heap is not empty");
                if let Some(block_min_doc_id) = state.iterators[value.index].next_min_doc_id(target)
                {
                    // The sub-iterator's next posting is >= target, even if
                    // its block starts before
                    value.min_doc_id = block_min_doc_id.max(target);
                    value.term_impact = None;
                    state.current.push(value);
                }
            }
        }

        state.current.peek().map(|top| top.min_doc_id)
    }

    fn current(&self) -> TermImpact {
        let state = &mut *self.state.borrow_mut();

        loop {
            let top = state.current.peek().expect("No current element");

            // If the top of the heap is loaded, return it
            if let Some(term_impact) = top.term_impact {
                state.last_doc_id = Some(term_impact.docid);
                return term_impact;
            }

            // Otherwise, load the posting (the sub-iterator is positioned
            // at a doc ID >= target, see `next_min_doc_id`)
            let mut element = state.current.pop().expect("No current element");
            let mut posting = state.iterators[element.index].current();
            element.min_doc_id = posting.docid;
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
    fn reorder_map(&self) -> Option<&Vec<DocId>> {
        self.inner.reorder_map()
    }

    fn block_iterator(
        &self,
        term_ix: crate::base::TermIndex,
    ) -> Box<dyn BlockTermImpactIterator + '_> {
        // Creates an iterator that merge all the posting lists
        let mut iterators = Vec::new();
        for j in 0..self.splits {
            iterators.push(self.inner.block_iterator(term_ix * self.splits + j));
        }

        // Maximum over all the splits
        let max_value = iterators
            .iter()
            .fold(0., |m: ImpactValue, it| m.max(it.max_value()));
        Box::new(SplitIndexTermIterator::new(iterators, max_value, 0.))
    }

    /// Decomposes the impact `v` of a posting in split `j` into a sum of
    /// contributions `c_i = min(v, M_i) - M_{i-1}` over iterators
    /// `i = 0..=j`, where iterator `i` covers splits `i..n`, `M_i` is the
    /// maximum impact of splits `0..=i` and `M_{-1} = 0`. Since splits are
    /// ordered by impact, `v >= M_i` for `i < j` and the sum telescopes to
    /// `min(v, M_j) = v`; the upper bound of iterator `i` is
    /// `M_i - M_{i-1} >= 0`.
    fn block_iterators(&self, term_ix: TermIndex) -> Vec<Box<dyn BlockTermImpactIterator + '_>> {
        let mut v: Vec<Box<dyn BlockTermImpactIterator>> = Vec::new();

        let mut previous_max: ImpactValue = 0.;
        for i in 0..self.splits {
            // Iterators {i, ..., n-1}
            let mut iterators = Vec::new();
            for j in i..self.splits {
                iterators.push(self.inner.block_iterator(term_ix * self.splits + j));
            }

            // Running maximum (an empty split must not lower the bound)
            let max_value = previous_max.max(iterators[0].max_value());

            v.push(Box::new(SplitIndexTermIterator::new(
                iterators,
                max_value,
                -previous_max,
            )));

            previous_max = max_value;
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
        let mut min = ImpactValue::INFINITY;
        let mut max: ImpactValue = 0.;
        for j in 0..self.splits {
            let (lo, hi) = self.inner.value_range(term_ix * self.splits + j);
            min = min.min(lo);
            max = max.max(hi);
        }
        (min.min(max), max)
    }
}

/// Per-term split information (computed lazily)
struct TermSplits {
    /// Split boundaries: split `q` holds values in
    /// `[thresholds[q], thresholds[q + 1])`
    thresholds: Vec<ImpactValue>,

    /// Maximum value within each split (0 if empty)
    max_values: Vec<ImpactValue>,
}

/// View on the index
struct SplitIndexView<'a> {
    /// Inner index that contains the postings
    source: &'a dyn SparseIndexView,

    /// Split quantiles (just one value if using two posting lists per term)
    quantiles: &'a Vec<f64>,

    /// The (cached) split values
    splits: Mutex<Vec<Option<TermSplits>>>,
}

impl<'a> SplitIndexView<'a> {
    pub fn new(source: &'a dyn SparseIndexView, quantiles: &'a Vec<f64>) -> Self {
        let mut splits = Vec::new();
        splits.resize_with(source.len(), || None);

        Self {
            source: source,
            quantiles: quantiles,
            splits: Mutex::new(splits),
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
    fn compute_splits(&self, values: &mut Vec<ImpactValue>) -> TermSplits {
        values.sort_by(|a, b| a.partial_cmp(b).unwrap());

        let mut thresholds = Vec::with_capacity(self.quantiles.len() + 2);
        thresholds.push(values.first().copied().unwrap_or(0.));
        for q in self.quantiles {
            let ix = (q * values.len() as f64).trunc() as usize;
            // Past the end (quantile 1): empty split
            let threshold = values.get(ix).copied().unwrap_or(ImpactValue::INFINITY);
            // Keep thresholds non-decreasing so that splits are disjoint
            thresholds.push(threshold.max(*thresholds.last().unwrap()));
        }
        thresholds.push(ImpactValue::INFINITY);

        let max_values = thresholds
            .windows(2)
            .map(|w| {
                // Largest value in [w[0], w[1])
                let end = values.partition_point(|&v| v < w[1]);
                match end.checked_sub(1).map(|i| values[i]) {
                    Some(v) if v >= w[0] => v,
                    _ => 0.,
                }
            })
            .collect();

        TermSplits {
            thresholds,
            max_values,
        }
    }

    /// Returns `(min, max, split max)` for the split term `term_ix`
    fn split_info(&self, term_ix: TermIndex) -> (ImpactValue, ImpactValue, ImpactValue) {
        let source_term_ix = term_ix / (self.quantiles.len() + 1);
        let quantile_ix = term_ix % (self.quantiles.len() + 1);

        let splits = &mut self.splits.lock().unwrap();
        let term_splits = splits[source_term_ix].get_or_insert_with(|| {
            let mut values: Vec<ImpactValue> = self
                .source
                .iterator(source_term_ix)
                .map(|posting| posting.value)
                .collect();
            self.compute_splits(&mut values)
        });

        (
            term_splits.thresholds[quantile_ix],
            term_splits.thresholds[quantile_ix + 1],
            term_splits.max_values[quantile_ix],
        )
    }
}

impl<'a> SparseIndexView for SplitIndexView<'a> {
    fn iterator<'b>(&'b self, term_ix: TermIndex) -> Box<dyn Iterator<Item = TermImpact> + 'b> {
        let source_term_ix = term_ix / (self.quantiles.len() + 1);
        let (min, max, _) = self.split_info(term_ix);

        Box::new(SplitIndexViewIterator {
            iterator: self.source.iterator(source_term_ix),
            min,
            max,
        })
    }

    fn max_doc_id(&self) -> DocId {
        self.source.max_doc_id()
    }

    fn doc_meta(&self) -> Option<&crate::docmeta::DocMetadata> {
        self.source.doc_meta()
    }
}

impl<'a> Len for SplitIndexView<'a> {
    fn len(&self) -> usize {
        self.source.len() * (self.quantiles.len() + 1)
    }
}

impl<'a> SparseIndexInformation for SplitIndexView<'a> {
    /// Same convention as the other indices, `(0, max)`: quantizing impact
    /// compressors derive their range from this, so the upper bound must be
    /// the actual maximum impact of the split -- not its lower threshold,
    /// which clamped every impact above the highest threshold.
    fn value_range(&self, term_ix: TermIndex) -> (ImpactValue, ImpactValue) {
        let (_, _, max) = self.split_info(term_ix);
        (0., max)
    }
}
