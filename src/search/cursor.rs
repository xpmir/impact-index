//! Monomorphizable cursor abstraction for the search hot path (P3).
//!
//! [`BlockTermImpactIterator`] is object-safe (`current(&self)`), which is
//! what lets it live behind `Box<dyn _>` everywhere else in the codebase.
//! That's also exactly what defeats devirtualization in the search loops:
//! every posting pays a vtable call through `ScoringBlockIterator` into the
//! inner iterator, plus another vtable call for `ScoringFunction::score`.
//!
//! [`TermCursor`] mirrors the same shallow-move/lazy-current contract but
//! takes `&mut self` throughout. That's a strictly *more* capable interface
//! (concrete cursors don't need `Cell`/`RefCell` for interior mutability
//! any more) and, crucially, one that a generic `fn search_maxscore_core<C:
//! TermCursor>(...)` can call with fully static dispatch: when `C` is a
//! concrete type such as the compressed-index + BM25 cursor, LLVM inlines
//! `next_min_doc_id`/`current` straight through, batched scoring (P1b) and
//! all. A blanket impl over `Box<dyn BlockTermImpactIterator>` keeps every
//! existing (index, scorer) combination working unchanged as the fallback
//! path.
//!
//! `search_maxscore`/`search_wand` pick between the two paths with a single
//! downcast at query entry (see [`as_bm25_compressed`]) — zero per-posting
//! dynamic dispatch either way, just two different concrete shapes for the
//! search loop.

use crate::base::{DocId, ImpactValue, TermImpact};
use crate::compress::CompressedIndex;
use crate::index::{BlockTermImpactIterator, SparseIndex};
use crate::scoring::bm25::BM25Scoring;
use crate::scoring::ScoredIndex;

/// A statically-dispatchable cursor over one term's scored postings.
///
/// Same shallow-move contract as [`BlockTermImpactIterator`]:
/// `next_min_doc_id` may only pick the right block, leaving the exact
/// posting to be resolved lazily by `current`.
pub trait TermCursor {
    /// Moves to the first block containing a doc id >= `min_doc_id`.
    /// Returns a lower bound on the next resolvable doc id.
    fn next_min_doc_id(&mut self, min_doc_id: DocId) -> Option<DocId>;

    /// Resolves (or returns the cached) current posting.
    fn current(&mut self) -> TermImpact;

    /// Term-level maximum score.
    fn max_value(&self) -> ImpactValue;

    /// Maximum document ID over all postings for this term.
    fn max_doc_id(&self) -> DocId;

    /// Maximum score within the current block.
    fn max_block_value(&self) -> ImpactValue;

    /// Maximum document ID within the current block.
    fn max_block_doc_id(&self) -> DocId;

    /// Minimum document ID within the current block.
    fn min_block_doc_id(&self) -> DocId;

    /// Total number of postings for this term.
    fn length(&self) -> usize;

    /// Advances to the very next posting (equivalent to `next_min_doc_id(0)`
    /// followed by `current()`), mirroring the old `Iterator::next()` used
    /// by the dyn path.
    #[inline]
    fn advance(&mut self) -> Option<TermImpact> {
        if self.next_min_doc_id(0).is_some() {
            Some(self.current())
        } else {
            None
        }
    }
}

/// Fallback: any existing `Box<dyn BlockTermImpactIterator>` is a valid
/// (dynamically-dispatched) `TermCursor`. This is what keeps every
/// (index, scorer) combination other than the monomorphized fast path
/// working exactly as before.
impl<'a> TermCursor for Box<dyn BlockTermImpactIterator + 'a> {
    #[inline]
    fn next_min_doc_id(&mut self, min_doc_id: DocId) -> Option<DocId> {
        (**self).next_min_doc_id(min_doc_id)
    }

    #[inline]
    fn current(&mut self) -> TermImpact {
        (**self).current()
    }

    #[inline]
    fn max_value(&self) -> ImpactValue {
        (**self).max_value()
    }

    #[inline]
    fn max_doc_id(&self) -> DocId {
        (**self).max_doc_id()
    }

    #[inline]
    fn max_block_value(&self) -> ImpactValue {
        (**self).max_block_value()
    }

    #[inline]
    fn max_block_doc_id(&self) -> DocId {
        (**self).max_block_doc_id()
    }

    #[inline]
    fn min_block_doc_id(&self) -> DocId {
        (**self).min_block_doc_id()
    }

    #[inline]
    fn length(&self) -> usize {
        (**self).length()
    }
}

/// Detects the one (index, scorer) combination monomorphized so far: a
/// [`ScoredIndex`] wrapping a [`CompressedIndex`] with [`BM25Scoring`].
///
/// Returns `None` for any other combination (raw index, split index, a
/// different scoring model, ...), in which case callers must fall back to
/// the generic `dyn BlockTermImpactIterator` path. This is the *only*
/// dynamic decision paid per query.
pub(crate) fn as_bm25_compressed<'a>(
    index: &'a dyn SparseIndex,
) -> Option<(&'a CompressedIndex, &'a BM25Scoring)> {
    let scored = index.as_any().downcast_ref::<ScoredIndex>()?;
    let compressed = scored
        .inner_index()
        .as_any()
        .downcast_ref::<CompressedIndex>()?;
    let bm25 = scored.model_any().downcast_ref::<BM25Scoring>()?;
    Some((compressed, bm25))
}
