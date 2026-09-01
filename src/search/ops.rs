//! Structured-query operator cursors ("virtual posting lists").
//!
//! Every composite here implements [`BlockTermImpactIterator`] over one or
//! more child cursors, so [`crate::search::wand`]/[`crate::search::maxscore`]
//! run over them completely unchanged (see `src/query.rs`'s `evaluate`,
//! which builds these from a [`crate::query::QueryNode`] tree).
//!
//! ## The "stricter-but-valid" iterator contract
//!
//! [`BlockTermImpactIterator::next_min_doc_id`] is allowed to be *shallow*
//! (a lower bound, resolved lazily by `current()`) -- that's what lets a
//! block-compressed leaf cursor skip decoding. Composites below don't do
//! that: their `next_min_doc_id` always fully resolves the exact next
//! matching document (recursing into children's `next_min_doc_id` +
//! `current()`), caches the composite's own impact, and returns the exact
//! docid; `current()` just reads that cache. This is a strictly stronger
//! (and still valid) special case of the general contract -- see
//! `src/index.rs`'s trait docs and `positions-plan.md` (Phase 3).
//!
//! A direct consequence: since every `next_min_doc_id` call here already
//! resolves "the current posting", the very next call must advance *past*
//! it -- mirrored from how `CompressedBlockTermImpactIterator`
//! (`src/compress/mod.rs`) uses `has_current`/`current_docid + 1`, just
//! computed eagerly here instead of lazily.
//!
//! ## Block bounds
//!
//! None of these composites override `max_block_value`/`max_block_doc_id`/
//! `min_block_doc_id`/`min_dl`/`min_block_dl` -- they rely on the trait
//! defaults (whole-term bounds). Children's blocks don't align with each
//! other, so composing them would require real work to stay safe; v1
//! trades block-level pruning *inside* a composite for correctness
//! simplicity (documented tradeoff, see `positions-plan.md` Phase 3.3).

use crate::base::{DocId, ImpactValue, TermImpact};
use crate::index::BlockTermImpactIterator;

/// One child of a [`SumCursor`]: its own weight, plus a cache of its
/// current resolved docid (`None` = needs advancing, or the child has been
/// dropped as exhausted and this struct is about to be removed).
struct WeightedChild<'a> {
    weight: f32,
    iter: Box<dyn BlockTermImpactIterator + 'a>,
    front: Option<DocId>,
}

/// Weighted OR: merges children's postings, summing `weight_i * value_i`
/// over whichever children are present at the winning docid.
///
/// Used both for nested `#combine` (query-level weighted sum) and as the
/// raw-tf merge underneath `#syn` (all weights 1.0, see `src/query.rs`).
/// A single-weighted-child `SumCursor` also serves as a generic "scale this
/// cursor's value by a constant" wrapper (used to apply `Term.weight`
/// outside of a `Combine` context).
pub(crate) struct SumCursor<'a> {
    children: Vec<WeightedChild<'a>>,
    cached: Option<TermImpact>,
    max_value: ImpactValue,
    max_doc_id: DocId,
    length: usize,
}

impl<'a> SumCursor<'a> {
    pub(crate) fn new(children: Vec<(f32, Box<dyn BlockTermImpactIterator + 'a>)>) -> Self {
        let max_value: ImpactValue = children.iter().map(|(w, c)| w * c.max_value()).sum();
        let max_doc_id = children
            .iter()
            .map(|(_, c)| c.max_doc_id())
            .max()
            .unwrap_or(0);
        // Cost estimate: sum of children's lengths, capped at the number of
        // distinct docids that could possibly exist (`max_doc_id + 1`).
        let length = children
            .iter()
            .map(|(_, c)| c.length())
            .sum::<usize>()
            .min(max_doc_id as usize + 1);

        let children = children
            .into_iter()
            .map(|(weight, iter)| WeightedChild {
                weight,
                iter,
                front: None,
            })
            .collect();

        Self {
            children,
            cached: None,
            max_value,
            max_doc_id,
            length,
        }
    }
}

impl<'a> BlockTermImpactIterator for SumCursor<'a> {
    fn next_min_doc_id(&mut self, doc_id: DocId) -> Option<DocId> {
        let target = doc_id.max(self.cached.map(|c| c.docid + 1).unwrap_or(0));

        // Advance any child whose cached front is stale (unknown, or
        // behind `target`); drop children that turn out exhausted -- OR
        // semantics mean a spent child just stops contributing, it doesn't
        // end the composite.
        self.children.retain_mut(|child| {
            if child.front.map_or(true, |f| f < target) {
                child.front = child
                    .iter
                    .next_min_doc_id(target)
                    .map(|_| child.iter.current().docid);
            }
            child.front.is_some()
        });

        let candidate = self.children.iter().filter_map(|c| c.front).min();
        match candidate {
            Some(docid) => {
                let value = self
                    .children
                    .iter()
                    .filter(|c| c.front == Some(docid))
                    .map(|c| c.weight * c.iter.current().value)
                    .sum();
                self.cached = Some(TermImpact { docid, value });
                Some(docid)
            }
            None => {
                self.cached = None;
                None
            }
        }
    }

    fn current(&self) -> TermImpact {
        self.cached
            .expect("SumCursor::current() called before a successful next_min_doc_id")
    }

    #[inline]
    fn max_value(&self) -> ImpactValue {
        self.max_value
    }

    #[inline]
    fn max_doc_id(&self) -> DocId {
        self.max_doc_id
    }

    #[inline]
    fn length(&self) -> usize {
        self.length
    }
}

/// One child of an [`AndCursor`]/[`PositionalCursor`]: unlike
/// [`WeightedChild`], never removed -- AND semantics mean any exhausted
/// child ends the whole composite.
struct AndChild<'a> {
    iter: Box<dyn BlockTermImpactIterator + 'a>,
    front: Option<DocId>,
}

/// Leapfrogs `children` (all currently primed to `>= target`, per
/// `front`) onto a single common docid `>= target`, advancing only the
/// children that need it. Returns `None` (setting `*exhausted = true`) the
/// moment any child runs out. On success, every `children[i].front ==
/// Some(returned docid)` and `children[i].iter.current()` is the resolved
/// posting at that docid.
fn leapfrog_align<'a>(
    children: &mut [AndChild<'a>],
    mut target: DocId,
    exhausted: &mut bool,
) -> Option<DocId> {
    'align: loop {
        for child in children.iter_mut() {
            if child.front.map_or(true, |f| f < target) {
                match child.iter.next_min_doc_id(target) {
                    Some(_) => child.front = Some(child.iter.current().docid),
                    None => {
                        *exhausted = true;
                        return None;
                    }
                }
            }
            let d = child.front.expect("just resolved above");
            if d > target {
                target = d;
                continue 'align;
            }
        }
        return Some(target);
    }
}

/// Boolean AND (`#band`): matches only docs present in every child;
/// value = sum of children's values at the aligned doc (Lucene MUST
/// semantics).
pub(crate) struct AndCursor<'a> {
    children: Vec<AndChild<'a>>,
    cached: Option<TermImpact>,
    max_value: ImpactValue,
    max_doc_id: DocId,
    length: usize,
    exhausted: bool,
}

impl<'a> AndCursor<'a> {
    pub(crate) fn new(children: Vec<Box<dyn BlockTermImpactIterator + 'a>>) -> Self {
        let max_value: ImpactValue = children.iter().map(|c| c.max_value()).sum();
        let max_doc_id = children.iter().map(|c| c.max_doc_id()).min().unwrap_or(0);
        let length = children.iter().map(|c| c.length()).min().unwrap_or(0);
        let exhausted = children.is_empty();

        let children = children
            .into_iter()
            .map(|iter| AndChild { iter, front: None })
            .collect();

        Self {
            children,
            cached: None,
            max_value,
            max_doc_id,
            length,
            exhausted,
        }
    }
}

impl<'a> BlockTermImpactIterator for AndCursor<'a> {
    fn next_min_doc_id(&mut self, doc_id: DocId) -> Option<DocId> {
        if self.exhausted {
            return None;
        }
        let target = doc_id.max(self.cached.map(|c| c.docid + 1).unwrap_or(0));

        let docid = leapfrog_align(&mut self.children, target, &mut self.exhausted)?;
        let value = self.children.iter().map(|c| c.iter.current().value).sum();
        self.cached = Some(TermImpact { docid, value });
        Some(docid)
    }

    fn current(&self) -> TermImpact {
        self.cached
            .expect("AndCursor::current() called before a successful next_min_doc_id")
    }

    #[inline]
    fn max_value(&self) -> ImpactValue {
        self.max_value
    }

    #[inline]
    fn max_doc_id(&self) -> DocId {
        self.max_doc_id
    }

    #[inline]
    fn length(&self) -> usize {
        self.length
    }
}

/// Which positional match to count on an aligned document.
enum MatchKind {
    /// `#1`: exact adjacency.
    Phrase,
    /// `#uwN`: unordered window of the given width (token positions).
    Window(u32),
}

/// Shared machinery for [`PhraseCursor`]/[`WindowCursor`]: `AndCursor`-style
/// leapfrog alignment over *raw positional* children, plus a positional
/// verification step on every aligned doc. Verification happens inside
/// `next_min_doc_id` (never in `current()`), so a doc with a zero match
/// count is simply skipped -- WAND/MaxScore never see it, and never need to
/// know positional composites exist.
struct PositionalCursor<'a> {
    children: Vec<AndChild<'a>>,
    kind: MatchKind,
    cached: Option<TermImpact>,
    max_value: ImpactValue,
    max_doc_id: DocId,
    length: usize,
    exhausted: bool,
}

impl<'a> PositionalCursor<'a> {
    fn new(children: Vec<Box<dyn BlockTermImpactIterator + 'a>>, kind: MatchKind) -> Self {
        let max_value: ImpactValue = match kind {
            // Every phrase match consumes a distinct start position from the
            // pivot (rarest) list, so tf_phrase <= min child tf -- `min` of
            // children's max values is a safe (and tight) bound.
            MatchKind::Phrase => children
                .iter()
                .map(|c| c.max_value())
                .reduce(f32::min)
                .unwrap_or(0.0),
            // NOT so for windows: `window_count`'s minimal-cover sweep
            // advances one pointer per counted window, so interleaved
            // occurrences ("a b a b a", width 3) yield a count EXCEEDING
            // every single child's tf. The count is bounded by the total
            // number of pointer advances, i.e. the sum of the children's
            // tfs -- `min` here would under-estimate the bound and let
            // WAND/MaxScore prune genuinely competitive documents.
            MatchKind::Window(_) => children.iter().map(|c| c.max_value()).sum(),
        };
        let max_doc_id = children.iter().map(|c| c.max_doc_id()).min().unwrap_or(0);
        let length = children.iter().map(|c| c.length()).min().unwrap_or(0);
        let exhausted = children.is_empty();

        let children = children
            .into_iter()
            .map(|iter| AndChild { iter, front: None })
            .collect();

        Self {
            children,
            kind,
            cached: None,
            max_value,
            max_doc_id,
            length,
            exhausted,
        }
    }
}

impl<'a> BlockTermImpactIterator for PositionalCursor<'a> {
    fn next_min_doc_id(&mut self, doc_id: DocId) -> Option<DocId> {
        if self.exhausted {
            return None;
        }
        let mut target = doc_id.max(self.cached.map(|c| c.docid + 1).unwrap_or(0));

        loop {
            let docid = leapfrog_align(&mut self.children, target, &mut self.exhausted)?;

            let lists: Vec<Vec<u32>> = self
                .children
                .iter_mut()
                .map(|c| {
                    c.iter
                        .positions()
                        .expect(
                            "positional composite child has no positions -- \
                             caller must check has_positions() before constructing one",
                        )
                        .to_vec()
                })
                .collect();
            let refs: Vec<&[u32]> = lists.iter().map(|v| v.as_slice()).collect();

            let count = match self.kind {
                MatchKind::Phrase => phrase_count(&refs),
                MatchKind::Window(width) => window_count(&refs, width),
            };

            if count > 0 {
                self.cached = Some(TermImpact {
                    docid,
                    value: count as f32,
                });
                return Some(docid);
            }

            // Not a match at this doc: keep searching from the next one
            // (this is what makes verification live *inside*
            // `next_min_doc_id` -- callers never see a zero-count posting).
            target = docid + 1;
        }
    }

    fn current(&self) -> TermImpact {
        self.cached
            .expect("PositionalCursor::current() called before a successful next_min_doc_id")
    }

    #[inline]
    fn max_value(&self) -> ImpactValue {
        self.max_value
    }

    #[inline]
    fn max_doc_id(&self) -> DocId {
        self.max_doc_id
    }

    #[inline]
    fn length(&self) -> usize {
        self.length
    }
}

/// Exact phrase (`#1`): adjacent positions, one virtual "term" whose tf is
/// the phrase's occurrence count. Constructed only over *raw* (unscored)
/// positional children -- see `src/query.rs::evaluate`, which wraps the
/// result with a compound scorer.
pub(crate) struct PhraseCursor<'a>(PositionalCursor<'a>);

impl<'a> PhraseCursor<'a> {
    pub(crate) fn new(children: Vec<Box<dyn BlockTermImpactIterator + 'a>>) -> Self {
        Self(PositionalCursor::new(children, MatchKind::Phrase))
    }
}

impl<'a> BlockTermImpactIterator for PhraseCursor<'a> {
    #[inline]
    fn next_min_doc_id(&mut self, doc_id: DocId) -> Option<DocId> {
        self.0.next_min_doc_id(doc_id)
    }
    #[inline]
    fn current(&self) -> TermImpact {
        self.0.current()
    }
    #[inline]
    fn max_value(&self) -> ImpactValue {
        self.0.max_value()
    }
    #[inline]
    fn max_doc_id(&self) -> DocId {
        self.0.max_doc_id()
    }
    #[inline]
    fn length(&self) -> usize {
        self.0.length()
    }
}

/// Unordered window (`#uwN`): minimal-interval semantics, see
/// [`window_count`]. Same construction/wrapping story as [`PhraseCursor`].
pub(crate) struct WindowCursor<'a>(PositionalCursor<'a>);

impl<'a> WindowCursor<'a> {
    pub(crate) fn new(children: Vec<Box<dyn BlockTermImpactIterator + 'a>>, width: u32) -> Self {
        Self(PositionalCursor::new(children, MatchKind::Window(width)))
    }
}

impl<'a> BlockTermImpactIterator for WindowCursor<'a> {
    #[inline]
    fn next_min_doc_id(&mut self, doc_id: DocId) -> Option<DocId> {
        self.0.next_min_doc_id(doc_id)
    }
    #[inline]
    fn current(&self) -> TermImpact {
        self.0.current()
    }
    #[inline]
    fn max_value(&self) -> ImpactValue {
        self.0.max_value()
    }
    #[inline]
    fn max_doc_id(&self) -> DocId {
        self.0.max_doc_id()
    }
    #[inline]
    fn length(&self) -> usize {
        self.0.length()
    }
}

/// Counts phrase (`#1`) matches on an already-aligned document: the number
/// of start positions `p` such that `p` is in `lists[0]`, `p + 1` is in
/// `lists[1]`, ..., `p + lists.len() - 1` is in `lists[lists.len() - 1]`.
///
/// Iterates the *rarest* list (fewest candidate starts) and probes the
/// others with a pointer that only moves forward: the probe target
/// `start + i` is non-decreasing as `start` increases (the pivot list is
/// sorted ascending), so this is a single linear merge over all lists, not
/// a binary search per candidate.
///
/// e.g. for `"a b a b"` with `a` at `[0, 2]` and `b` at `[1, 3]`,
/// `phrase_count(&[&[0, 2], &[1, 3]])` is `2` (phrase "a b" matches at
/// position 0 and position 2). A stopword gap breaks adjacency: `a` at
/// `[0]`, `b` at `[2]` (a dropped token left a gap at position 1) gives
/// `phrase_count(&[&[0], &[2]]) == 0`.
pub(crate) fn phrase_count(lists: &[&[u32]]) -> u32 {
    let k = lists.len();
    if k == 0 || lists.iter().any(|l| l.is_empty()) {
        return 0;
    }
    if k == 1 {
        // Degenerate single-term "phrase": every occurrence is a match
        // (matches QueryNode::validate's "1 term degenerates to Term").
        return lists[0].len() as u32;
    }

    let pivot = lists
        .iter()
        .enumerate()
        .min_by_key(|(_, l)| l.len())
        .map(|(i, _)| i)
        .expect("k >= 1 checked above");

    let mut pointers = vec![0usize; k];
    let mut count = 0u32;

    'outer: for &p in lists[pivot] {
        // Candidate phrase start: term `pivot` sits at `start + pivot`.
        let start = match p.checked_sub(pivot as u32) {
            Some(s) => s,
            None => continue, // p < pivot: no valid start position
        };
        for i in 0..k {
            if i == pivot {
                continue;
            }
            let target = start + i as u32;
            let list = lists[i];
            while pointers[i] < list.len() && list[pointers[i]] < target {
                pointers[i] += 1;
            }
            if pointers[i] >= list.len() || list[pointers[i]] != target {
                continue 'outer;
            }
        }
        count += 1;
    }
    count
}

/// Counts unordered-window (`#uwN`) matches on an already-aligned document:
/// the number of *minimal* windows covering at least one occurrence of
/// every term with span `< width` (span = max position - min position
/// among the window's chosen occurrences).
///
/// Standard "smallest range covering all lists" two-pointer sweep: at each
/// step, look at the current frontier (one pointer per list), record
/// whether its span is a match, then advance the pointer sitting at the
/// minimum position (the only one that can shrink -- or, moving on, find a
/// new -- minimal window). Stops when any list is exhausted. This counts
/// non-degenerate minimal covers (Indri/Terrier-style); duplicate/overlap
/// counting subtleties beyond that are out of scope for v1.
///
/// e.g. `a` at `[0]`, `b` at `[1]`, `width = 3`: span is `1 < 3`, one
/// match. `a` at `[0]`, `b` at `[10]`, `width = 3`: span `10 >= 3`, zero
/// matches.
pub(crate) fn window_count(lists: &[&[u32]], width: u32) -> u32 {
    let k = lists.len();
    if k == 0 || lists.iter().any(|l| l.is_empty()) {
        return 0;
    }

    let mut pointers = vec![0usize; k];
    let mut count = 0u32;

    loop {
        let mut min_val = u32::MAX;
        let mut min_idx = 0usize;
        let mut max_val = 0u32;
        for (i, &ptr) in pointers.iter().enumerate() {
            let v = lists[i][ptr];
            if v < min_val {
                min_val = v;
                min_idx = i;
            }
            if v > max_val {
                max_val = v;
            }
        }

        if max_val - min_val < width {
            count += 1;
        }

        pointers[min_idx] += 1;
        if pointers[min_idx] >= lists[min_idx].len() {
            break;
        }
    }

    count
}
