//! Document filters applied by the search cores before heap insertion.
//!
//! A filter is checked once per *candidate* (a document the algorithm is
//! about to score), never per posting. Rejecting a candidate can only keep
//! the heap threshold lower, so pruning stays exact: the filtered top-k is
//! the top-k of the accepted documents.
//!
//! The unfiltered entry points ([`crate::search::maxscore::search_maxscore`],
//! [`crate::search::wand::search_wand`]) run the same generic cores with
//! [`NoFilter`], whose `accept` is a constant `true` the compiler removes.

use crate::base::DocId;

/// Decides whether a document may appear in the results.
///
/// Doc ids are the ones the caller sees, i.e. the *original* ids of a
/// reordered index (the search entry points translate before asking).
pub trait DocFilter {
    /// `true` only for filters that accept every document, letting the
    /// search entry points skip the doc-id translation of reordered indices.
    const ACCEPTS_ALL: bool = false;

    /// Returns `true` if `docid` may be returned.
    fn accept(&self, docid: DocId) -> bool;
}

/// Accepts every document (the unfiltered search path).
#[derive(Clone, Copy, Debug, Default)]
pub struct NoFilter;

impl DocFilter for NoFilter {
    const ACCEPTS_ALL: bool = true;

    #[inline(always)]
    fn accept(&self, _docid: DocId) -> bool {
        true
    }
}

impl<F: DocFilter + ?Sized> DocFilter for &F {
    const ACCEPTS_ALL: bool = F::ACCEPTS_ALL;

    #[inline(always)]
    fn accept(&self, docid: DocId) -> bool {
        (**self).accept(docid)
    }
}

/// `None` accepts everything, so a list of segments can mix filtered and
/// unfiltered ones under a single filter type.
impl<F: DocFilter> DocFilter for Option<F> {
    #[inline(always)]
    fn accept(&self, docid: DocId) -> bool {
        match self {
            Some(f) => f.accept(docid),
            None => true,
        }
    }
}

/// A growable set of document ids, stored as a bitset (one bit per doc id
/// up to the largest inserted one).
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct DocBitSet {
    words: Vec<u64>,
}

impl DocBitSet {
    /// An empty set.
    pub fn new() -> Self {
        Self::default()
    }

    /// An empty set with room for doc ids `< capacity` without reallocation.
    pub fn with_capacity(capacity: usize) -> Self {
        Self {
            words: Vec::with_capacity(capacity.div_ceil(64)),
        }
    }

    /// Builds a set from a boolean mask (`mask[docid]` is `true` iff
    /// `docid` is in the set).
    pub fn from_mask(mask: &[bool]) -> Self {
        let mut words = vec![0u64; mask.len().div_ceil(64)];
        for (docid, _) in mask.iter().enumerate().filter(|(_, &b)| b) {
            words[docid / 64] |= 1 << (docid % 64);
        }
        Self { words }
    }

    /// Builds a set from little-endian 64-bit words (bit `i` of word `w`
    /// is doc id `64 * w + i`).
    pub fn from_words(words: Vec<u64>) -> Self {
        Self { words }
    }

    /// The underlying 64-bit words (see [`DocBitSet::from_words`]).
    pub fn words(&self) -> &[u64] {
        &self.words
    }

    /// Adds `docid`; returns `true` if it was not already present.
    pub fn insert(&mut self, docid: DocId) -> bool {
        let (w, b) = Self::position(docid);
        if w >= self.words.len() {
            self.words.resize(w + 1, 0);
        }
        let was_set = self.words[w] & b != 0;
        self.words[w] |= b;
        !was_set
    }

    /// Removes `docid`; returns `true` if it was present.
    pub fn remove(&mut self, docid: DocId) -> bool {
        let (w, b) = Self::position(docid);
        match self.words.get_mut(w) {
            Some(word) if *word & b != 0 => {
                *word &= !b;
                true
            }
            _ => false,
        }
    }

    /// Whether `docid` is in the set.
    #[inline]
    pub fn contains(&self, docid: DocId) -> bool {
        let (w, b) = Self::position(docid);
        self.words.get(w).is_some_and(|word| word & b != 0)
    }

    /// Number of documents in the set.
    pub fn len(&self) -> usize {
        self.words.iter().map(|w| w.count_ones() as usize).sum()
    }

    /// Whether the set is empty.
    pub fn is_empty(&self) -> bool {
        self.words.iter().all(|&w| w == 0)
    }

    /// Iterates over the doc ids in the set, in increasing order.
    pub fn iter(&self) -> impl Iterator<Item = DocId> + '_ {
        self.words.iter().enumerate().flat_map(|(w, &word)| {
            let mut word = word;
            std::iter::from_fn(move || {
                if word == 0 {
                    return None;
                }
                let b = word.trailing_zeros() as u64;
                word &= word - 1;
                Some(64 * w as u64 + b)
            })
        })
    }

    #[inline(always)]
    fn position(docid: DocId) -> (usize, u64) {
        ((docid / 64) as usize, 1 << (docid % 64))
    }
}

impl FromIterator<DocId> for DocBitSet {
    fn from_iter<I: IntoIterator<Item = DocId>>(iter: I) -> Self {
        let mut set = Self::new();
        for docid in iter {
            set.insert(docid);
        }
        set
    }
}

/// Rejects the documents in the set (tombstones of deleted documents).
#[derive(Clone, Copy, Debug)]
pub struct Exclude<'a>(pub &'a DocBitSet);

impl DocFilter for Exclude<'_> {
    #[inline]
    fn accept(&self, docid: DocId) -> bool {
        !self.0.contains(docid)
    }
}

/// Accepts only the documents in the set (restriction to a sub-collection).
#[derive(Clone, Copy, Debug)]
pub struct Only<'a>(pub &'a DocBitSet);

impl DocFilter for Only<'_> {
    #[inline]
    fn accept(&self, docid: DocId) -> bool {
        self.0.contains(docid)
    }
}

/// An owned combination of tombstones and an allow-list, for callers (the
/// Python bindings, segment lists) that keep filters alongside indices.
#[derive(Clone, Debug, Default)]
pub struct DocSetFilter {
    /// Documents never returned.
    pub deleted: Option<DocBitSet>,
    /// If set, only these documents may be returned.
    pub allowed: Option<DocBitSet>,
}

impl DocFilter for DocSetFilter {
    #[inline]
    fn accept(&self, docid: DocId) -> bool {
        self.deleted.as_ref().is_none_or(|d| !d.contains(docid))
            && self.allowed.as_ref().is_none_or(|a| a.contains(docid))
    }
}

/// Applies `F` to the original id of an internal doc id (reordered index).
pub(crate) struct Remapped<'a, F> {
    pub map: &'a [DocId],
    pub inner: F,
}

impl<F: DocFilter> DocFilter for Remapped<'_, F> {
    #[inline]
    fn accept(&self, docid: DocId) -> bool {
        self.inner.accept(self.map[docid as usize])
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn bitset_insert_remove_iter() {
        let mut s = DocBitSet::new();
        assert!(s.is_empty());
        assert!(s.insert(3));
        assert!(!s.insert(3));
        assert!(s.insert(64));
        assert!(s.insert(200));
        assert_eq!(s.len(), 3);
        assert!(s.contains(64) && !s.contains(65) && !s.contains(10_000));
        assert_eq!(s.iter().collect::<Vec<_>>(), vec![3, 64, 200]);
        assert!(s.remove(64));
        assert!(!s.remove(64));
        assert!(!s.remove(10_000));
        assert_eq!(s.iter().collect::<Vec<_>>(), vec![3, 200]);
        assert_eq!(
            DocBitSet::from_mask(&[false, true, true]),
            [1, 2].into_iter().collect()
        );
    }

    #[test]
    fn filters() {
        let s: DocBitSet = [1, 5].into_iter().collect();
        assert!(!Exclude(&s).accept(1) && Exclude(&s).accept(2));
        assert!(Only(&s).accept(5) && !Only(&s).accept(2));
        assert!(None::<Exclude>.accept(1));
        let f = DocSetFilter {
            deleted: Some([1].into_iter().collect()),
            allowed: Some(s.clone()),
        };
        assert!(!f.accept(1) && f.accept(5) && !f.accept(2));
    }
}
