//! Block table of a compressed index decoded term by term (feature
//! `lazy-pages`), for indexes read through a [`crate::storage::ReaderFiles`].
//!
//! Decoded, the block table takes about six times the size of `index.bin`.
//! A lazily read index instead keeps the bytes of `index.bin` and the start
//! of each term in it, and decodes the blocks of a term when a query uses
//! it. Decoded terms go to the [`RangeCache`] of the posting lists, and are
//! evicted with them; iterators share them, so eviction never frees blocks
//! that are still in use.

use std::io::{self, Cursor};
use std::ops::Deref;
use std::sync::Arc;

use super::{
    read_term_blocks, read_term_header, BlockPositions, CompressedIndexInformation,
    TermBlockInformation, TermStats,
};
use crate::base::{DocId, TermIndex};
use crate::storage::RangeCache;

/// The decoded blocks of a term
pub(super) struct DecodedTerm {
    pages: Vec<TermBlockInformation>,
    stats: TermStats,
}

/// The blocks of a term: borrowed from a decoded block table, or shared
/// with the cache
pub(super) enum Pages<'a> {
    Borrowed(&'a [TermBlockInformation]),
    Shared(Arc<DecodedTerm>),
}

impl Deref for Pages<'_> {
    type Target = [TermBlockInformation];

    #[inline]
    fn deref(&self) -> &[TermBlockInformation] {
        match self {
            Pages::Borrowed(pages) => pages,
            Pages::Shared(term) => &term.pages,
        }
    }
}

/// Iterator over the blocks of a term (copies of them, since the iterator
/// may own the blocks)
pub(super) struct BlockIter<'a> {
    pages: Pages<'a>,
    next: usize,
}

impl<'a> BlockIter<'a> {
    pub(super) fn borrowed(pages: &'a [TermBlockInformation]) -> Self {
        Self {
            pages: Pages::Borrowed(pages),
            next: 0,
        }
    }
}

impl Iterator for BlockIter<'_> {
    type Item = TermBlockInformation;

    #[inline]
    fn next(&mut self) -> Option<TermBlockInformation> {
        let page = self.pages.get(self.next)?.clone();
        self.next += 1;
        Some(page)
    }
}

/// The block table as the bytes of `index.bin`
pub(super) struct LazyBlocks {
    raw: Vec<u8>,
    version: u32,
    /// Offset of each term record in `raw`
    offsets: Vec<u32>,
    /// Start of each term in `docids.dat` and `impacts.dat`
    starts: Vec<(u64, u64)>,
    /// Start of each term in `positions.dat` (empty without positions)
    positions_starts: Vec<u64>,
    /// Largest document id of the index
    max_doc_id: DocId,
    cache: Arc<RangeCache>,
    /// Key of this index in the cache
    owner: u64,
}

impl LazyBlocks {
    /// Reads the header of `index.bin` and the start of each term: the
    /// returned information has no terms. The whole file is parsed, so
    /// decoding a term later cannot fail.
    pub(super) fn read(
        raw: Vec<u8>,
        cache: Arc<RangeCache>,
    ) -> io::Result<(CompressedIndexInformation, Self)> {
        if raw.len() > u32::MAX as usize {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                "index.bin is larger than 4 GiB",
            ));
        }
        let mut reader = Cursor::new(&raw[..]);
        let version = CompressedIndexInformation::read_binary_version(&mut reader)?;
        let (num_terms, information) =
            CompressedIndexInformation::read_binary_header(&mut reader, version)?;
        let has_positions = information.positions_compressor.is_some();

        // A term takes at least a few bytes: do not trust num_terms to
        // allocate
        let capacity = num_terms.min(raw.len());
        let mut offsets = Vec::with_capacity(capacity);
        let mut starts = Vec::with_capacity(capacity);
        let mut positions_starts = Vec::with_capacity(if has_positions { capacity } else { 0 });
        let mut positions = BlockPositions::default();
        let mut max_doc_id = 0;
        for _ in 0..num_terms {
            offsets.push(reader.position() as u32);
            starts.push((positions.docid, positions.impact));
            if has_positions {
                positions_starts.push(positions.positions);
            }
            let header = read_term_header(&mut reader)?;
            max_doc_id = max_doc_id.max(header.max_doc_id);
            read_term_blocks(&mut reader, version, &header, &mut positions, |_| ())?;
        }

        let owner = cache.new_owner();
        Ok((
            information,
            Self {
                raw,
                version,
                offsets,
                starts,
                positions_starts,
                max_doc_id,
                cache,
                owner,
            },
        ))
    }

    /// Number of terms
    pub(super) fn len(&self) -> usize {
        self.offsets.len()
    }

    /// Largest document id of the index
    pub(super) fn max_doc_id(&self) -> DocId {
        self.max_doc_id
    }

    /// Number of postings of a term (0 if out of range), from its header:
    /// its blocks are not decoded
    pub(super) fn term_postings(&self, term: TermIndex) -> usize {
        let Some(&offset) = self.offsets.get(term) else {
            return 0;
        };
        read_term_header(&mut Cursor::new(&self.raw[offset as usize..]))
            .map(|header| header.length)
            .unwrap_or_else(|e| panic!("index.bin: cannot read term {term}: {e}"))
    }

    /// The decoded blocks of a term (in range), from the cache if there
    fn decode(&self, term: TermIndex) -> io::Result<Arc<DecodedTerm>> {
        if let Some(object) = self.cache.get_object(self.owner, term) {
            if let Ok(decoded) = object.downcast::<DecodedTerm>() {
                return Ok(decoded);
            }
        }

        let mut reader = Cursor::new(&self.raw[self.offsets[term] as usize..]);
        let (docid, impact) = self.starts[term];
        let mut positions = BlockPositions {
            docid,
            impact,
            positions: self.positions_starts.get(term).copied().unwrap_or(0),
        };
        let header = read_term_header(&mut reader)?;
        let mut pages = Vec::with_capacity(header.num_blocks);
        read_term_blocks(&mut reader, self.version, &header, &mut positions, |page| {
            pages.push(page)
        })?;
        let min_dl = pages
            .iter()
            .map(|p| p.min_doc_length as u32)
            .min()
            .unwrap_or(0);

        let size = std::mem::size_of::<DecodedTerm>()
            + pages.capacity() * std::mem::size_of::<TermBlockInformation>();
        let decoded = Arc::new(DecodedTerm {
            pages,
            stats: TermStats {
                max_value: header.max_value,
                max_doc_id: header.max_doc_id,
                length: header.length,
                min_dl,
            },
        });
        self.cache
            .insert_object(self.owner, term, decoded.clone(), size);
        Ok(decoded)
    }

    /// Decoding a term only fails if `index.bin` changed in memory, which
    /// [`Self::read`] rules out
    fn decoded(&self, term: TermIndex) -> Arc<DecodedTerm> {
        self.decode(term)
            .unwrap_or_else(|e| panic!("index.bin: cannot decode term {term}: {e}"))
    }

    /// The blocks of a term, `None` if out of range
    pub(super) fn pages(&self, term: TermIndex) -> io::Result<Option<Pages<'static>>> {
        if term >= self.len() {
            return Ok(None);
        }
        Ok(Some(Pages::Shared(self.decode(term)?)))
    }

    /// Iterator over the blocks of a term (none if out of range)
    pub(super) fn block_iter(&self, term: TermIndex) -> BlockIter<'static> {
        let pages = if term < self.len() {
            Pages::Shared(self.decoded(term))
        } else {
            Pages::Borrowed(&[])
        };
        BlockIter { pages, next: 0 }
    }

    /// Statistics of a term (panics if out of range)
    pub(super) fn term_stats(&self, term: TermIndex) -> TermStats {
        assert!(
            term < self.len(),
            "term {term} out of range ({} terms)",
            self.len()
        );
        self.decoded(term).stats
    }
}
