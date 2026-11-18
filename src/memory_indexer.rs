//! A forward index built in memory, for targets without a file system
//! (WASM), with optional spilling of postings to an append-only scratch
//! stream when a memory budget is exceeded.
//!
//! ```text
//! add → [over budget] → spill(target, writer) → … → with_spill(reader) → compress_to
//! ```
//!
//! The scratch stream belongs to the caller: [`MemoryIndexer::spill`]
//! appends to a [`Write`], and [`MemoryIndexer::with_spill`] reads it back
//! through a [`SpillReader`] at compression time.

use std::io::{self, Write};
use std::sync::Mutex;

use crate::base::{DocId, ImpactValue, Len, TermImpact, TermIndex};
use crate::index::{SparseIndexInformation, SparseIndexView};

/// A piece of a term's postings in the scratch stream: `gaps_len` bytes of
/// docid gaps, then `count` little-endian `f32` values
#[derive(Clone, Copy)]
struct SpilledChunk {
    offset: u64,
    gaps_len: u32,
    count: u32,
}

/// Postings of one term: document id gaps as LEB128 varints (the first from
/// 0, or from the last spilled document) and impact values, ~6 bytes per
/// posting instead of 16 for a `TermImpact`.
#[derive(Default)]
struct MemoryPostings {
    gaps: Vec<u8>,
    values: Vec<ImpactValue>,
    last_docid: DocId,
    /// Earlier postings, in the scratch stream
    spilled: Vec<SpilledChunk>,
}

/// Grows `v` by ~25% when full, instead of doubling as `Vec::push` does:
/// keeps the unused capacity, i.e. the build memory overhead, small
fn reserve_compact<T>(v: &mut Vec<T>, additional: usize) {
    if v.capacity() - v.len() < additional {
        v.reserve_exact(additional.max(v.len() / 4).max(4));
    }
}

/// Decodes docid gaps and values into postings, docids starting after
/// `docid`
fn decode<'a>(
    gaps: &'a [u8],
    values: &'a [ImpactValue],
    mut docid: DocId,
) -> impl Iterator<Item = TermImpact> + 'a {
    let mut bytes = gaps.iter();
    values.iter().map(move |&value| {
        let mut shift = 0;
        loop {
            let byte = *bytes.next().expect("one gap per value");
            docid += ((byte & 0x7f) as DocId) << shift;
            if byte & 0x80 == 0 {
                break;
            }
            shift += 7;
        }
        TermImpact { docid, value }
    })
}

impl MemoryPostings {
    /// Bytes held in memory (without the unused capacity)
    fn bytes(&self) -> usize {
        self.gaps.len() + self.values.len() * std::mem::size_of::<ImpactValue>()
    }

    /// Returns the number of bytes added
    fn push(&mut self, docid: DocId, value: ImpactValue) -> usize {
        let before = self.gaps.len();
        let mut gap = docid - self.last_docid;
        reserve_compact(&mut self.gaps, 10);
        loop {
            let byte = (gap & 0x7f) as u8;
            gap >>= 7;
            if gap == 0 {
                self.gaps.push(byte);
                break;
            }
            self.gaps.push(byte | 0x80);
        }
        reserve_compact(&mut self.values, 1);
        self.values.push(value);
        self.last_docid = docid;
        self.gaps.len() - before + std::mem::size_of::<ImpactValue>()
    }
}

/// Reads back the scratch stream written by [`MemoryIndexer::spill`]
pub trait SpillReader: Sync {
    /// Exactly `length` bytes at `offset`
    fn read(&self, offset: u64, length: usize) -> io::Result<Vec<u8>>;
}

/// A scratch stream kept in memory (tests, or a caller with no better place)
impl SpillReader for Vec<u8> {
    fn read(&self, offset: u64, length: usize) -> io::Result<Vec<u8>> {
        let start = offset as usize;
        self.get(start..start + length)
            .map(<[u8]>::to_vec)
            .ok_or_else(|| io::Error::new(io::ErrorKind::UnexpectedEof, "spill: read past end"))
    }
}

/// A forward index built in memory, for collections that fit in RAM and
/// targets without a file system (WASM). Compress it into an index with
/// [`crate::compress::CompressionTransform::compress_to`] (after
/// [`MemoryIndexer::with_spill`] if postings were spilled).
#[derive(Default)]
pub struct MemoryIndexer {
    /// Postings of each term, by increasing document id
    postings: Vec<MemoryPostings>,
    /// Maximum impact of each term
    max_values: Vec<ImpactValue>,
    last_docid: Option<DocId>,
    /// Largest document id with postings
    max_doc_id: DocId,
    num_postings: usize,
    /// Bytes of postings in memory
    memory_bytes: usize,
    /// Bytes written to the scratch stream
    spilled_bytes: u64,
    /// Start of the region of the scratch stream written by each `spill`
    regions: Vec<u64>,
}

impl MemoryIndexer {
    pub fn new() -> Self {
        Self::default()
    }

    /// Adds a document; document ids must be strictly increasing, and terms
    /// unique within a document.
    pub fn add(
        &mut self,
        docid: DocId,
        terms: &[TermIndex],
        values: &[ImpactValue],
    ) -> io::Result<()> {
        let invalid = |msg: String| io::Error::new(io::ErrorKind::InvalidInput, msg);
        if terms.len() != values.len() {
            return Err(invalid(
                "Value and term lists should have the same length".into(),
            ));
        }
        if self.last_docid.is_some_and(|last| docid <= last) {
            return Err(invalid(format!(
                "Document ids must be increasing ({docid} after {})",
                self.last_docid.unwrap()
            )));
        }
        // Validate everything before adding anything
        let mut sorted = terms.to_vec();
        sorted.sort_unstable();
        if sorted.windows(2).any(|w| w[0] == w[1]) {
            return Err(invalid(format!("Repeated term in document {docid}")));
        }
        if let Some(v) = values.iter().find(|v| !(**v > 0.)) {
            return Err(invalid(format!(
                "Impact values must be positive ({v} in document {docid})"
            )));
        }

        if let Some(&max_term) = sorted.last() {
            if max_term >= self.postings.len() {
                self.postings.resize_with(max_term + 1, Default::default);
                self.max_values.resize(max_term + 1, 0.);
            }
        }
        for (&term, &value) in terms.iter().zip(values) {
            self.memory_bytes += self.postings[term].push(docid, value);
            self.max_values[term] = self.max_values[term].max(value);
        }
        if !terms.is_empty() {
            self.max_doc_id = docid;
        }
        self.num_postings += terms.len();
        self.last_docid = Some(docid);
        Ok(())
    }

    /// Number of postings
    pub fn num_postings(&self) -> usize {
        self.num_postings
    }

    /// Bytes of postings held in memory (spilled postings excluded)
    pub fn memory_bytes(&self) -> usize {
        self.memory_bytes
    }

    /// Bytes written to the scratch stream so far
    pub fn spilled_bytes(&self) -> u64 {
        self.spilled_bytes
    }

    /// Moves the postings of the largest terms to the end of the scratch
    /// stream `out` until at most `target` bytes remain in memory. `out`
    /// must receive the bytes in order, after those of the previous calls.
    /// Each call writes one region, by increasing term index, so that
    /// compression (term by term) reads each region sequentially.
    /// On error, the scratch stream is in an unknown state: the indexer
    /// must not be used anymore.
    pub fn spill(&mut self, target: usize, out: &mut dyn Write) -> io::Result<()> {
        if self.memory_bytes <= target {
            return Ok(());
        }
        let mut terms: Vec<(usize, TermIndex)> = self
            .postings
            .iter()
            .enumerate()
            .filter(|(_, p)| !p.values.is_empty())
            .map(|(t, p)| (p.bytes(), t))
            .collect();
        terms.sort_unstable_by(|a, b| b.cmp(a));
        let mut freed = 0;
        let mut selected: Vec<(TermIndex, usize)> = terms
            .into_iter()
            .take_while(|&(bytes, _)| {
                let more = self.memory_bytes - freed > target;
                freed += bytes;
                more
            })
            .map(|(bytes, term)| (term, bytes))
            .collect();
        selected.sort_unstable();
        self.regions.push(self.spilled_bytes);
        for (term, bytes) in selected {
            let p = &mut self.postings[term];
            out.write_all(&p.gaps)?;
            for v in &p.values {
                out.write_all(&v.to_le_bytes())?;
            }
            p.spilled.push(SpilledChunk {
                offset: self.spilled_bytes,
                gaps_len: p.gaps.len() as u32,
                count: p.values.len() as u32,
            });
            self.spilled_bytes += bytes as u64;
            self.memory_bytes -= bytes;
            // Frees the memory; `last_docid` stays, for the next gaps
            p.gaps = Vec::new();
            p.values = Vec::new();
        }
        Ok(())
    }

    /// Whether some postings are in the scratch stream
    pub fn is_spilled(&self) -> bool {
        self.spilled_bytes > 0
    }

    /// The index, with spilled postings read through `reader`. Small
    /// chunks are read through one read-ahead buffer per spilled region,
    /// using at most `read_ahead` bytes in all (each buffer at most 1 MiB):
    /// fewer and larger reads, when each read has a fixed cost.
    pub fn with_spill<'a>(
        &'a self,
        reader: &'a dyn SpillReader,
        read_ahead: usize,
    ) -> SpilledIndex<'a> {
        let buffer_size = (read_ahead / self.regions.len().max(1)).min(1 << 20);
        SpilledIndex {
            indexer: self,
            reader,
            error: Mutex::new(None),
            buffer_size,
            buffers: Mutex::new((0..self.regions.len()).map(|_| (0, Vec::new())).collect()),
        }
    }

    fn check_not_spilled(&self) {
        assert!(
            !self.is_spilled(),
            "MemoryIndexer with spilled postings: use with_spill"
        );
    }
}

impl Len for MemoryIndexer {
    fn len(&self) -> usize {
        self.postings.len()
    }
}

impl SparseIndexInformation for MemoryIndexer {
    fn value_range(&self, term_ix: TermIndex) -> (ImpactValue, ImpactValue) {
        // Same as `SparseBuilderIndex`, so both compress identically
        (0., self.max_values[term_ix])
    }
}

/// Postings of an indexer without spilled postings (panics otherwise: use
/// [`MemoryIndexer::with_spill`])
impl SparseIndexView for MemoryIndexer {
    fn iterator<'a>(&'a self, term_ix: TermIndex) -> Box<dyn Iterator<Item = TermImpact> + 'a> {
        self.check_not_spilled();
        let p = &self.postings[term_ix];
        Box::new(decode(&p.gaps, &p.values, 0))
    }

    fn max_doc_id(&self) -> DocId {
        self.max_doc_id
    }
}

/// A [`MemoryIndexer`] whose spilled postings are read back through a
/// [`SpillReader`]. `iterator` cannot fail: on a read error, it yields no
/// postings and keeps the error, to be checked with
/// [`SpilledIndex::check`] once done (e.g. after compression).
pub struct SpilledIndex<'a> {
    indexer: &'a MemoryIndexer,
    reader: &'a dyn SpillReader,
    error: Mutex<Option<io::Error>>,
    /// Size of the read-ahead buffers (0: none)
    buffer_size: usize,
    /// Read-ahead buffer of each region: offset and bytes
    buffers: Mutex<Vec<(u64, Vec<u8>)>>,
}

impl SpilledIndex<'_> {
    /// The first read error, if any
    pub fn check(&self) -> io::Result<()> {
        match self.error.lock().unwrap().take() {
            Some(e) => Err(e),
            None => Ok(()),
        }
    }

    /// `length` bytes at `offset`, through the read-ahead buffer of the
    /// region
    fn read(&self, offset: u64, length: usize) -> io::Result<Vec<u8>> {
        let regions = &self.indexer.regions;
        if length >= self.buffer_size || regions.is_empty() {
            return self.reader.read(offset, length);
        }
        let region = regions.partition_point(|&start| start <= offset).max(1) - 1;
        let end = regions
            .get(region + 1)
            .copied()
            .unwrap_or(self.indexer.spilled_bytes);
        let mut buffers = self.buffers.lock().unwrap();
        let (start, data) = &mut buffers[region];
        let inside = |start: u64, data: &Vec<u8>| {
            offset >= start && offset + length as u64 <= start + data.len() as u64
        };
        if !inside(*start, data) {
            let size = (self.buffer_size as u64).min(end.saturating_sub(offset)) as usize;
            *data = self.reader.read(offset, size.max(length))?;
            *start = offset;
            if !inside(*start, data) {
                return Err(io::Error::new(
                    io::ErrorKind::UnexpectedEof,
                    format!("spill: short read at {offset}"),
                ));
            }
        }
        let from = (offset - *start) as usize;
        Ok(data[from..from + length].to_vec())
    }

    fn postings(&self, term_ix: TermIndex) -> io::Result<Vec<TermImpact>> {
        let p = &self.indexer.postings[term_ix];
        let mut result = Vec::with_capacity(
            p.values.len() + p.spilled.iter().map(|c| c.count as usize).sum::<usize>(),
        );
        let mut last = 0;
        for chunk in &p.spilled {
            let count = chunk.count as usize;
            let gaps_len = chunk.gaps_len as usize;
            let length = gaps_len + count * std::mem::size_of::<ImpactValue>();
            let bytes = self.read(chunk.offset, length)?;
            if bytes.len() != length {
                return Err(io::Error::new(
                    io::ErrorKind::UnexpectedEof,
                    format!(
                        "spill: read {} bytes at {} instead of {length}",
                        bytes.len(),
                        chunk.offset
                    ),
                ));
            }
            // One complete varint per value, so that decoding cannot fail
            let gaps = &bytes[..gaps_len];
            let ends = gaps.iter().filter(|b| *b & 0x80 == 0).count();
            if ends != count || gaps.last().is_some_and(|b| b & 0x80 != 0) {
                return Err(io::Error::new(
                    io::ErrorKind::InvalidData,
                    format!("spill: corrupt chunk at {}", chunk.offset),
                ));
            }
            let values: Vec<ImpactValue> = bytes[gaps_len..]
                .as_chunks::<4>()
                .0
                .iter()
                .map(|b| ImpactValue::from_le_bytes(*b))
                .collect();
            result.extend(decode(gaps, &values, last));
            last = result.last().map_or(last, |r| r.docid);
        }
        result.extend(decode(&p.gaps, &p.values, last));
        Ok(result)
    }
}

impl Len for SpilledIndex<'_> {
    fn len(&self) -> usize {
        self.indexer.len()
    }
}

impl SparseIndexInformation for SpilledIndex<'_> {
    fn value_range(&self, term_ix: TermIndex) -> (ImpactValue, ImpactValue) {
        self.indexer.value_range(term_ix)
    }
}

impl SparseIndexView for SpilledIndex<'_> {
    fn iterator<'a>(&'a self, term_ix: TermIndex) -> Box<dyn Iterator<Item = TermImpact> + 'a> {
        match self.postings(term_ix) {
            Ok(postings) => Box::new(postings.into_iter()),
            Err(e) => {
                self.error.lock().unwrap().get_or_insert(e);
                Box::new(std::iter::empty())
            }
        }
    }

    fn max_doc_id(&self) -> DocId {
        self.indexer.max_doc_id
    }
}
