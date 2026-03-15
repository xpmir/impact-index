//! Compression of posting lists (document IDs and impact values).
//!
//! Provides a trait-based compression framework with pluggable compressors:
//! - [`docid::EliasFanoCompressor`]: Elias-Fano encoding for document IDs
//! - [`impact::Quantizer`]: Fixed-range quantization for impact values
//! - [`impact::GlobalQuantizerFactory`]: Auto-ranging quantizer based on global statistics
//!
//! The [`CompressionTransform`] applies compression to a raw forward index,
//! producing a block-based compressed index on disk.

use std::{
    cell::RefCell,
    fmt,
    fs::{create_dir, File},
    io::{Seek, Write},
    path::Path,
};

use super::{
    index::{BlockTermImpactIterator, SparseIndex, SparseIndexView},
    transforms::IndexTransform,
};
use crate::{
    base::{save_index, DocId, ImpactValue, IndexLoader, Len, TermImpact, TermIndex},
    index::SparseIndexInformation,
    utils::buffer::{Buffer, MemoryBuffer, MmapBuffer, Slice},
};
use indicatif::{ProgressBar, ProgressIterator, ProgressStyle};
use log::{debug, info};
use rayon::prelude::*;
use serde::{Deserialize, Serialize};

pub mod docid;
pub mod impact;

//
// ---- Compressed index global information  ---
//

#[derive(Serialize, Deserialize)]
pub struct TermBlockInformation {
    /// Position within the document ID stream
    pub docid_position_range: (u64, u64),

    /// Position within the impact value stream
    pub impact_position_range: (u64, u64),

    /// Number of records
    pub length: usize,

    /// Maximum value for this page
    pub max_value: ImpactValue,

    /// Maximum document ID for this page
    pub min_doc_id: DocId,

    /// Maximum document ID for this page
    pub max_doc_id: DocId,
}

impl std::fmt::Display for TermBlockInformation {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(
            f,
            "(docids: {}-{}, impacts: {}-{}, len: {}, max_v: {}, docid: {}-{})",
            self.docid_position_range.0,
            self.docid_position_range.1,
            self.impact_position_range.0,
            self.impact_position_range.1,
            self.length,
            self.max_value,
            self.min_doc_id,
            self.max_doc_id
        )
    }
}

//
// ---- Compression ---
//

/// Trait for encoding/decoding a sequence of values within a block.
pub trait Compressor<T>: Sync + Send {
    /// Writes compressed values to the given writer.
    fn write(
        &self,
        writer: &mut dyn Write,
        values: &[T],
        term_index: TermIndex,
        info: &TermBlockInformation,
    );
    /// Reads compressed values from a memory slice, returning an iterator.
    fn read<'a>(
        &self,
        slice: Box<dyn Slice + 'a>,
        term_index: TermIndex,
        info: &TermBlockInformation,
    ) -> Box<dyn Iterator<Item = T> + Send + 'a>;

    /// Decode compressed values into a reusable buffer.
    ///
    /// The buffer is cleared and filled with decoded values.
    /// Default implementation falls back to `read().collect()`.
    fn decode_into(
        &self,
        slice: Box<dyn Slice + '_>,
        term_index: TermIndex,
        info: &TermBlockInformation,
        buffer: &mut Vec<T>,
    ) {
        buffer.clear();
        buffer.extend(self.read(slice, term_index, info));
    }

    /// Decode from a raw byte slice (zero-allocation fast path).
    ///
    /// Default implementation wraps bytes in a Slice and calls decode_into.
    fn decode_into_bytes(
        &self,
        data: &[u8],
        term_index: TermIndex,
        info: &TermBlockInformation,
        buffer: &mut Vec<T>,
    ) {
        // Default: wrap in a Slice and delegate
        struct ByteSlice<'a>(&'a [u8]);
        impl<'a> Slice for ByteSlice<'a> {
            fn data(&self) -> &[u8] {
                self.0
            }
            fn read(&mut self, index: usize, buf: &mut [u8]) -> std::io::Result<usize> {
                let src = &self.0[index..];
                let len = buf.len().min(src.len());
                buf[..len].copy_from_slice(&src[..len]);
                Ok(len)
            }
        }
        self.decode_into(Box::new(ByteSlice(data)), term_index, info, buffer);
    }
}

/// A serializable compressor for document IDs.
#[typetag::serde(tag = "type")]
pub trait DocIdCompressor: Compressor<DocId> {}

/// Factory for creating [`DocIdCompressor`] instances, potentially
/// using global index statistics.
pub trait DocIdCompressorFactory: Sync + Send {
    /// Creates a compressor, optionally inspecting the index for statistics.
    fn create(&self, index: &dyn SparseIndexView) -> Box<dyn DocIdCompressor>;
    /// Clones this factory.
    fn clone(&self) -> Box<dyn DocIdCompressorFactory>;
}

/// A serializable compressor for impact values.
#[typetag::serde(tag = "type")]
pub trait ImpactCompressor: Compressor<ImpactValue> {}

/// Factory for creating [`ImpactCompressor`] instances, potentially
/// using global index statistics.
pub trait ImpactCompressorFactory: Sync + Send {
    /// Creates a compressor, optionally inspecting the index for statistics.
    fn create(&self, index: &dyn SparseIndexView) -> Box<dyn ImpactCompressor>;
    /// Clones this factory.
    fn clone(&self) -> Box<dyn ImpactCompressorFactory>;
}

/// Block-based index information for a term
#[derive(Serialize, Deserialize)]
pub struct TermBlocksInformation {
    pub pages: Vec<TermBlockInformation>,
    pub max_value: ImpactValue,
    pub max_doc_id: DocId,
    pub length: usize,
}

/// Global information on the index structure
#[derive(Serialize, Deserialize)]
pub struct CompressedIndexInformation {
    pub terms: Vec<TermBlocksInformation>,
    doc_ids_compressor: Box<dyn DocIdCompressor>,
    values_compressor: Box<dyn ImpactCompressor>,
}

const COMPRESSED_INDEX_MAGIC: u32 = 0x49445832; // "IDX2"
const COMPRESSED_INDEX_VERSION: u32 = 3;

/// Write a u64 as variable-length integer (1-9 bytes).
fn write_vint(writer: &mut dyn Write, mut v: u64) -> std::io::Result<()> {
    while v >= 0x80 {
        writer.write_all(&[(v as u8) | 0x80])?;
        v >>= 7;
    }
    writer.write_all(&[v as u8])
}

/// Read a variable-length integer.
fn read_vint(reader: &mut dyn std::io::Read) -> std::io::Result<u64> {
    let mut result: u64 = 0;
    let mut shift = 0;
    loop {
        let mut byte = [0u8; 1];
        reader.read_exact(&mut byte)?;
        result |= ((byte[0] & 0x7F) as u64) << shift;
        if byte[0] & 0x80 == 0 {
            return Ok(result);
        }
        shift += 7;
    }
}

impl CompressedIndexInformation {
    /// Write metadata in compact binary format.
    ///
    /// Format v2: per-block stores only byte lengths (not absolute positions).
    /// Per-term stores stream starting offsets. Positions reconstructed at load
    /// via prefix sum.
    ///
    /// Per-term:  20 bytes (num_blocks:u32, max_value:f32, max_doc_id:u64, length:u32)
    /// Per-block: 24 bytes (docid_len:u32, impact_len:u32, length:u16, _pad:u16,
    ///                       max_value:f32, min_doc_id:u64, max_doc_id:u64)
    fn write_binary(&self, writer: &mut dyn Write) -> std::io::Result<()> {
        use byteorder::{LittleEndian, WriteBytesExt};

        // Header
        writer.write_u32::<LittleEndian>(COMPRESSED_INDEX_MAGIC)?;
        writer.write_u32::<LittleEndian>(COMPRESSED_INDEX_VERSION)?;
        writer.write_u32::<LittleEndian>(self.terms.len() as u32)?;

        // Compressor header (CBOR, small, only once)
        let mut compressor_buf = Vec::new();
        ciborium::ser::into_writer(
            &(&self.doc_ids_compressor, &self.values_compressor),
            &mut compressor_buf,
        )
        .expect("Failed to serialize compressors");
        writer.write_u32::<LittleEndian>(compressor_buf.len() as u32)?;
        writer.write_all(&compressor_buf)?;

        for term in &self.terms {
            // Term header
            write_vint(writer, term.pages.len() as u64)?;
            writer.write_f32::<LittleEndian>(term.max_value)?;
            write_vint(writer, term.max_doc_id)?;
            write_vint(writer, term.length as u64)?;

            // Compute min block value for quantization range
            let min_block_val = term
                .pages
                .iter()
                .map(|b| b.max_value)
                .fold(f32::INFINITY, f32::min);
            writer.write_f32::<LittleEndian>(min_block_val)?;
            // Quantization: block max_value encoded as 1 byte in [min_block_val, term.max_value]
            let range = term.max_value - min_block_val;

            // Block records: VInt + delta doc IDs + 1-byte quantized max_value
            let mut prev_doc_id: u64 = 0;
            for block in &term.pages {
                let docid_len =
                    (block.docid_position_range.1 - block.docid_position_range.0) as u64;
                let impact_len =
                    (block.impact_position_range.1 - block.impact_position_range.0) as u64;
                write_vint(writer, docid_len)?;
                write_vint(writer, impact_len)?;
                write_vint(writer, block.length as u64)?;
                // 1-byte quantized max_value (rounded UP for safe upper bound)
                let q = if range > 0.0 {
                    (((block.max_value - min_block_val) / range * 255.0).ceil() as u32).min(255)
                        as u8
                } else {
                    255u8
                };
                writer.write_all(&[q])?;
                // Delta-encode doc IDs
                write_vint(writer, block.min_doc_id - prev_doc_id)?;
                write_vint(writer, block.max_doc_id - block.min_doc_id)?;
                prev_doc_id = block.max_doc_id;
            }
        }
        Ok(())
    }

    /// Read metadata from compact binary format.
    /// Reconstructs absolute byte positions via prefix sum.
    fn read_binary(reader: &mut dyn std::io::Read) -> std::io::Result<Self> {
        use byteorder::{LittleEndian, ReadBytesExt};

        let magic = reader.read_u32::<LittleEndian>()?;
        if magic != COMPRESSED_INDEX_MAGIC {
            return Err(std::io::Error::new(
                std::io::ErrorKind::InvalidData,
                "Not a compressed index binary file",
            ));
        }
        let _version = reader.read_u32::<LittleEndian>()?;
        let num_terms = reader.read_u32::<LittleEndian>()? as usize;

        // Compressor header
        let compressor_len = reader.read_u32::<LittleEndian>()? as usize;
        let mut compressor_buf = vec![0u8; compressor_len];
        reader.read_exact(&mut compressor_buf)?;
        let (doc_ids_compressor, values_compressor): (
            Box<dyn DocIdCompressor>,
            Box<dyn ImpactCompressor>,
        ) = ciborium::de::from_reader(&compressor_buf[..])
            .map_err(|e| std::io::Error::new(std::io::ErrorKind::InvalidData, e.to_string()))?;

        // Read terms and blocks (VInt + delta encoded)
        let mut terms = Vec::with_capacity(num_terms);
        let mut docid_pos: u64 = 0;
        let mut impact_pos: u64 = 0;

        for _ in 0..num_terms {
            let num_blocks = read_vint(reader)? as usize;
            let max_value = reader.read_f32::<LittleEndian>()?;
            let max_doc_id = read_vint(reader)?;
            let length = read_vint(reader)? as usize;
            let min_block_val = reader.read_f32::<LittleEndian>()?;
            let range = max_value - min_block_val;

            let mut pages = Vec::with_capacity(num_blocks);
            let mut prev_doc_id: u64 = 0;
            for _ in 0..num_blocks {
                let docid_len = read_vint(reader)?;
                let impact_len = read_vint(reader)?;
                let block_length = read_vint(reader)? as usize;
                // Dequantize 1-byte max_value
                let mut q_byte = [0u8; 1];
                reader.read_exact(&mut q_byte)?;
                let block_max_value = min_block_val + (q_byte[0] as f32 / 255.0) * range;
                let min_doc_id = prev_doc_id + read_vint(reader)?;
                let block_max_doc_id = min_doc_id + read_vint(reader)?;
                prev_doc_id = block_max_doc_id;

                pages.push(TermBlockInformation {
                    docid_position_range: (docid_pos, docid_pos + docid_len),
                    impact_position_range: (impact_pos, impact_pos + impact_len),
                    length: block_length,
                    max_value: block_max_value,
                    min_doc_id,
                    max_doc_id: block_max_doc_id,
                });

                docid_pos += docid_len;
                impact_pos += impact_len;
            }

            terms.push(TermBlocksInformation {
                pages,
                max_value,
                max_doc_id,
                length,
            });
        }

        Ok(CompressedIndexInformation {
            terms,
            doc_ids_compressor,
            values_compressor,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::compress::docid::EliasFanoCompressor;
    use crate::compress::impact::Identity;

    #[test]
    fn test_binary_metadata_roundtrip() {
        let info = CompressedIndexInformation {
            terms: vec![
                TermBlocksInformation {
                    pages: vec![
                        TermBlockInformation {
                            docid_position_range: (0, 100),
                            impact_position_range: (0, 200),
                            length: 128,
                            max_value: 3.14,
                            min_doc_id: 0,
                            max_doc_id: 500,
                        },
                        TermBlockInformation {
                            docid_position_range: (100, 180),
                            impact_position_range: (200, 350),
                            length: 64,
                            max_value: 2.71,
                            min_doc_id: 501,
                            max_doc_id: 999,
                        },
                    ],
                    max_value: 3.14,
                    max_doc_id: 999,
                    length: 192,
                },
                TermBlocksInformation {
                    pages: vec![TermBlockInformation {
                        docid_position_range: (180, 220),
                        impact_position_range: (350, 400),
                        length: 32,
                        max_value: 1.0,
                        min_doc_id: 10,
                        max_doc_id: 800,
                    }],
                    max_value: 1.0,
                    max_doc_id: 800,
                    length: 32,
                },
            ],
            doc_ids_compressor: Box::new(EliasFanoCompressor {}),
            values_compressor: Box::new(Identity {}),
        };

        // Write
        let mut buf = Vec::new();
        info.write_binary(&mut buf).unwrap();

        // Read back
        let mut cursor = std::io::Cursor::new(&buf);
        let loaded = CompressedIndexInformation::read_binary(&mut cursor).unwrap();

        // Verify
        assert_eq!(loaded.terms.len(), 2);
        assert_eq!(loaded.terms[0].pages.len(), 2);
        assert_eq!(loaded.terms[0].pages[0].length, 128);
        assert_eq!(loaded.terms[0].pages[0].docid_position_range, (0, 100));
        assert_eq!(loaded.terms[0].pages[0].impact_position_range, (0, 200));
        assert!((loaded.terms[0].pages[0].max_value - 3.14).abs() < 1e-5);
        assert_eq!(loaded.terms[0].pages[0].min_doc_id, 0);
        assert_eq!(loaded.terms[0].pages[0].max_doc_id, 500);

        // Second block of first term
        assert_eq!(loaded.terms[0].pages[1].docid_position_range, (100, 180));
        assert_eq!(loaded.terms[0].pages[1].impact_position_range, (200, 350));

        // Second term
        assert_eq!(loaded.terms[1].pages[0].docid_position_range, (180, 220));
        assert_eq!(loaded.terms[1].pages[0].impact_position_range, (350, 400));

        // Size check: should be much smaller than CBOR
        eprintln!(
            "Binary metadata size: {} bytes (2 terms, 3 blocks)",
            buf.len()
        );
    }
}

pub struct CompressedIndex {
    information: CompressedIndexInformation,

    /// View on document IDs
    docid_buffer: Box<dyn Buffer>,

    /// View on document IDs
    impact_buffer: Box<dyn Buffer>,
}

//
// ---- Iterators over compressed block indices

pub struct CompressedIndexIterator<'a> {
    /// Iterator over page information
    info_iter: Box<std::slice::Iter<'a, TermBlockInformation>>,

    /// Current info
    info: Option<&'a TermBlockInformation>,

    /// Reusable buffer for decoded doc IDs
    docids: Vec<DocId>,

    /// Reusable buffer for decoded impact values
    impacts: Vec<ImpactValue>,

    /// Current position within the decoded arrays
    index: usize,

    /// Whether the current block has been decoded
    block_loaded: bool,

    // Term index (for reference)
    term_index: TermIndex,

    /// Our sparse index
    sparse_index: &'a CompressedIndex,
}

impl<'a> CompressedIndexIterator<'a> {
    fn new<'c: 'a>(index: &'c CompressedIndex, term_index: TermIndex) -> Self {
        let mut iter = if term_index < index.information.terms.len() {
            Box::new(index.information.terms[term_index].pages.iter())
        } else {
            Box::new([].iter())
        };

        let info = iter.next();

        Self {
            sparse_index: &index,
            info_iter: iter,
            info: info,
            docids: Vec::with_capacity(128),
            impacts: Vec::with_capacity(128),
            index: 0,
            block_loaded: false,
            term_index: term_index,
        }
    }

    /// Move the iterator to the first block where a document of
    /// at least `min_doc_id` is present.
    /// Does NOT decode the block — that happens lazily.
    fn move_iterator(&mut self, min_doc_id: DocId) -> bool {
        // Loop until the condition is met
        while let Some(info) = self.info {
            if info.max_doc_id >= min_doc_id {
                debug!(
                    "[{}] Moving iterator OK - max(doc_id) = {} >= {}",
                    self.term_index, info.max_doc_id, min_doc_id
                );
                return true;
            }

            // Go to the next block
            self.next_block();

            if let Some(info) = self.info {
                debug!("[{}] Read the next block (move): {}", self.term_index, info);
            } else {
                debug!("[{}] EOF for blocks (move)", self.term_index);
            }
        }
        false
    }

    /// Decode the current block into reusable buffers.
    ///
    /// Fast path: if the buffer supports direct byte access (in-memory),
    /// pass raw &[u8] slices to avoid Box<dyn Slice> heap allocations.
    fn decode_block(&mut self) {
        if let Some(info) = self.info {
            // Try fast path: direct byte access (no Box allocation)
            let docid_bytes = self.sparse_index.docid_buffer.as_bytes();
            let impact_bytes = self.sparse_index.impact_buffer.as_bytes();

            if let (Some(db), Some(ib)) = (docid_bytes, impact_bytes) {
                let docid_slice =
                    &db[info.docid_position_range.0 as usize..info.docid_position_range.1 as usize];
                self.sparse_index
                    .information
                    .doc_ids_compressor
                    .decode_into_bytes(docid_slice, self.term_index, info, &mut self.docids);

                let impact_slice = &ib
                    [info.impact_position_range.0 as usize..info.impact_position_range.1 as usize];
                self.sparse_index
                    .information
                    .values_compressor
                    .decode_into_bytes(impact_slice, self.term_index, info, &mut self.impacts);
            } else {
                // Fallback: Box<dyn Slice> path
                let slice = self.sparse_index.docid_buffer.slice(
                    info.docid_position_range.0 as usize,
                    info.docid_position_range.1 as usize,
                );
                self.sparse_index
                    .information
                    .doc_ids_compressor
                    .decode_into(slice, self.term_index, info, &mut self.docids);

                let slice = self.sparse_index.impact_buffer.slice(
                    info.impact_position_range.0 as usize,
                    info.impact_position_range.1 as usize,
                );
                self.sparse_index.information.values_compressor.decode_into(
                    slice,
                    self.term_index,
                    info,
                    &mut self.impacts,
                );
            }

            self.index = 0;
            self.block_loaded = true;
        }
    }

    /// Ensure the current block is decoded
    #[inline]
    fn ensure_block_loaded(&mut self) {
        if !self.block_loaded && self.info.is_some() {
            self.decode_block();
        }
    }

    /// Binary search for first doc >= min_doc_id from current position
    #[inline]
    fn find_geq(&self, min_doc_id: DocId) -> Option<usize> {
        let search_slice = &self.docids[self.index..];
        let pos = search_slice.partition_point(|&d| d < min_doc_id);
        if pos < search_slice.len() {
            Some(self.index + pos)
        } else {
            None
        }
    }

    /// Moves to the next block (invalidates decoded data without deallocating)
    fn next_block(&mut self) {
        self.info = self.info_iter.next();
        // Clear but keep capacity for reuse — no allocation on next decode
        self.docids.clear();
        self.impacts.clear();
        self.index = 0;
        self.block_loaded = false;
    }
}

impl<'a> Iterator for CompressedIndexIterator<'a> {
    type Item = TermImpact;

    fn next(&mut self) -> Option<Self::Item> {
        // Move to next block if current is exhausted
        if self.block_loaded && self.index >= self.docids.len() {
            self.next_block();
        }

        if self.info.is_none() {
            return None;
        }

        self.ensure_block_loaded();

        if self.index < self.docids.len() {
            let impact = TermImpact {
                docid: self.docids[self.index],
                value: self.impacts[self.index],
            };
            self.index += 1;
            Some(impact)
        } else {
            None
        }
    }
}

struct CompressedBlockTermImpactIterator<'a> {
    /// Iterator for this term
    iterator: RefCell<CompressedIndexIterator<'a>>,

    // Requested minimum document ID
    current_min_docid: Option<DocId>,

    // We need a RefCell for method current()
    current_value: RefCell<Option<TermImpact>>,

    // Maximum value over all postings
    max_value: ImpactValue,

    // Maximum document ID over all postings
    max_doc_id: DocId,

    // Number of postings
    length: usize,
}

impl<'a> CompressedBlockTermImpactIterator<'a> {
    fn new(index: &'a CompressedIndex, term_index: TermIndex) -> Self {
        let info = &index.information.terms[term_index];
        Self {
            iterator: RefCell::new(CompressedIndexIterator::new(index, term_index)),
            current_value: RefCell::new(None),
            max_value: info.max_value,
            max_doc_id: info.max_doc_id,
            length: info.length,
            current_min_docid: None,
        }
    }
}

impl<'a> BlockTermImpactIterator for CompressedBlockTermImpactIterator<'a> {
    fn next_min_doc_id(&mut self, min_doc_id: DocId) -> Option<DocId> {
        // Sets the current minimum document ID
        self.current_min_docid = Some(min_doc_id.max(
            if let Some(impact) = self.current_value.get_mut() {
                impact.docid + 1
            } else {
                0
            },
        ));
        let min_doc_id = self.current_min_docid.expect("Should not be None");

        // Move to the block having at least one document greater that min_doc_id
        if self.iterator.get_mut().move_iterator(min_doc_id) {
            debug!(
                "[{}] We have a candidate for doc_id >= {}",
                self.iterator.get_mut().term_index,
                min_doc_id
            );
            Some(self.min_block_doc_id())
        } else {
            debug!("[{}] End of iterator", self.iterator.get_mut().term_index);
            None
        }
    }

    /// Returns the current posting using binary search within the decoded block.
    fn current(&self) -> TermImpact {
        let min_docid = self.current_min_docid.expect("Should not be null");

        let mut current_value = self.current_value.borrow_mut();

        if current_value
            .and_then(|x| Some(x.docid < min_docid))
            .or(Some(true))
            .unwrap()
        {
            let mut iterator = self.iterator.borrow_mut();
            iterator.ensure_block_loaded();

            if let Some(pos) = iterator.find_geq(min_docid) {
                *current_value = Some(TermImpact {
                    docid: iterator.docids[pos],
                    value: iterator.impacts[pos],
                });
                iterator.index = pos + 1;
            } else {
                panic!("Did not find current impact for min_docid={}", min_docid);
            }
        }

        return current_value.expect("No current value");
    }

    fn max_value(&self) -> ImpactValue {
        return self.max_value;
    }

    fn max_block_doc_id(&self) -> DocId {
        self.iterator
            .borrow()
            .info
            .expect("Iterator was over")
            .max_doc_id
    }

    fn min_block_doc_id(&self) -> DocId {
        self.iterator
            .borrow()
            .info
            .expect("Iterator was over")
            .min_doc_id
    }

    fn max_block_value(&self) -> ImpactValue {
        self.iterator
            .borrow()
            .info
            .expect("Iterator was over")
            .max_value
    }

    fn max_doc_id(&self) -> DocId {
        return self.max_doc_id;
    }

    fn length(&self) -> usize {
        return self.length;
    }
}

impl SparseIndex for CompressedIndex {
    fn block_iterator(
        &self,
        term_index: crate::base::TermIndex,
    ) -> Box<dyn super::index::BlockTermImpactIterator + '_> {
        Box::new(CompressedBlockTermImpactIterator::new(self, term_index))
    }

    fn max_doc_id(&self) -> DocId {
        self.information
            .terms
            .iter()
            .map(|term| term.max_doc_id)
            .max()
            .unwrap_or(0)
    }
}

impl SparseIndexInformation for CompressedIndex {
    fn value_range(&self, term_ix: TermIndex) -> (ImpactValue, ImpactValue) {
        return (0., self.information.terms[term_ix].max_value);
    }
}

impl Len for CompressedIndex {
    fn len(&self) -> usize {
        self.information.terms.len()
    }
}

/// Transform that compresses a raw forward index into a block-based
/// compressed index with separate streams for document IDs and impact values.
pub struct CompressionTransform {
    /// Maximum number of postings per block.
    pub max_block_size: usize,

    /// Factory for creating the document ID compressor.
    pub doc_ids_compressor_factory: Box<dyn DocIdCompressorFactory>,

    /// Factory for creating the impact value compressor.
    pub impacts_compressor_factory: Box<dyn ImpactCompressorFactory>,
}

/// Result of compressing one term's posting list in parallel.
struct CompressedTermResult {
    info: TermBlocksInformation,
    docid_bytes: Vec<u8>,
    impact_bytes: Vec<u8>,
}

/// Chunk size for parallel compression (number of terms per batch).
const COMPRESSION_CHUNK_SIZE: usize = 10_000;

impl IndexTransform for CompressionTransform {
    /// Compress the impact values using chunked parallel processing.
    ///
    /// Terms are processed in chunks to limit memory usage: each chunk is
    /// compressed in parallel with rayon, written to disk, then freed.
    fn process(&self, path: &Path, index: &dyn SparseIndexView) -> Result<(), std::io::Error> {
        use std::io::BufWriter;

        // Create the directory if needed
        if !path.is_dir() {
            info!("Creating path {}", path.display());
            create_dir(path)?;
        }

        let doc_ids_compressor = self.doc_ids_compressor_factory.create(index);
        let values_compressor = self.impacts_compressor_factory.create(index);
        let max_block_size = self.max_block_size;

        let pb = ProgressBar::new(index.len() as u64);
        pb.set_style(
            ProgressStyle::default_bar()
                .template(
                    "{spinner:.green} [{elapsed_precise}] [{bar:40.cyan/blue}] {pos}/{len} ({eta})",
                )
                .progress_chars("=> "),
        );

        let mut impact_writer = BufWriter::new(
            File::options()
                .write(true)
                .truncate(true)
                .create(true)
                .open(path.join("impacts.dat"))
                .expect("Could not create the values file"),
        );

        let mut docid_writer = BufWriter::new(
            File::options()
                .write(true)
                .truncate(true)
                .create(true)
                .open(path.join("docids.dat"))
                .expect("Could not create the document IDs file"),
        );

        let mut terms_info = Vec::with_capacity(index.len());
        let mut docid_offset: u64 = 0;
        let mut impact_offset: u64 = 0;

        // Process terms in chunks to limit memory usage
        for chunk_start in (0..index.len()).step_by(COMPRESSION_CHUNK_SIZE) {
            let chunk_end = (chunk_start + COMPRESSION_CHUNK_SIZE).min(index.len());

            // Compress this chunk in parallel
            let chunk_results: Vec<CompressedTermResult> = (chunk_start..chunk_end)
                .into_par_iter()
                .map(|term_index| {
                    let mut docid_buf: Vec<u8> = Vec::new();
                    let mut impact_buf: Vec<u8> = Vec::new();

                    let mut it = index.iterator(term_index);
                    let mut flag = true;
                    let mut term_information = TermBlocksInformation {
                        pages: Vec::new(),
                        max_value: 0f32,
                        max_doc_id: 0,
                        length: 0,
                    };
                    let mut max_doc_id = 0;
                    let mut docid_position: u64 = 0;
                    let mut impact_position: u64 = 0;

                    while flag {
                        let mut impacts = Vec::new();
                        let mut docids = Vec::<DocId>::new();
                        flag = false;
                        let mut min_doc_id: DocId = DocId::MAX;

                        while let Some(ti) = it.next() {
                            if min_doc_id == DocId::MAX {
                                min_doc_id = ti.docid;
                            }
                            assert!(
                                (ti.docid > max_doc_id) || (max_doc_id == 0),
                                "{} is not greater than {}",
                                ti.docid,
                                max_doc_id
                            );
                            max_doc_id = ti.docid;
                            docids.push(ti.docid);
                            impacts.push(ti.value);
                            if docids.len() == max_block_size {
                                flag = true;
                                break;
                            }
                        }

                        if docids.is_empty() {
                            break;
                        }

                        let mut block_info = TermBlockInformation {
                            docid_position_range: (docid_position, 0),
                            impact_position_range: (impact_position, 0),
                            length: impacts.len(),
                            max_value: impacts.iter().fold(0f32, |cur, x| cur.max(*x)),
                            min_doc_id,
                            max_doc_id,
                        };

                        assert!(max_doc_id >= min_doc_id);

                        doc_ids_compressor.write(&mut docid_buf, &docids, term_index, &block_info);
                        values_compressor.write(&mut impact_buf, &impacts, term_index, &block_info);

                        docid_position = docid_buf.len() as u64;
                        block_info.docid_position_range.1 = docid_position;

                        impact_position = impact_buf.len() as u64;
                        block_info.impact_position_range.1 = impact_position;

                        term_information.max_value =
                            term_information.max_value.max(block_info.max_value);
                        term_information.max_doc_id =
                            term_information.max_doc_id.max(block_info.max_doc_id);
                        term_information.length += block_info.length;
                        term_information.pages.push(block_info);
                    }

                    pb.inc(1);

                    CompressedTermResult {
                        info: term_information,
                        docid_bytes: docid_buf,
                        impact_bytes: impact_buf,
                    }
                })
                .collect();

            // Write this chunk sequentially, fixing up byte positions
            for mut result in chunk_results {
                for page in result.info.pages.iter_mut() {
                    page.docid_position_range.0 += docid_offset;
                    page.docid_position_range.1 += docid_offset;
                    page.impact_position_range.0 += impact_offset;
                    page.impact_position_range.1 += impact_offset;
                }

                docid_writer.write_all(&result.docid_bytes)?;
                impact_writer.write_all(&result.impact_bytes)?;

                docid_offset += result.docid_bytes.len() as u64;
                impact_offset += result.impact_bytes.len() as u64;

                terms_info.push(result.info);
            }
        }

        pb.finish();

        let information = CompressedIndexInformation {
            terms: terms_info,
            doc_ids_compressor,
            values_compressor,
        };

        // Write compact binary metadata
        let meta_path = path.join("index.bin");
        eprintln!("Writing binary metadata to {}...", meta_path.display());
        let meta_file = std::io::BufWriter::new(
            File::options()
                .write(true)
                .create(true)
                .truncate(true)
                .open(&meta_path)?,
        );
        information.write_binary(&mut std::io::BufWriter::new(meta_file))?;

        // Write a minimal CBOR stub (just type tag + compressors, no term data)
        // so load_index dispatches to CompressedIndexLoader which reads index.bin
        let stub = CompressedIndexLoader {
            information: CompressedIndexInformation {
                terms: Vec::new(),
                doc_ids_compressor: information.doc_ids_compressor,
                values_compressor: information.values_compressor,
            },
        };
        save_index(Box::new(stub), path)
    }
}

#[derive(Serialize, Deserialize)]
struct CompressedIndexLoader {
    information: CompressedIndexInformation,
}

#[typetag::serde]
impl IndexLoader for CompressedIndexLoader {
    /// Loads a block-based index. Prefers binary metadata (index.bin) if available.
    fn into_index(self: Box<Self>, path: &Path, in_memory: bool) -> Box<dyn SparseIndex> {
        // Try loading binary metadata (compact), fall back to CBOR (from self)
        let bin_path = path.join("index.bin");
        let information = if bin_path.exists() {
            let mut reader =
                std::io::BufReader::new(File::open(&bin_path).expect("Failed to open index.bin"));
            CompressedIndexInformation::read_binary(&mut reader)
                .expect("Failed to read binary metadata")
        } else {
            self.information
        };

        let docid_path = path.join("docids.dat");
        let impact_path = path.join("impacts.dat");
        Box::new(CompressedIndex {
            information,
            docid_buffer: if in_memory {
                Box::new(MemoryBuffer::new(&docid_path))
            } else {
                Box::new(MmapBuffer::new(&docid_path))
            },
            impact_buffer: if in_memory {
                Box::new(MemoryBuffer::new(&impact_path))
            } else {
                Box::new(MmapBuffer::new(&impact_path))
            },
        })
    }
}
