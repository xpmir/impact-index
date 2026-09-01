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
    cell::{Cell, RefCell},
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
    scoring::ScoringFunction,
    search::cursor::TermCursor,
    utils::buffer::{Buffer, MemoryBuffer, MmapBuffer, Slice},
};
use indicatif::{ProgressBar, ProgressIterator, ProgressStyle};
use log::{debug, info};
use rayon::prelude::*;
use serde::{Deserialize, Serialize};

pub mod docid;
pub mod impact;
pub mod positions;

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

    /// Minimum document length among the documents stored in this block
    /// (P1a), used by dl-monotone scorers (BM25, LM-Dirichlet) for a tight,
    /// model-agnostic upper bound via
    /// [`crate::scoring::ScoringFunction::max_score_with_dl`].
    ///
    /// `0` means "not available" (docmeta wasn't present when the index was
    /// built/migrated) -- scorers treat that as a sentinel and fall back to
    /// their collection-wide bound, never as a literal document length of
    /// zero. Document lengths above `u16::MAX` saturate to `u16::MAX`; since
    /// this is always a *minimum*, saturating down-clamps a huge length to
    /// something smaller, which only ever loosens (never tightens past
    /// safe) the resulting bound.
    pub min_doc_length: u16,

    /// Position within the (flattened, per-block) position-delta stream in
    /// `positions.dat`. `(0, 0)` means this block has no positions (either
    /// the whole index has none, or it was read from a pre-positions v4
    /// binary). `#[serde(default)]` so the CBOR stub written by
    /// [`CompressionTransform::process`] (which carries no term data) and
    /// any value predating this field still deserialize.
    #[serde(default)]
    pub positions_position_range: (u64, u64),

    /// Sum of this block's tfs, i.e. the number of deltas the position
    /// codec must decode from `positions_position_range` -- carried
    /// alongside the byte range because a chunked codec ([`positions::BitPackingPositions`])
    /// can't recover the count from the byte length alone. `0` when
    /// `positions_position_range` is `(0, 0)`.
    #[serde(default)]
    pub num_positions: u32,
}

impl std::fmt::Display for TermBlockInformation {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(
            f,
            "(docids: {}-{}, impacts: {}-{}, len: {}, max_v: {}, docid: {}-{}, min_dl: {}, positions: {}-{} (n={}))",
            self.docid_position_range.0,
            self.docid_position_range.1,
            self.impact_position_range.0,
            self.impact_position_range.1,
            self.length,
            self.max_value,
            self.min_doc_id,
            self.max_doc_id,
            self.min_doc_length,
            self.positions_position_range.0,
            self.positions_position_range.1,
            self.num_positions,
        )
    }
}

//
// ---- Compression ---
//

/// Trait for encoding/decoding a sequence of values within a block.
pub trait Compressor<T>: Sync + Send {
    /// Human-readable codec name, used in manifest / diagnostics summaries.
    ///
    /// The default implementation derives it from the concrete type name
    /// (e.g. `EliasFanoCompressor`); this stays object-safe because it only
    /// uses `Self` as a type parameter inside the body, never in the
    /// signature, so each concrete compressor gets its own vtable entry
    /// without needing to implement this itself.
    fn codec_name(&self) -> &'static str {
        let full = std::any::type_name::<Self>();
        full.rsplit("::").next().unwrap_or(full)
    }

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
pub trait DocIdCompressor: Compressor<DocId> {
    /// Decode doc-id OFFSETS (relative to `info.min_doc_id`) into a reusable
    /// `u32` buffer (P5).
    ///
    /// All current codecs (BitPacking, PFOR, EliasFano) already encode
    /// offsets-from-min internally: a block spans at most `max_block_size`
    /// sorted, distinct document IDs, so `max_doc_id - min_doc_id` fits
    /// comfortably in 32 bits for any realistic collection/block-size
    /// combination — this is asserted in each concrete implementation's
    /// hot decode path. Storing offsets as `u32` (rather than absolute
    /// `u64` doc IDs) halves the decoded buffer's footprint and lets the
    /// SIMD bitpacker (`BitPacker4x`, whose native output unit is `u32`)
    /// decompress directly into the buffer with no scalar widening loop.
    ///
    /// The default implementation falls back to the `u64` decode path and
    /// narrows each value; concrete compressors should override this for
    /// a zero-widen fast path.
    fn decode_offsets_into(&self, data: &[u8], info: &TermBlockInformation, buffer: &mut Vec<u32>) {
        let mut tmp: Vec<DocId> = Vec::with_capacity(info.length);
        self.decode_into_bytes(data, 0, info, &mut tmp);
        buffer.clear();
        buffer.extend(tmp.iter().map(|&d| (d - info.min_doc_id) as u32));
    }
}

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
pub trait ImpactCompressor: Compressor<ImpactValue> {
    /// Whether decoding returns exactly the values that were encoded.
    ///
    /// Positional indices REQUIRE a lossless impact codec: the stored value
    /// is the term frequency, and position-run boundaries are recovered at
    /// read time from the *decoded* values
    /// ([`CompressedIndexIterator::ensure_positions_loaded`]). A lossy
    /// codec that returns `0.99998` for a stored `1.0` would shift every
    /// later run in the block -- silently, in release builds. Defaults to
    /// `false` so a new codec must opt in explicitly;
    /// [`CompressionTransform::process`] refuses to build a positional
    /// index with a codec that doesn't.
    fn lossless(&self) -> bool {
        false
    }
}

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
    /// Minimum of `pages[..].min_doc_length` (P1a), i.e. the minimum
    /// document length across the whole term, used for the term-level
    /// score bound. `0` if `pages` is empty or every block's
    /// `min_doc_length` is the "not available" sentinel. Always derived
    /// from `pages` (at build time or at [`CompressedIndexInformation::read_binary`]
    /// time) rather than stored as independent state.
    pub min_dl: u32,
}

/// Global information on the index structure
#[derive(Serialize, Deserialize)]
pub struct CompressedIndexInformation {
    pub terms: Vec<TermBlocksInformation>,
    doc_ids_compressor: Box<dyn DocIdCompressor>,
    values_compressor: Box<dyn ImpactCompressor>,
    /// `Some` iff the index has positions (`positions.dat` was written by
    /// [`CompressionTransform::process`]).
    positions_compressor: Option<Box<dyn positions::PositionsCompressor>>,
}

const COMPRESSED_INDEX_MAGIC: u32 = 0x49445832; // "IDX2"
/// v4 (P1a): adds a per-block `min_doc_length: u16` trailer to every block
/// record (see [`TermBlockInformation::min_doc_length`]). v3 -> v4
/// directories must be migrated via `manifest::update_index` (registered
/// as the `(1, migrate_v1_to_v2)` step) -- `read_binary` below refuses to
/// silently reinterpret v3 bytes.
///
/// v5 adds an optional position stream: the compressor header becomes a
/// 3-tuple `(doc_ids, values, positions)` and every block record gains a
/// `positions_position_range` byte-length + `num_positions` count (both
/// `0` when the index has no positions). v4 -> v5 directories must be
/// migrated via `manifest::update_index` (registered as the
/// `(2, migrate_v2_to_v3)` step, see `manifest.rs`) -- `read_binary` below
/// refuses to silently reinterpret v4 bytes; only [`read_binary_impl`] (via
/// [`migrate_compressed_v4_to_v5`]) understands the old 2-tuple/no-positions
/// layout, for the sake of migrating it away.
const COMPRESSED_INDEX_VERSION: u32 = 5;

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
    ///
    /// Format v4 (P1a) appends one more field per block:
    ///   `min_doc_length: u16` (see [`TermBlockInformation::min_doc_length`]).
    ///
    /// Format v5 (positions) changes the compressor header to a 3-tuple
    /// `(doc_ids, values, positions)` and appends two more fields per
    /// block: `positions_len: VInt` and `num_positions: VInt` (both `0`
    /// when the index has no positions) -- see
    /// [`TermBlockInformation::positions_position_range`] /
    /// [`TermBlockInformation::num_positions`].
    ///
    /// `pub` (rather than the usual module-private helper) so integration
    /// tests can inspect exactly what was written, e.g. to hand-verify
    /// migrated `min_doc_length` values (`tests/*.rs` link this crate
    /// externally and can't see private items).
    pub fn write_binary(&self, writer: &mut dyn Write) -> std::io::Result<()> {
        use byteorder::{LittleEndian, WriteBytesExt};

        // Header
        writer.write_u32::<LittleEndian>(COMPRESSED_INDEX_MAGIC)?;
        writer.write_u32::<LittleEndian>(COMPRESSED_INDEX_VERSION)?;
        writer.write_u32::<LittleEndian>(self.terms.len() as u32)?;

        // Compressor header (CBOR, small, only once)
        let mut compressor_buf = Vec::new();
        ciborium::ser::into_writer(
            &(
                &self.doc_ids_compressor,
                &self.values_compressor,
                &self.positions_compressor,
            ),
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
                // P1a: per-block minimum document length (0 = not available).
                writer.write_u16::<LittleEndian>(block.min_doc_length)?;
                // v5: position stream byte length + posting count (0/0 = none).
                let positions_len =
                    block.positions_position_range.1 - block.positions_position_range.0;
                write_vint(writer, positions_len)?;
                write_vint(writer, block.num_positions as u64)?;
            }
        }
        Ok(())
    }

    /// Writes the pre-positions (v4) binary layout: 2-tuple compressor
    /// header, no per-block position fields. `#[doc(hidden)]` -- used only
    /// by the migration integration test to synthesize a legacy v4 index
    /// to migrate from; the live writer is [`Self::write_binary`] (v5).
    ///
    /// Panics if `self.positions_compressor` is `Some`: v4 has no way to
    /// represent a positions codec, so writing this layout for a
    /// positional index would silently drop it.
    #[doc(hidden)]
    pub fn write_binary_v4(&self, writer: &mut dyn Write) -> std::io::Result<()> {
        use byteorder::{LittleEndian, WriteBytesExt};

        assert!(
            self.positions_compressor.is_none(),
            "write_binary_v4 cannot represent a positions compressor (v4 predates positions)"
        );

        // Header
        writer.write_u32::<LittleEndian>(COMPRESSED_INDEX_MAGIC)?;
        writer.write_u32::<LittleEndian>(4)?;
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

            let min_block_val = term
                .pages
                .iter()
                .map(|b| b.max_value)
                .fold(f32::INFINITY, f32::min);
            writer.write_f32::<LittleEndian>(min_block_val)?;
            let range = term.max_value - min_block_val;

            let mut prev_doc_id: u64 = 0;
            for block in &term.pages {
                let docid_len =
                    (block.docid_position_range.1 - block.docid_position_range.0) as u64;
                let impact_len =
                    (block.impact_position_range.1 - block.impact_position_range.0) as u64;
                write_vint(writer, docid_len)?;
                write_vint(writer, impact_len)?;
                write_vint(writer, block.length as u64)?;
                let q = if range > 0.0 {
                    (((block.max_value - min_block_val) / range * 255.0).ceil() as u32).min(255)
                        as u8
                } else {
                    255u8
                };
                writer.write_all(&[q])?;
                write_vint(writer, block.min_doc_id - prev_doc_id)?;
                write_vint(writer, block.max_doc_id - block.min_doc_id)?;
                prev_doc_id = block.max_doc_id;
                writer.write_u16::<LittleEndian>(block.min_doc_length)?;
            }
        }
        Ok(())
    }

    /// Read metadata from compact binary format.
    /// Reconstructs absolute byte positions via prefix sum.
    ///
    /// This is the *live* reader -- it rejects anything other than
    /// [`COMPRESSED_INDEX_VERSION`] via [`crate::manifest::check_format_version`],
    /// with an actionable error pointing at `Index.update`/`update_index`
    /// rather than silently misreading an older layout (see the `manifest`
    /// module docs). Older layouts are understood only by
    /// [`read_binary_impl`] (this method's shared body, parameterized by
    /// version) via the migration helpers that call it directly
    /// ([`migrate_add_min_dl`]'s `read_binary_v3` for v3,
    /// [`migrate_compressed_v4_to_v5`] for v4), and only for the sake of
    /// migrating them away.
    pub fn read_binary(reader: &mut dyn std::io::Read) -> std::io::Result<Self> {
        use byteorder::{LittleEndian, ReadBytesExt};

        let magic = reader.read_u32::<LittleEndian>()?;
        if magic != COMPRESSED_INDEX_MAGIC {
            return Err(std::io::Error::new(
                std::io::ErrorKind::InvalidData,
                "Not a compressed index binary file",
            ));
        }
        let version = reader.read_u32::<LittleEndian>()?;
        crate::manifest::check_format_version(version, COMPRESSED_INDEX_VERSION)?;
        Self::read_binary_impl(reader, COMPRESSED_INDEX_VERSION)
    }

    /// Shared body of [`Self::read_binary`] and [`migrate_compressed_v4_to_v5`],
    /// parameterized by the on-disk version (already consumed from the
    /// header by the caller): `4` reads the pre-positions 2-tuple
    /// compressor header and fills every block's position fields with the
    /// `(0, 0)`/`0` "none" sentinel; `5` reads the current 3-tuple header
    /// and the per-block position fields. Any other version is a
    /// programming error (callers must have already checked), not a
    /// reachable on-disk state.
    fn read_binary_impl(reader: &mut dyn std::io::Read, version: u32) -> std::io::Result<Self> {
        use byteorder::{LittleEndian, ReadBytesExt};

        let num_terms = reader.read_u32::<LittleEndian>()? as usize;

        // Compressor header
        let compressor_len = reader.read_u32::<LittleEndian>()? as usize;
        let mut compressor_buf = vec![0u8; compressor_len];
        reader.read_exact(&mut compressor_buf)?;
        let (doc_ids_compressor, values_compressor, positions_compressor): (
            Box<dyn DocIdCompressor>,
            Box<dyn ImpactCompressor>,
            Option<Box<dyn positions::PositionsCompressor>>,
        ) = if version >= 5 {
            ciborium::de::from_reader(&compressor_buf[..])
                .map_err(|e| std::io::Error::new(std::io::ErrorKind::InvalidData, e.to_string()))?
        } else {
            let (doc_ids, values): (Box<dyn DocIdCompressor>, Box<dyn ImpactCompressor>) =
                ciborium::de::from_reader(&compressor_buf[..]).map_err(|e| {
                    std::io::Error::new(std::io::ErrorKind::InvalidData, e.to_string())
                })?;
            (doc_ids, values, None)
        };

        // Read terms and blocks (VInt + delta encoded)
        let mut terms = Vec::with_capacity(num_terms);
        let mut docid_pos: u64 = 0;
        let mut impact_pos: u64 = 0;
        let mut positions_pos: u64 = 0;

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
                let min_doc_length = reader.read_u16::<LittleEndian>()?;

                let (positions_position_range, num_positions) = if version >= 5 {
                    let positions_len = read_vint(reader)?;
                    let num_positions = read_vint(reader)? as u32;
                    let range = (positions_pos, positions_pos + positions_len);
                    positions_pos += positions_len;
                    (range, num_positions)
                } else {
                    ((0u64, 0u64), 0u32)
                };

                pages.push(TermBlockInformation {
                    docid_position_range: (docid_pos, docid_pos + docid_len),
                    impact_position_range: (impact_pos, impact_pos + impact_len),
                    length: block_length,
                    max_value: block_max_value,
                    min_doc_id,
                    max_doc_id: block_max_doc_id,
                    min_doc_length,
                    positions_position_range,
                    num_positions,
                });

                docid_pos += docid_len;
                impact_pos += impact_len;
            }

            // Per-term min_dl (P1a) is always *derived* from the blocks,
            // never stored independently -- see `TermBlocksInformation::min_dl`.
            let min_dl = pages
                .iter()
                .map(|p| p.min_doc_length as u32)
                .min()
                .unwrap_or(0);

            terms.push(TermBlocksInformation {
                pages,
                max_value,
                max_doc_id,
                length,
                min_dl,
            });
        }

        Ok(CompressedIndexInformation {
            terms,
            doc_ids_compressor,
            values_compressor,
            positions_compressor,
        })
    }
}

/// Reads the pre-P1a (v3) binary layout: identical to
/// [`CompressedIndexInformation::read_binary`] except that per-block
/// records have no `min_doc_length` trailer. Every field it can't recover
/// (`min_doc_length`, `min_dl`) is filled with the `0` ("not available")
/// sentinel -- [`migrate_add_min_dl`] (the only caller) immediately
/// overwrites those from `docmeta` before writing the file back out in the
/// current (v4) layout.
///
/// This exists *only* for the v1->v2 manifest migration step -- the live
/// loader (`read_binary`) intentionally does *not* fall back to this: an
/// unmigrated v3 directory must fail with the actionable
/// "run Index.update" error, not be silently reinterpreted.
fn read_binary_v3(reader: &mut dyn std::io::Read) -> std::io::Result<CompressedIndexInformation> {
    use byteorder::{LittleEndian, ReadBytesExt};

    let magic = reader.read_u32::<LittleEndian>()?;
    if magic != COMPRESSED_INDEX_MAGIC {
        return Err(std::io::Error::new(
            std::io::ErrorKind::InvalidData,
            "Not a compressed index binary file",
        ));
    }
    let version = reader.read_u32::<LittleEndian>()?;
    if version != 3 {
        return Err(std::io::Error::new(
            std::io::ErrorKind::InvalidData,
            format!(
                "migrate_add_min_dl expected a legacy v3 compressed index, found v{} \
                 (index.bin should have been at v3 for the v1->v2 manifest migration step \
                 to apply -- this indicates a manifest/binary version mismatch)",
                version
            ),
        ));
    }
    let num_terms = reader.read_u32::<LittleEndian>()? as usize;

    let compressor_len = reader.read_u32::<LittleEndian>()? as usize;
    let mut compressor_buf = vec![0u8; compressor_len];
    reader.read_exact(&mut compressor_buf)?;
    let (doc_ids_compressor, values_compressor): (
        Box<dyn DocIdCompressor>,
        Box<dyn ImpactCompressor>,
    ) = ciborium::de::from_reader(&compressor_buf[..])
        .map_err(|e| std::io::Error::new(std::io::ErrorKind::InvalidData, e.to_string()))?;

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
            let mut q_byte = [0u8; 1];
            reader.read_exact(&mut q_byte)?;
            let block_max_value = min_block_val + (q_byte[0] as f32 / 255.0) * range;
            let min_doc_id = prev_doc_id + read_vint(reader)?;
            let block_max_doc_id = min_doc_id + read_vint(reader)?;
            prev_doc_id = block_max_doc_id;
            // No min_doc_length trailer in v3 -- filled in by the caller.

            pages.push(TermBlockInformation {
                docid_position_range: (docid_pos, docid_pos + docid_len),
                impact_position_range: (impact_pos, impact_pos + impact_len),
                length: block_length,
                max_value: block_max_value,
                min_doc_id,
                max_doc_id: block_max_doc_id,
                min_doc_length: 0,
                positions_position_range: (0, 0),
                num_positions: 0,
            });

            docid_pos += docid_len;
            impact_pos += impact_len;
        }

        terms.push(TermBlocksInformation {
            pages,
            max_value,
            max_doc_id,
            length,
            min_dl: 0,
        });
    }

    Ok(CompressedIndexInformation {
        terms,
        doc_ids_compressor,
        values_compressor,
        // v3 predates positions entirely.
        positions_compressor: None,
    })
}

/// P1a migration step (registered as `(1, migrate_v1_to_v2)` in
/// `manifest::update_index`'s step table): rewrites a compressed index
/// directory's `index.bin` from the v3 layout (no per-block
/// `min_doc_length`) to v4, recomputing that statistic from `docmeta.*`
/// (if present) by decoding each block's existing doc-id bytes -- the
/// posting files themselves (`docids.dat`/`impacts.dat`) are never
/// touched, only the metadata file is rewritten. Streams one term's blocks
/// at a time (via an mmap over `docids.dat`) rather than holding the whole
/// index's postings in memory.
///
/// If `docmeta.*` is missing, every block/term gets `min_doc_length = 0`,
/// which is exactly the old global-min-based bound behavior via
/// `ScoringFunction::max_score_with_dl`'s `min_dl == 0` fallback -- still
/// correct, just not tighter. If `index.bin` doesn't exist at all (this
/// directory isn't a compressed-index directory, e.g. a `Split` wrapper's
/// own directory, which has no `index.bin` of its own), this is a no-op.
pub(crate) fn migrate_add_min_dl(path: &Path) -> std::io::Result<()> {
    use byteorder::{LittleEndian, ReadBytesExt};

    let bin_path = path.join("index.bin");
    if !bin_path.exists() {
        return Ok(());
    }

    // Peek the on-disk binary version before committing to the v3 layout:
    // a manifest-less directory can be genuinely legacy (v3 binary, no
    // min_doc_length -- the case this migration exists for) or can already
    // carry a newer (v4 or v5) binary layout with just a missing/stale
    // manifest (e.g. rebuilt with this library but never re-stamped) --
    // that needs no data rewrite at all. Compared against the literal `3`
    // (not `COMPRESSED_INDEX_VERSION`, which has since moved past v4): this
    // step only ever understands the v3 -> v4 transition, regardless of
    // how many format versions exist beyond that.
    let version = {
        let mut reader = std::io::BufReader::new(File::open(&bin_path)?);
        let _magic = reader.read_u32::<LittleEndian>()?;
        reader.read_u32::<LittleEndian>()?
    };
    if version != 3 {
        return Ok(());
    }

    let mut info = {
        let mut reader = std::io::BufReader::new(File::open(&bin_path)?);
        read_binary_v3(&mut reader)?
    };

    let doc_meta = crate::docmeta::DocMetadata::load(path).ok();
    let docid_buffer = doc_meta
        .is_some()
        .then(|| MmapBuffer::new(&path.join("docids.dat")));

    for term in info.terms.iter_mut() {
        let mut term_min_dl = u32::MAX;
        let mut offsets: Vec<u32> = Vec::new();

        for block in term.pages.iter_mut() {
            let min_doc_length: u16 = match (&doc_meta, &docid_buffer) {
                (Some(dm), Some(buf)) => {
                    let bytes = buf
                        .as_bytes()
                        .expect("MmapBuffer::as_bytes always returns Some");
                    let slice = &bytes[block.docid_position_range.0 as usize
                        ..block.docid_position_range.1 as usize];
                    info.doc_ids_compressor
                        .decode_offsets_into(slice, block, &mut offsets);

                    offsets
                        .iter()
                        .map(|&o| {
                            dm.doc_lengths
                                .get((block.min_doc_id + o as DocId) as usize)
                                .copied()
                                .unwrap_or(0)
                        })
                        .min()
                        .unwrap_or(0)
                        .min(u16::MAX as u32) as u16
                }
                _ => 0,
            };
            block.min_doc_length = min_doc_length;
            term_min_dl = term_min_dl.min(min_doc_length as u32);
        }

        term.min_dl = if term.pages.is_empty() {
            0
        } else {
            term_min_dl
        };
    }

    // Write to a temp file, then rename over index.bin: a crash/interrupt
    // partway through never leaves a truncated or half-written metadata
    // file in place.
    //
    // `info.write_binary` now writes the *current* (v5) layout, not v4 --
    // `info.positions_compressor` is `None` (v3 predates positions), so
    // this just stamps a v5 header/block trailer with empty position
    // fields. That's correct, not a mismatch with this step's own v1->v2
    // manifest version: the manifest-level v2->v3 step (`migrate_v2_to_v3`
    // in `manifest.rs`) will subsequently find the binary already at v5
    // and no-op its own data rewrite, then stamp manifest v3. Each step in
    // the chain only cares that the binary is at *least* as new as what it
    // produces, not exactly equal.
    let tmp_path = path.join("index.bin.migrating");
    {
        let mut writer = std::io::BufWriter::new(File::create(&tmp_path)?);
        info.write_binary(&mut writer)?;
    }
    std::fs::rename(&tmp_path, &bin_path)?;

    Ok(())
}

/// v4 -> v5 migration step (registered as `(2, migrate_v2_to_v3)` in
/// `manifest::update_index`'s step table, see `manifest.rs`): rewrites a
/// compressed index directory's `index.bin` from the pre-positions v4
/// layout (2-tuple compressor header, no per-block position fields) to v5.
/// A v4 index has no positions by construction, so this is a metadata-only
/// rewrite -- `positions_compressor` becomes `None` and every block's
/// `positions_position_range`/`num_positions` become `(0, 0)`/`0`; the
/// posting files themselves (`docids.dat`/`impacts.dat`) are never
/// touched, and no `positions.dat` is created.
///
/// No-op if `index.bin` doesn't exist (e.g. a `Split` wrapper's own
/// directory) or is already at the current version (idempotent, and covers
/// the case where the binary was rebuilt with this library but its
/// manifest is merely stale).
pub(crate) fn migrate_compressed_v4_to_v5(path: &Path) -> std::io::Result<()> {
    use byteorder::{LittleEndian, ReadBytesExt};

    let bin_path = path.join("index.bin");
    if !bin_path.exists() {
        return Ok(());
    }

    let version = {
        let mut reader = std::io::BufReader::new(File::open(&bin_path)?);
        let _magic = reader.read_u32::<LittleEndian>()?;
        reader.read_u32::<LittleEndian>()?
    };
    if version == COMPRESSED_INDEX_VERSION {
        return Ok(());
    }

    let info = {
        let mut reader = std::io::BufReader::new(File::open(&bin_path)?);
        let _magic = reader.read_u32::<LittleEndian>()?;
        let _version = reader.read_u32::<LittleEndian>()?;
        CompressedIndexInformation::read_binary_impl(&mut reader, version)?
    };

    let tmp_path = path.join("index.bin.migrating");
    {
        let mut writer = std::io::BufWriter::new(File::create(&tmp_path)?);
        info.write_binary(&mut writer)?;
    }
    std::fs::rename(&tmp_path, &bin_path)?;

    Ok(())
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
                            min_doc_length: 5,
                            positions_position_range: (0, 0),
                            num_positions: 0,
                        },
                        TermBlockInformation {
                            docid_position_range: (100, 180),
                            impact_position_range: (200, 350),
                            length: 64,
                            max_value: 2.71,
                            min_doc_id: 501,
                            max_doc_id: 999,
                            min_doc_length: 12,
                            positions_position_range: (0, 0),
                            num_positions: 0,
                        },
                    ],
                    max_value: 3.14,
                    max_doc_id: 999,
                    length: 192,
                    min_dl: 5,
                },
                TermBlocksInformation {
                    pages: vec![TermBlockInformation {
                        docid_position_range: (180, 220),
                        impact_position_range: (350, 400),
                        length: 32,
                        max_value: 1.0,
                        min_doc_id: 10,
                        max_doc_id: 800,
                        min_doc_length: 3,
                        positions_position_range: (0, 0),
                        num_positions: 0,
                    }],
                    max_value: 1.0,
                    max_doc_id: 800,
                    length: 32,
                    min_dl: 3,
                },
            ],
            doc_ids_compressor: Box::new(EliasFanoCompressor {}),
            values_compressor: Box::new(Identity {}),
            positions_compressor: None,
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
        assert_eq!(loaded.terms[0].pages[0].min_doc_length, 5);
        assert_eq!(loaded.terms[0].pages[1].min_doc_length, 12);
        assert_eq!(loaded.terms[0].min_dl, 5);
        assert_eq!(loaded.terms[1].pages[0].min_doc_length, 3);
        assert_eq!(loaded.terms[1].min_dl, 3);

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

    /// View on impact values
    impact_buffer: Box<dyn Buffer>,

    /// View on position deltas, `Some` iff `positions.dat` exists (i.e. the
    /// index was built from a positional source). Never touched unless a
    /// caller actually asks for positions -- see
    /// [`CompressedIndexIterator::ensure_positions_loaded`].
    positions_buffer: Option<Box<dyn Buffer>>,

    /// Source directory path
    source_dir: Option<std::path::PathBuf>,

    /// Document metadata (auto-loaded if present)
    doc_meta: Option<crate::docmeta::DocMetadata>,

    /// Analyzer config (auto-loaded if present)
    analyzer_config: Option<crate::vocab::analyzer::AnalyzerConfig>,

    /// Document-id reorder map (`new_docid -> original_docid`), auto-loaded
    /// if present (i.e. this index was produced by `ReorderTransform`).
    reorder_map: Option<Vec<DocId>>,
}

//
// ---- Iterators over compressed block indices

pub struct CompressedIndexIterator<'a> {
    /// Iterator over page information
    info_iter: Box<std::slice::Iter<'a, TermBlockInformation>>,

    /// Current info
    info: Option<&'a TermBlockInformation>,

    /// Reusable buffer for decoded doc IDs, stored as `u32` OFFSETS from the
    /// current block's `min_doc_id` (P5). All current codecs (BitPacking,
    /// PFOR, EliasFano) already encode offsets-from-min that fit in `u32` by
    /// construction: a block spans at most `max_block_size` sorted, distinct
    /// document IDs, so `max_doc_id - min_doc_id` fits comfortably in 32
    /// bits for any realistic collection/block-size combination. Keeping the
    /// buffer as `u32` halves its footprint versus `u64` and lets the SIMD
    /// bitpacker (`BitPacker4x`) decompress directly into it with no scalar
    /// widening loop. The absolute `DocId` is reconstructed only at access
    /// points via `info.min_doc_id + offset as DocId`.
    docids: Vec<u32>,

    /// Reusable buffer for decoded impact values
    impacts: Vec<ImpactValue>,

    /// Current position (cursor) within the decoded arrays
    index: usize,

    /// Whether the current block has been decoded
    block_loaded: bool,

    /// Flattened, decoded (absolute, not delta) positions for every
    /// posting of the current block, filled lazily by
    /// [`Self::ensure_positions_loaded`] -- untouched for a query that
    /// never asks for positions.
    positions_flat: Vec<u32>,

    /// Prefix sums of `impacts` (tf per posting) for the current block:
    /// posting `i`'s run is `positions_flat[run_offsets[i]..run_offsets[i + 1]]`.
    /// Length is block length + 1.
    run_offsets: Vec<u32>,

    /// Whether `positions_flat`/`run_offsets` are valid for the current
    /// block (reset in [`Self::next_block`]).
    positions_loaded: bool,

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
            positions_flat: Vec::new(),
            run_offsets: Vec::new(),
            positions_loaded: false,
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
                    .decode_offsets_into(docid_slice, info, &mut self.docids);

                let impact_slice = &ib
                    [info.impact_position_range.0 as usize..info.impact_position_range.1 as usize];
                self.sparse_index
                    .information
                    .values_compressor
                    .decode_into_bytes(impact_slice, self.term_index, info, &mut self.impacts);
            } else {
                // Fallback: Box<dyn Slice> path. `Slice::data()` already
                // returns a contiguous `&[u8]`, so the u32-offset decode
                // path can be used directly here too.
                let slice = self.sparse_index.docid_buffer.slice(
                    info.docid_position_range.0 as usize,
                    info.docid_position_range.1 as usize,
                );
                self.sparse_index
                    .information
                    .doc_ids_compressor
                    .decode_offsets_into(slice.data(), info, &mut self.docids);

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

    /// Galloping (exponential) search for the first offset >= `min_offset`,
    /// starting at cursor position `start` within the decoded block (P4).
    ///
    /// This replaces the previous `partition_point` binary search over the
    /// whole remaining slice: since callers overwhelmingly advance to the
    /// very next posting (sequential WAND/MaxScore traversal), the fast
    /// path below is a single comparison (O(1)); a larger jump costs
    /// O(log gap) via exponential probing followed by a bounded binary
    /// search, rather than O(log n) over the whole remaining block.
    #[inline]
    fn gallop_geq(&self, start: usize, min_offset: u32) -> Option<usize> {
        let docids = &self.docids;
        let len = docids.len();
        if start >= len {
            return None;
        }

        // Fast path: already positioned, or the very next posting matches.
        if docids[start] >= min_offset {
            return Some(start);
        }

        // Exponential probing to bracket the target.
        let mut prev = start;
        let mut bound = 1usize;
        let mut cur = start + bound;
        while cur < len && docids[cur] < min_offset {
            prev = cur;
            bound *= 2;
            cur = start + bound;
        }
        let hi = cur.min(len);
        let lo = prev + 1;

        if lo >= hi {
            return if hi < len { Some(hi) } else { None };
        }

        // Bounded binary search within the bracket (lo, hi).
        let pos = docids[lo..hi].partition_point(|&d| d < min_offset);
        let result = lo + pos;
        if result < len {
            Some(result)
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
        self.positions_loaded = false;
    }

    /// Decodes the current block's position runs into
    /// `positions_flat`/`run_offsets`, if not already done. Never called
    /// from the hot WAND/MaxScore path -- only from
    /// [`CompressedBlockTermImpactIterator::positions`], itself only
    /// reachable via [`BlockTermImpactIterator::positions`], which nothing
    /// in the search algorithms calls.
    ///
    /// Requires the block already decoded (`ensure_block_loaded`): tfs come
    /// from `self.impacts`, already-scored or not (this iterator only ever
    /// holds raw values -- scoring happens in a wrapping iterator, see
    /// `scoring::ScoringBlockIterator`).
    fn ensure_positions_loaded(&mut self) {
        if self.positions_loaded {
            return;
        }
        let info = self.info.expect("block should be loaded");

        let positions_buffer = self
            .sparse_index
            .positions_buffer
            .as_ref()
            .expect("ensure_positions_loaded called on an index with no positions.dat");
        let positions_compressor = self
            .sparse_index
            .information
            .positions_compressor
            .as_ref()
            .expect("ensure_positions_loaded called on an index with no positions compressor");

        let data = positions_buffer
            .as_bytes()
            .expect("positions buffer should expose contiguous bytes");
        let slice = &data
            [info.positions_position_range.0 as usize..info.positions_position_range.1 as usize];
        positions_compressor.decode_into(
            slice,
            info.num_positions as usize,
            &mut self.positions_flat,
        );

        // Run boundaries from this block's tfs (the stored value IS the tf
        // for a positional/BoW index -- asserted at build time in
        // `CompressionTransform::process`).
        self.run_offsets.clear();
        self.run_offsets.reserve(self.impacts.len() + 1);
        self.run_offsets.push(0);
        let mut cumulative = 0u32;
        for &tf in &self.impacts {
            // `round`, not truncation: even a lossless integer codec goes
            // through f32, and the transform-time gate (`lossless()`)
            // guarantees the decoded value is the exact tf -- rounding
            // keeps that true against any benign representation change.
            cumulative += tf.round() as u32;
            self.run_offsets.push(cumulative);
        }
        // Hard check (not debug_assert): this is the cold positions-decode
        // path, and a mismatch here means every later run in the block
        // would be silently mis-sliced -- fail loudly instead.
        assert_eq!(
            cumulative, info.num_positions,
            "sum of block tfs ({}) does not match stored num_positions ({}): \
             positional data is corrupt (was this index compressed with a \
             lossy impact codec by an older library version?)",
            cumulative, info.num_positions
        );

        // Deltas -> absolute positions, per run (first stays, rest add the
        // previous absolute position within the same run).
        for w in self.run_offsets.windows(2) {
            let (start, end) = (w[0] as usize, w[1] as usize);
            let mut prev = 0u32;
            for p in &mut self.positions_flat[start..end] {
                *p += prev;
                prev = *p;
            }
        }

        self.positions_loaded = true;
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
            let min_doc_id = self.info.expect("block should be loaded").min_doc_id;
            let impact = TermImpact {
                docid: min_doc_id + self.docids[self.index] as DocId,
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
    /// Decoder/cursor for this term's blocks. Needs interior mutability
    /// because `current()` (`&self`) may need to decode a block and/or
    /// advance the cursor the first time it is called after a
    /// `next_min_doc_id` that actually moved forward.
    iterator: RefCell<CompressedIndexIterator<'a>>,

    /// Requested minimum document ID, updated only by `next_min_doc_id`
    /// (`&mut self`).
    current_min_docid: DocId,

    /// Whether a posting has ever been resolved (so `current_docid` below
    /// is meaningful).
    has_current: Cell<bool>,

    /// Cached resolved posting (split into plain `Cell<Copy>` fields rather
    /// than a single `Cell<Option<TermImpact>>`: cheaper to read/write and
    /// avoids `Option` matching on the hot path).
    current_docid: Cell<DocId>,
    current_value: Cell<ImpactValue>,

    /// Cached block metadata (updated in next_min_doc_id, no RefCell needed)
    cached_block_min_doc_id: DocId,
    cached_block_max_doc_id: DocId,
    cached_block_max_value: ImpactValue,
    /// Cached current block's minimum document length (P1a), 0 = unknown.
    cached_block_min_dl: u32,

    // Maximum value over all postings
    max_value: ImpactValue,

    // Maximum document ID over all postings
    max_doc_id: DocId,

    // Number of postings
    length: usize,

    /// Minimum document length across the whole term (P1a), 0 = unknown.
    min_dl: u32,

    /// Copy of `current()`'s position run, filled lazily by
    /// [`BlockTermImpactIterator::positions`]. A copy (not a borrow of the
    /// iterator's decoded block) because the block buffers live behind the
    /// `RefCell`'s interior, and `positions(&mut self)` still has to
    /// return a plain `&[u32]` -- the run is tiny (a document's tf), so
    /// this is cheap.
    current_positions: Vec<u32>,
}

impl<'a> CompressedBlockTermImpactIterator<'a> {
    fn new(index: &'a CompressedIndex, term_index: TermIndex) -> Self {
        let info = &index.information.terms[term_index];
        Self {
            iterator: RefCell::new(CompressedIndexIterator::new(index, term_index)),
            current_min_docid: 0,
            has_current: Cell::new(false),
            current_docid: Cell::new(0),
            current_value: Cell::new(0.0),
            cached_block_min_doc_id: 0,
            cached_block_max_doc_id: 0,
            cached_block_max_value: 0.0,
            cached_block_min_dl: 0,
            max_value: info.max_value,
            max_doc_id: info.max_doc_id,
            length: info.length,
            min_dl: info.min_dl,
            current_positions: Vec::new(),
        }
    }
}

impl<'a> BlockTermImpactIterator for CompressedBlockTermImpactIterator<'a> {
    fn next_min_doc_id(&mut self, min_doc_id: DocId) -> Option<DocId> {
        // Sets the current minimum document ID. This is a *shallow* move:
        // it only picks the right block (block-skip logic unchanged) and
        // does not itself resolve/advance the intra-block cursor — that
        // stays lazy, in `current()`, so that repeated `next_min_doc_id`
        // calls (or a `current()` call with no intervening
        // `next_min_doc_id`, as `transforms/split.rs` does) remain
        // idempotent and never skip a posting.
        let min_doc_id = min_doc_id.max(if self.has_current.get() {
            self.current_docid.get() + 1
        } else {
            0
        });
        self.current_min_docid = min_doc_id;

        // Move to the block having at least one document >= min_doc_id.
        // `get_mut` (not `borrow_mut`): we already hold `&mut self` here.
        let iterator = self.iterator.get_mut();
        if iterator.move_iterator(min_doc_id) {
            // Cache block metadata (avoids RefCell borrow in tight loops)
            let info = iterator.info.expect("Iterator has block");
            self.cached_block_min_doc_id = info.min_doc_id;
            self.cached_block_max_doc_id = info.max_doc_id;
            self.cached_block_max_value = info.max_value;
            self.cached_block_min_dl = info.min_doc_length as u32;

            debug!(
                "[{}] We have a candidate for doc_id >= {}",
                iterator.term_index, min_doc_id
            );
            Some(self.cached_block_min_doc_id)
        } else {
            debug!("[{}] End of iterator", iterator.term_index);
            None
        }
    }

    /// Returns the current posting.
    ///
    /// If already positioned at (or past) `current_min_docid` — the common
    /// case when `current()` is called more than once without an
    /// intervening `next_min_doc_id`, e.g. by `transforms/split.rs` — this
    /// is a plain read of the cached fields (P4): no search, no re-check
    /// of the decoded block. Otherwise, resolves the exact posting via a
    /// galloping (exponential) search from the cursor's current position,
    /// which is O(1) when the target is the very next posting (the common
    /// case for sequential WAND/MaxScore advance) and O(log gap) otherwise
    /// — replacing the previous `partition_point` binary search over the
    /// whole remaining block.
    fn current(&self) -> TermImpact {
        let min_docid = self.current_min_docid;

        if !self.has_current.get() || self.current_docid.get() < min_docid {
            let mut iterator = self.iterator.borrow_mut();
            iterator.ensure_block_loaded();

            let info = iterator.info.expect("Iterator has block");
            let min_offset = min_docid.saturating_sub(info.min_doc_id) as u32;
            let start = iterator.index;
            let pos = iterator.gallop_geq(start, min_offset).unwrap_or_else(|| {
                panic!("Did not find current impact for min_docid={}", min_docid)
            });

            let docid = info.min_doc_id + iterator.docids[pos] as DocId;
            let value = iterator.impacts[pos];
            iterator.index = pos + 1;

            self.current_docid.set(docid);
            self.current_value.set(value);
            self.has_current.set(true);
        }

        TermImpact {
            docid: self.current_docid.get(),
            value: self.current_value.get(),
        }
    }

    #[inline]
    fn max_value(&self) -> ImpactValue {
        self.max_value
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
    fn max_block_value(&self) -> ImpactValue {
        self.cached_block_max_value
    }

    #[inline]
    fn max_doc_id(&self) -> DocId {
        self.max_doc_id
    }

    #[inline]
    fn min_dl(&self) -> u32 {
        self.min_dl
    }

    #[inline]
    fn min_block_dl(&self) -> u32 {
        self.cached_block_min_dl
    }

    #[inline]
    fn length(&self) -> usize {
        self.length
    }

    /// Positions of `current()`'s document, `None` if the index has no
    /// positions. Not on the hot path: never called by WAND/MaxScore, and
    /// touches `positions.dat` (mmap slice + codec decode) only when
    /// actually invoked.
    fn positions(&mut self) -> Option<&[u32]> {
        // Resolve the current posting first (mirrors `current()`'s own
        // contract): afterwards `iterator.index` is `pos + 1`, so the
        // block-local index of the resolved posting is `iterator.index - 1`.
        self.current();

        let has_positions = {
            let iterator = self.iterator.borrow();
            iterator.sparse_index.positions_buffer.is_some()
                && iterator
                    .sparse_index
                    .information
                    .positions_compressor
                    .is_some()
        };
        if !has_positions {
            return None;
        }

        let mut iterator = self.iterator.borrow_mut();
        iterator.ensure_block_loaded();
        iterator.ensure_positions_loaded();

        let posting_ix = iterator.index - 1;
        let start = iterator.run_offsets[posting_ix] as usize;
        let end = iterator.run_offsets[posting_ix + 1] as usize;

        // Copy the (tiny) run out: `iterator` is borrowed from the
        // `RefCell`, so we can't return a reference into it directly.
        self.current_positions.clear();
        self.current_positions
            .extend_from_slice(&iterator.positions_flat[start..end]);
        Some(&self.current_positions)
    }
}

impl CompressedIndex {
    /// Number of postings for a term (used as BM25's document frequency).
    pub(crate) fn term_length(&self, term_index: TermIndex) -> u64 {
        self.information.terms[term_index].length as u64
    }

    /// Builds a monomorphized, scored cursor over a term's postings (P3):
    /// a concrete `TermCursor` combining block movement, the galloping
    /// intra-block cursor (P4/P5) and batched scoring (P1b), with no
    /// `Box<dyn _>` or `Cell`-based interior mutability anywhere in the
    /// hot path. `S` is a concrete scorer type (e.g. `BM25TermScorer`), not
    /// `Box<dyn ScoringFunction>`, so the whole chain inlines through a
    /// generic search loop instantiated for this `(CompressedIndex, S)`
    /// pair.
    pub(crate) fn typed_cursor<S: ScoringFunction>(
        &self,
        term_index: TermIndex,
        scorer: S,
    ) -> CompressedScoringCursor<'_, S> {
        CompressedScoringCursor::new(self, term_index, scorer)
    }
}

/// Concrete, monomorphizable (cursor x scorer) combination for
/// [`CompressedIndex`] (P3), with lazy batched scoring in chunks (P1b).
///
/// Mirrors [`CompressedBlockTermImpactIterator`]'s shallow-move/lazy-current
/// contract, but:
/// - takes `&mut self` throughout (no `RefCell`/`Cell`: this type isn't
///   required to be object-safe), and
/// - resolves scores in chunks (`ScoringFunction::score_chunk`) from the
///   cursor's current position rather than one posting — or one whole
///   128-posting block — at a time, so block-max pruning that only touches
///   a handful of postings per block doesn't pay to score the rest.
pub(crate) struct CompressedScoringCursor<'a, S: ScoringFunction> {
    /// Block decoder/cursor (shared with the dyn path).
    iterator: CompressedIndexIterator<'a>,

    /// Concrete scorer for this term (e.g. `BM25TermScorer`).
    scorer: S,

    /// Requested minimum document ID, updated only by `next_min_doc_id`.
    current_min_docid: DocId,
    has_current: bool,
    current_docid: DocId,
    /// Position of the resolved posting within the decoded block arrays.
    current_pos: usize,

    /// Lazily-filled batch of scores (P1b), covering
    /// `[chunk_start, chunk_start + chunk_len)` of the current block.
    chunk: Vec<f32>,
    chunk_start: usize,
    chunk_len: usize,

    /// Cached block metadata (already scored).
    cached_block_min_doc_id: DocId,
    cached_block_max_doc_id: DocId,
    cached_block_max_value: ImpactValue,

    /// Term-level metadata (already scored).
    max_value: ImpactValue,
    max_doc_id: DocId,
    length: usize,
}

/// Number of postings scored per batch (P1b): small enough that block-max
/// pruning skipping most of a 128-posting block doesn't waste work, large
/// enough to amortize the scorer call and let the per-chunk loop vectorize.
const SCORE_CHUNK_SIZE: usize = 32;

impl<'a, S: ScoringFunction> CompressedScoringCursor<'a, S> {
    fn new(index: &'a CompressedIndex, term_index: TermIndex, scorer: S) -> Self {
        let info = &index.information.terms[term_index];
        // Term-level bound tightened with the term's own min_dl (P1a).
        let max_value = scorer.max_score_with_dl(info.max_value, info.min_dl);
        Self {
            iterator: CompressedIndexIterator::new(index, term_index),
            scorer,
            current_min_docid: 0,
            has_current: false,
            current_docid: 0,
            current_pos: 0,
            chunk: vec![0.0; SCORE_CHUNK_SIZE],
            chunk_start: 0,
            chunk_len: 0,
            cached_block_min_doc_id: 0,
            cached_block_max_doc_id: 0,
            cached_block_max_value: 0.0,
            max_value,
            max_doc_id: info.max_doc_id,
            length: info.length,
        }
    }

    /// Scores a chunk of `SCORE_CHUNK_SIZE` postings (or fewer, at the tail
    /// of the block) starting at `start_pos`.
    #[inline]
    fn fill_chunk(&mut self, start_pos: usize) {
        let info = self.iterator.info.expect("block should be loaded");
        let end = (start_pos + SCORE_CHUNK_SIZE).min(self.iterator.docids.len());
        let n = end - start_pos;
        self.scorer.score_chunk(
            info.min_doc_id,
            &self.iterator.docids[start_pos..end],
            &self.iterator.impacts[start_pos..end],
            &mut self.chunk[..n],
        );
        self.chunk_start = start_pos;
        self.chunk_len = n;

        // P1a safety net: the block bound (computed with min_doc_length,
        // possibly f16-adjusted -- see `BM25TermScorer::max_score_with_dl`)
        // must dominate every real score in the block, not just the ones a
        // particular query happens to touch.
        #[cfg(debug_assertions)]
        {
            let bound = self.cached_block_max_value;
            for &s in &self.chunk[..n] {
                debug_assert!(
                    s <= bound + bound.abs() * 1e-3 + 1e-4,
                    "chunk score {} exceeds block bound {} (P1a safety violation)",
                    s,
                    bound
                );
            }
        }
    }
}

impl<'a, S: ScoringFunction> TermCursor for CompressedScoringCursor<'a, S> {
    fn next_min_doc_id(&mut self, min_doc_id: DocId) -> Option<DocId> {
        let min_doc_id = min_doc_id.max(if self.has_current {
            self.current_docid + 1
        } else {
            0
        });
        self.current_min_docid = min_doc_id;

        let had_block = self.iterator.info.is_some();
        let prev_block_min_doc_id = self.cached_block_min_doc_id;

        if self.iterator.move_iterator(min_doc_id) {
            let info = self.iterator.info.expect("Iterator has block");
            self.cached_block_min_doc_id = info.min_doc_id;
            self.cached_block_max_doc_id = info.max_doc_id;
            // Block-level bound tightened with this block's own min_dl (P1a).
            self.cached_block_max_value = self
                .scorer
                .max_score_with_dl(info.max_value, info.min_doc_length as u32);

            // A block-max value is scored lazily by `current()`'s chunking;
            // invalidate any stale chunk from a previous block.
            if !had_block || self.cached_block_min_doc_id != prev_block_min_doc_id {
                self.chunk_len = 0;
            }

            Some(self.cached_block_min_doc_id)
        } else {
            None
        }
    }

    fn current(&mut self) -> TermImpact {
        let min_docid = self.current_min_docid;

        if !self.has_current || self.current_docid < min_docid {
            self.iterator.ensure_block_loaded();

            let info = self.iterator.info.expect("Iterator has block");
            let min_offset = min_docid.saturating_sub(info.min_doc_id) as u32;
            let start = self.iterator.index;
            let pos = self
                .iterator
                .gallop_geq(start, min_offset)
                .unwrap_or_else(|| {
                    panic!("Did not find current impact for min_docid={}", min_docid)
                });

            let docid = info.min_doc_id + self.iterator.docids[pos] as DocId;
            self.current_docid = docid;
            self.current_pos = pos;
            self.iterator.index = pos + 1;
            self.has_current = true;

            // Ensure the batch (P1b) covers `pos`, computing a new one
            // lazily from the cursor position if it doesn't.
            if self.chunk_len == 0
                || pos < self.chunk_start
                || pos >= self.chunk_start + self.chunk_len
            {
                self.fill_chunk(pos);
            }
        }

        let local = self.current_pos - self.chunk_start;
        TermImpact {
            docid: self.current_docid,
            value: self.chunk[local],
        }
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
    fn max_block_value(&self) -> ImpactValue {
        self.cached_block_max_value
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
        self.length
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

    fn doc_meta(&self) -> Option<&crate::docmeta::DocMetadata> {
        self.doc_meta.as_ref()
    }

    fn analyzer_config(&self) -> Option<&crate::vocab::analyzer::AnalyzerConfig> {
        self.analyzer_config.as_ref()
    }

    fn source_path(&self) -> Option<&std::path::Path> {
        self.source_dir.as_deref()
    }

    fn reorder_map(&self) -> Option<&Vec<DocId>> {
        self.reorder_map.as_ref()
    }

    fn as_any(&self) -> &dyn std::any::Any {
        self
    }

    fn has_positions(&self) -> bool {
        // Both must be present: a compressor with no buffer (shouldn't
        // happen -- `process` only records one when it also wrote the
        // file) or a buffer with no compressor (a stale `positions.dat`
        // left behind by something else) are each treated as "no
        // positions" rather than panicking later on the mismatched half.
        self.positions_buffer.is_some() && self.information.positions_compressor.is_some()
    }

    fn positions_iterator<'a>(
        &'a self,
        term_index: TermIndex,
    ) -> Option<Box<dyn Iterator<Item = Vec<u32>> + 'a>> {
        if !SparseIndex::has_positions(self) {
            return None;
        }
        Some(Box::new(crate::index::BlockPositionsAdapter(
            self.block_iterator(term_index),
        )))
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

    /// Codec for the position stream, used only when the source view has
    /// positions ([`SparseIndexView::has_positions`]). `None` means: use
    /// [`positions::BitPackingPositions`] (the default) when the source
    /// has positions, and write no position stream at all otherwise --
    /// unlike the doc-id/impact compressors there is no factory here (no
    /// build-time statistic needs inspecting to choose a position codec),
    /// so this is a plain optional codec rather than a
    /// `PositionsCompressorFactory`.
    pub positions_codec: Option<Box<dyn positions::PositionsCompressor>>,
}

/// Result of compressing one term's posting list in parallel.
struct CompressedTermResult {
    info: TermBlocksInformation,
    docid_bytes: Vec<u8>,
    impact_bytes: Vec<u8>,
    /// Empty when the index has no positions.
    positions_bytes: Vec<u8>,
}

/// Chunk size for parallel compression (number of terms per batch).
const COMPRESSION_CHUNK_SIZE: usize = 10_000;

/// Resolves the codec to use for a build with positions: the configured
/// [`CompressionTransform::positions_codec`], or [`positions::BitPackingPositions`]
/// (the default) when unset.
///
/// `process` only ever has a `&dyn PositionsCompressor` borrow of a
/// configured codec (through `&self`, which it cannot move out of); a
/// `typetag`-serialized codec round-trips through CBOR by construction
/// (that's how `index.bin` itself stores one), so serializing it and
/// reading the bytes straight back produces an independent owned instance
/// to store in [`CompressedIndexInformation`] without requiring
/// [`positions::PositionsCompressor`] to be `Clone`.
fn resolve_positions_compressor(
    configured: Option<&dyn positions::PositionsCompressor>,
) -> Box<dyn positions::PositionsCompressor> {
    match configured {
        None => Box::new(positions::BitPackingPositions),
        Some(codec) => {
            let mut buf = Vec::new();
            ciborium::ser::into_writer(codec, &mut buf)
                .expect("serialize positions codec for cloning");
            ciborium::de::from_reader(&buf[..]).expect("deserialize cloned positions codec")
        }
    }
}

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

        // Positions are opt-in per source index: a query that never calls
        // `positions()` must never read a byte of `positions.dat`, so a
        // non-positional index gets byte-identical output to before this
        // feature existed -- no file created, no compressor recorded.
        let has_positions = index.has_positions();
        if has_positions && !values_compressor.lossless() {
            return Err(std::io::Error::new(
                std::io::ErrorKind::InvalidInput,
                "positional indices require a lossless impact codec (the stored value is the \
                 term frequency, and position-run boundaries are recovered from decoded \
                 values): use Identity or BitPackedIntCompressor instead of a quantizer, or \
                 rebuild the source index without positions",
            ));
        }
        let positions_codec: Option<Box<dyn positions::PositionsCompressor>> =
            has_positions.then(|| resolve_positions_compressor(self.positions_codec.as_deref()));
        let positions_codec_ref: Option<&dyn positions::PositionsCompressor> =
            positions_codec.as_deref();

        // P1a: per-block minimum document length, computed here (at build
        // time) when doc lengths are available. `None` (no docmeta) means
        // every block gets `min_doc_length = 0`, i.e. the pre-P1a
        // collection-wide-min behavior via `ScoringFunction::max_score_with_dl`'s
        // `min_dl == 0` fallback -- still correct, just not tighter.
        let doc_meta = index.doc_meta();

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

        // Only created when the source has positions -- an index without
        // positions never gets a `positions.dat` at all.
        let mut positions_writer: Option<BufWriter<File>> = has_positions.then(|| {
            BufWriter::new(
                File::options()
                    .write(true)
                    .truncate(true)
                    .create(true)
                    .open(path.join("positions.dat"))
                    .expect("Could not create the positions file"),
            )
        });

        let mut terms_info = Vec::with_capacity(index.len());
        let mut docid_offset: u64 = 0;
        let mut impact_offset: u64 = 0;
        let mut positions_offset: u64 = 0;

        // Process terms in chunks to limit memory usage
        for chunk_start in (0..index.len()).step_by(COMPRESSION_CHUNK_SIZE) {
            let chunk_end = (chunk_start + COMPRESSION_CHUNK_SIZE).min(index.len());

            // Compress this chunk in parallel
            let chunk_results: Vec<CompressedTermResult> = (chunk_start..chunk_end)
                .into_par_iter()
                .map(|term_index| {
                    let mut docid_buf: Vec<u8> = Vec::new();
                    let mut impact_buf: Vec<u8> = Vec::new();
                    let mut positions_buf: Vec<u8> = Vec::new();

                    let mut it = index.iterator(term_index);
                    // Aligned 1:1 with `it` (same length/order, see
                    // `SparseIndexView::positions_iterator`'s contract):
                    // every posting pulled from `it` below also pulls one
                    // run from `pos_it`.
                    let mut pos_it = has_positions.then(|| {
                        index.positions_iterator(term_index).expect(
                            "has_positions() true but positions_iterator returned None for a term",
                        )
                    });
                    let mut flag = true;
                    let mut term_information = TermBlocksInformation {
                        pages: Vec::new(),
                        max_value: 0f32,
                        max_doc_id: 0,
                        length: 0,
                        min_dl: 0,
                    };
                    let mut max_doc_id = 0;
                    let mut docid_position: u64 = 0;
                    let mut impact_position: u64 = 0;
                    let mut positions_position: u64 = 0;

                    while flag {
                        let mut impacts = Vec::new();
                        let mut docids = Vec::<DocId>::new();
                        // Flattened position deltas for this block (Lucene
                        // `.pos` model): per posting, first position
                        // absolute then gaps, runs concatenated in posting
                        // order -- boundaries are implied by each
                        // posting's tf (`impacts[i] as usize`), not stored.
                        let mut block_deltas: Vec<u32> = Vec::new();
                        let mut block_num_positions: u32 = 0;
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

                            if let Some(pos_it) = pos_it.as_mut() {
                                let run = pos_it.next().expect(
                                    "positions_iterator ended before the impact iterator for the same term",
                                );
                                assert_eq!(
                                    ti.value as usize,
                                    run.len(),
                                    "positional index: stored value must equal tf \
                                     (positions.len()) for docid {}",
                                    ti.docid
                                );
                                let mut prev = 0u32;
                                for (i, &p) in run.iter().enumerate() {
                                    block_deltas.push(if i == 0 { p } else { p - prev });
                                    prev = p;
                                }
                                block_num_positions += run.len() as u32;
                            }

                            if docids.len() == max_block_size {
                                flag = true;
                                break;
                            }
                        }

                        if docids.is_empty() {
                            break;
                        }

                        // P1a: minimum document length among this block's
                        // documents. Saturates to `u16::MAX` for lengths
                        // beyond that (rounding a huge length DOWN to
                        // something smaller only ever loosens, never
                        // unsafely tightens, the resulting score bound); a
                        // missing per-doc length (shouldn't happen when
                        // `doc_meta` is `Some` and complete) falls back to
                        // `0`, the same safe direction.
                        let min_doc_length: u16 = match doc_meta {
                            Some(dm) => docids
                                .iter()
                                .map(|&d| dm.doc_lengths.get(d as usize).copied().unwrap_or(0))
                                .min()
                                .unwrap_or(0)
                                .min(u16::MAX as u32)
                                as u16,
                            None => 0,
                        };

                        let mut block_info = TermBlockInformation {
                            docid_position_range: (docid_position, 0),
                            impact_position_range: (impact_position, 0),
                            length: impacts.len(),
                            max_value: impacts.iter().fold(0f32, |cur, x| cur.max(*x)),
                            min_doc_id,
                            max_doc_id,
                            min_doc_length,
                            positions_position_range: (positions_position, 0),
                            num_positions: block_num_positions,
                        };

                        assert!(max_doc_id >= min_doc_id);

                        doc_ids_compressor.write(&mut docid_buf, &docids, term_index, &block_info);
                        values_compressor.write(&mut impact_buf, &impacts, term_index, &block_info);

                        docid_position = docid_buf.len() as u64;
                        block_info.docid_position_range.1 = docid_position;

                        impact_position = impact_buf.len() as u64;
                        block_info.impact_position_range.1 = impact_position;

                        if let Some(codec) = positions_codec_ref {
                            codec.write(&mut positions_buf, &block_deltas);
                            positions_position = positions_buf.len() as u64;
                            block_info.positions_position_range.1 = positions_position;
                        }

                        term_information.max_value =
                            term_information.max_value.max(block_info.max_value);
                        term_information.max_doc_id =
                            term_information.max_doc_id.max(block_info.max_doc_id);
                        term_information.length += block_info.length;
                        term_information.pages.push(block_info);
                    }

                    term_information.min_dl = term_information
                        .pages
                        .iter()
                        .map(|p| p.min_doc_length as u32)
                        .min()
                        .unwrap_or(0);

                    pb.inc(1);

                    CompressedTermResult {
                        info: term_information,
                        docid_bytes: docid_buf,
                        impact_bytes: impact_buf,
                        positions_bytes: positions_buf,
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
                    page.positions_position_range.0 += positions_offset;
                    page.positions_position_range.1 += positions_offset;
                }

                docid_writer.write_all(&result.docid_bytes)?;
                impact_writer.write_all(&result.impact_bytes)?;
                if let Some(w) = positions_writer.as_mut() {
                    w.write_all(&result.positions_bytes)?;
                }

                docid_offset += result.docid_bytes.len() as u64;
                impact_offset += result.impact_bytes.len() as u64;
                positions_offset += result.positions_bytes.len() as u64;

                terms_info.push(result.info);
            }
        }

        pb.finish();
        if let Some(w) = positions_writer.as_mut() {
            w.flush()?;
        }

        let information = CompressedIndexInformation {
            terms: terms_info,
            doc_ids_compressor,
            values_compressor,
            positions_compressor: positions_codec,
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
                positions_compressor: information.positions_compressor,
            },
        };

        // Record the on-disk format version + build parameters so a future
        // format change can be detected on load (see `manifest` module).
        let mut codecs = format!(
            "{}+{}",
            stub.information.doc_ids_compressor.codec_name(),
            stub.information.values_compressor.codec_name(),
        );
        if let Some(pc) = &stub.information.positions_compressor {
            codecs.push('+');
            codecs.push_str(pc.codec_name());
        }
        let builder_info = crate::manifest::BuilderInfo::new()
            .with_block_size(max_block_size)
            .with_codecs(codecs);
        if has_positions {
            crate::manifest::write_manifest_with_features(
                path,
                crate::manifest::IndexKind::Compressed,
                builder_info,
                vec!["positions".to_string()],
            )?;
        } else {
            crate::manifest::write_manifest(
                path,
                crate::manifest::IndexKind::Compressed,
                builder_info,
            )?;
        }

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
        // Auto-detect auxiliary components
        let doc_meta = crate::docmeta::DocMetadata::load(path).ok();
        let analyzer_config = {
            let cfg = crate::vocab::analyzer::TextAnalyzer::load_config(path);
            if cfg.stemmer == "none" && !cfg.stop_words {
                None
            } else {
                Some(cfg)
            }
        };

        let reorder_map = crate::transforms::reorder::ReorderMap::load(path).ok();

        // `positions.dat` is only opened (mmap'd or loaded into memory) if
        // it exists; a non-positional index never even has this file, and
        // this is the ONLY place its bytes are touched during loading --
        // it's not decoded until a caller actually asks for positions.
        let positions_path = path.join("positions.dat");
        let positions_buffer: Option<Box<dyn Buffer>> = if positions_path.exists() {
            Some(if in_memory {
                Box::new(MemoryBuffer::new(&positions_path))
            } else {
                Box::new(MmapBuffer::new(&positions_path))
            })
        } else {
            None
        };

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
            positions_buffer,
            source_dir: Some(path.to_path_buf()),
            doc_meta,
            analyzer_config,
            reorder_map,
        })
    }
}
