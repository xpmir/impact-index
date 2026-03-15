//! Document ID compression schemes.
//!
//! - [`EliasFanoCompressor`]: Elias-Fano encoding via the `sucds` crate
//! - [`BitPackingCompressor`]: SIMD bitpacking via the `bitpacking` crate (faster)

use std::io::Write;

use super::{Compressor, DocIdCompressor, DocIdCompressorFactory, TermBlockInformation};
use crate::{
    base::{DocId, TermIndex},
    index::SparseIndexView,
    utils::buffer::Slice,
};
use bitpacking::{BitPacker, BitPacker4x};
use ouroboros::self_referencing;
use serde::{Deserialize, Serialize};
use sucds::{EliasFano, EliasFanoBuilder, Searial};

/// Compresses document IDs using Elias-Fano encoding.
///
/// Within each block, document IDs are stored as offsets from the minimum
/// document ID, achieving near-optimal compression for sorted integer sequences.
#[derive(Serialize, Deserialize, Clone, Copy)]
pub struct EliasFanoCompressor {}

#[typetag::serde]
impl DocIdCompressor for EliasFanoCompressor {}

impl DocIdCompressorFactory for EliasFanoCompressor {
    fn create(&self, _index: &dyn SparseIndexView) -> Box<dyn DocIdCompressor> {
        Box::new(EliasFanoCompressor {})
    }

    fn clone(&self) -> Box<dyn DocIdCompressorFactory> {
        Box::new(Clone::clone(self))
    }
}

#[self_referencing]
struct EliasFanoIterator {
    data: EliasFano,
    min_doc_id: DocId,
    #[borrows(data)]
    #[covariant]
    pub iterator: sucds::elias_fano::iter::Iter<'this>,
}

unsafe impl<'a> Send for EliasFanoIterator {}

impl<'a> Iterator for EliasFanoIterator {
    type Item = DocId;

    fn next(&mut self) -> Option<Self::Item> {
        self.with_mut(|fields| {
            if let Some(x) = fields.iterator.next() {
                Some((x as DocId) + *fields.min_doc_id)
            } else {
                None
            }
        })
    }
}

impl Compressor<DocId> for EliasFanoCompressor {
    fn write(
        &self,
        writer: &mut dyn Write,
        values: &[DocId],
        _term_index: TermIndex,
        info: &TermBlockInformation,
    ) {
        let mut c = EliasFanoBuilder::new(
            (info.max_doc_id - info.min_doc_id + 1) as usize,
            values.len(),
        )
        .expect("Error when building");

        for &x in values {
            c.push((x - info.min_doc_id) as usize)
                .expect("Could not add a doc ID");
        }
        c.build()
            .serialize_into(writer)
            .expect("Error while serializing");
    }

    fn read<'a>(
        &self,
        slice: Box<dyn Slice + 'a>,
        _term_index: TermIndex,
        info: &TermBlockInformation,
    ) -> Box<dyn Iterator<Item = DocId> + Send + 'a> {
        let data = EliasFano::deserialize_from(slice.data()).expect("Error while reading");
        Box::new(
            EliasFanoIteratorBuilder {
                data: data,
                min_doc_id: info.min_doc_id,
                iterator_builder: |data: &EliasFano| data.iter(0),
            }
            .build(),
        )
    }
}

// ---------------------------------------------------------------------------
// BitPacking compressor (SIMD-accelerated)
// ---------------------------------------------------------------------------

const BITPACK_BLOCK_LEN: usize = 128; // BitPacker4x::BLOCK_LEN

/// Marker byte indicating a tail block (< 128 postings) stored as raw u32 gaps.
const BITPACK_TAIL_MARKER: u8 = 0xFF;

/// Compresses document IDs using SIMD bitpacking (SSE3/scalar fallback).
///
/// Full blocks (128 doc IDs) are delta-encoded and packed with
/// [`BitPacker4x`]. Tail blocks (< 128) use raw little-endian u32 gaps.
///
/// Format per block:
/// - Full block: `[num_bits: u8] [packed data: 16 * num_bits bytes]`
/// - Tail block: `[0xFF marker: u8] [u32 gaps in little-endian: 4 * length bytes]`
#[derive(Serialize, Deserialize, Clone, Copy)]
pub struct BitPackingCompressor {}

#[typetag::serde]
impl DocIdCompressor for BitPackingCompressor {}

impl DocIdCompressorFactory for BitPackingCompressor {
    fn create(&self, _index: &dyn SparseIndexView) -> Box<dyn DocIdCompressor> {
        Box::new(BitPackingCompressor {})
    }

    fn clone(&self) -> Box<dyn DocIdCompressorFactory> {
        Box::new(Clone::clone(self))
    }
}

impl Compressor<DocId> for BitPackingCompressor {
    fn write(
        &self,
        writer: &mut dyn Write,
        values: &[DocId],
        _term_index: TermIndex,
        info: &TermBlockInformation,
    ) {
        // Convert to u32 offsets from min_doc_id
        let offsets: Vec<u32> = values
            .iter()
            .map(|&x| (x - info.min_doc_id) as u32)
            .collect();

        if offsets.len() == BITPACK_BLOCK_LEN {
            // Full block: SIMD bitpacking with sorted delta encoding
            let bitpacker = BitPacker4x::new();
            let num_bits = bitpacker.num_bits_sorted(0, &offsets);
            writer.write_all(&[num_bits]).expect("write num_bits");

            if num_bits > 0 {
                let mut compressed = vec![0u8; BITPACK_BLOCK_LEN * 4];
                let written = bitpacker.compress_sorted(0, &offsets, &mut compressed, num_bits);
                writer
                    .write_all(&compressed[..written])
                    .expect("write packed");
            }
        } else {
            // Tail block: store as raw u32 gaps
            writer
                .write_all(&[BITPACK_TAIL_MARKER])
                .expect("write marker");
            let mut prev = 0u32;
            for &v in &offsets {
                let gap = v - prev;
                writer.write_all(&gap.to_le_bytes()).expect("write gap");
                prev = v;
            }
        }
    }

    fn read<'a>(
        &self,
        slice: Box<dyn Slice + 'a>,
        _term_index: TermIndex,
        info: &TermBlockInformation,
    ) -> Box<dyn Iterator<Item = DocId> + Send + 'a> {
        let data = slice.data();
        let marker = data[0];
        let min_doc_id = info.min_doc_id;
        let length = info.length;

        if marker != BITPACK_TAIL_MARKER && length == BITPACK_BLOCK_LEN {
            // Full block: SIMD decompress all 128 at once
            let num_bits = marker;
            let mut decompressed = vec![0u32; BITPACK_BLOCK_LEN];
            if num_bits > 0 {
                let bitpacker = BitPacker4x::new();
                bitpacker.decompress_sorted(0, &data[1..], &mut decompressed, num_bits);
            }
            // decompressed contains offsets from min_doc_id
            Box::new(
                decompressed
                    .into_iter()
                    .map(move |v| v as DocId + min_doc_id),
            )
        } else {
            // Tail block: read raw u32 gaps and prefix-sum
            let mut values = Vec::with_capacity(length);
            let mut cumulative = 0u32;
            for i in 0..length {
                let offset = 1 + i * 4;
                let gap = u32::from_le_bytes([
                    data[offset],
                    data[offset + 1],
                    data[offset + 2],
                    data[offset + 3],
                ]);
                cumulative += gap;
                values.push(cumulative as DocId + min_doc_id);
            }
            Box::new(values.into_iter())
        }
    }

    fn decode_into_bytes(
        &self,
        data: &[u8],
        _term_index: TermIndex,
        info: &TermBlockInformation,
        buffer: &mut Vec<DocId>,
    ) {
        let marker = data[0];
        let min_doc_id = info.min_doc_id;

        buffer.clear();

        if marker != BITPACK_TAIL_MARKER && info.length == BITPACK_BLOCK_LEN {
            // Full block: SIMD decompress into a temp u32 buffer, then convert
            let num_bits = marker;
            let mut decompressed = [0u32; BITPACK_BLOCK_LEN];
            if num_bits > 0 {
                let bitpacker = BitPacker4x::new();
                bitpacker.decompress_sorted(0, &data[1..], &mut decompressed, num_bits);
            }
            buffer.reserve(BITPACK_BLOCK_LEN);
            for &v in &decompressed {
                buffer.push(v as DocId + min_doc_id);
            }
        } else {
            // Tail block: read raw u32 gaps and prefix-sum
            buffer.reserve(info.length);
            let mut cumulative = 0u32;
            for i in 0..info.length {
                let offset = 1 + i * 4;
                let gap = u32::from_le_bytes([
                    data[offset],
                    data[offset + 1],
                    data[offset + 2],
                    data[offset + 3],
                ]);
                cumulative += gap;
                buffer.push(cumulative as DocId + min_doc_id);
            }
        }
    }
}

// ---------------------------------------------------------------------------
// PFOR (Patched Frame-of-Reference) compressor
// ---------------------------------------------------------------------------

/// PFOR-delta compressor: bitpack at a base bit-width, store outliers separately.
///
/// For blocks where most deltas are small but a few are large, this achieves
/// better compression than uniform bitpacking (which uses max bit-width for all).
///
/// Format per full block (128 delta-encoded doc ID offsets):
/// - `[base_bits: u8]` — bit-width for bitpacked values
/// - `[num_exceptions: u8]` — number of outlier values
/// - `[bitpacked 128 clamped deltas: 16 * base_bits bytes]`
/// - `[exceptions: num_exceptions * 5 bytes (1 byte index + 4 byte LE value)]`
///
/// Tail blocks (< 128): same raw u32 gap format as BitPackingCompressor.
#[derive(Serialize, Deserialize, Clone, Copy)]
pub struct PForCompressor {}

#[typetag::serde]
impl DocIdCompressor for PForCompressor {}

impl DocIdCompressorFactory for PForCompressor {
    fn create(&self, _index: &dyn SparseIndexView) -> Box<dyn DocIdCompressor> {
        Box::new(PForCompressor {})
    }
    fn clone(&self) -> Box<dyn DocIdCompressorFactory> {
        Box::new(Clone::clone(self))
    }
}

impl PForCompressor {
    /// Compute deltas from sorted offsets.
    fn compute_deltas(offsets: &[u32]) -> Vec<u32> {
        let mut deltas = Vec::with_capacity(offsets.len());
        let mut prev = 0u32;
        for &v in offsets {
            deltas.push(v - prev);
            prev = v;
        }
        deltas
    }

    /// Find the optimal base bit-width that minimizes total encoded size.
    fn optimal_base_bits(deltas: &[u32]) -> u8 {
        let mut best_bits = 0u8;
        let mut best_cost = usize::MAX;

        for b in 0..=32u8 {
            let mask = if b >= 32 { u32::MAX } else { (1u32 << b) - 1 };
            let num_exceptions = deltas.iter().filter(|&&d| d > mask).count();
            let cost = (BITPACK_BLOCK_LEN * b as usize) / 8 + num_exceptions * 5;
            if cost < best_cost {
                best_cost = cost;
                best_bits = b;
            }
        }
        best_bits
    }

    fn decode_block(&self, data: &[u8], info: &TermBlockInformation, buffer: &mut Vec<DocId>) {
        let min_doc_id = info.min_doc_id;
        buffer.clear();

        let base_bits = data[0];
        let num_exceptions = data[1] as usize;

        if base_bits == BITPACK_TAIL_MARKER {
            // Tail block
            buffer.reserve(info.length);
            let mut cumulative = 0u32;
            for i in 0..info.length {
                let offset = 2 + i * 4;
                let gap = u32::from_le_bytes([
                    data[offset],
                    data[offset + 1],
                    data[offset + 2],
                    data[offset + 3],
                ]);
                cumulative += gap;
                buffer.push(cumulative as DocId + min_doc_id);
            }
            return;
        }

        // Full block: decompress bitpacked deltas
        let mut deltas = [0u32; BITPACK_BLOCK_LEN];
        let packed_size = if base_bits > 0 {
            let bitpacker = BitPacker4x::new();
            bitpacker.decompress(&data[2..], &mut deltas, base_bits);
            (BITPACK_BLOCK_LEN * base_bits as usize) / 8
        } else {
            0
        };

        // Patch exceptions
        let exc_start = 2 + packed_size;
        for e in 0..num_exceptions {
            let offset = exc_start + e * 5;
            let idx = data[offset] as usize;
            let val = u32::from_le_bytes([
                data[offset + 1],
                data[offset + 2],
                data[offset + 3],
                data[offset + 4],
            ]);
            deltas[idx] = val;
        }

        // Prefix-sum to reconstruct offsets
        buffer.reserve(BITPACK_BLOCK_LEN);
        let mut cumulative = 0u32;
        for &d in &deltas {
            cumulative += d;
            buffer.push(cumulative as DocId + min_doc_id);
        }
    }
}

impl Compressor<DocId> for PForCompressor {
    fn write(
        &self,
        writer: &mut dyn Write,
        values: &[DocId],
        _term_index: TermIndex,
        info: &TermBlockInformation,
    ) {
        let offsets: Vec<u32> = values
            .iter()
            .map(|&x| (x - info.min_doc_id) as u32)
            .collect();

        if offsets.len() == BITPACK_BLOCK_LEN {
            let deltas = Self::compute_deltas(&offsets);
            let base_bits = Self::optimal_base_bits(&deltas);
            let mask = if base_bits >= 32 {
                u32::MAX
            } else {
                (1u32 << base_bits) - 1
            };

            let exceptions: Vec<(u8, u32)> = deltas
                .iter()
                .enumerate()
                .filter(|(_, &d)| d > mask)
                .map(|(i, &d)| (i as u8, d))
                .collect();

            writer.write_all(&[base_bits]).expect("write base_bits");
            writer
                .write_all(&[exceptions.len() as u8])
                .expect("write num_exc");

            if base_bits > 0 {
                let clamped: Vec<u32> = deltas.iter().map(|&d| d & mask).collect();
                let bitpacker = BitPacker4x::new();
                let mut compressed = vec![0u8; BITPACK_BLOCK_LEN * 4];
                let written = bitpacker.compress(&clamped, &mut compressed, base_bits);
                writer
                    .write_all(&compressed[..written])
                    .expect("write packed");
            }

            for &(idx, val) in &exceptions {
                writer.write_all(&[idx]).expect("write exc idx");
                writer.write_all(&val.to_le_bytes()).expect("write exc val");
            }
        } else {
            // Tail block
            writer
                .write_all(&[BITPACK_TAIL_MARKER])
                .expect("write marker");
            writer.write_all(&[0u8]).expect("write dummy");
            let mut prev = 0u32;
            for &v in &offsets {
                let gap = v - prev;
                writer.write_all(&gap.to_le_bytes()).expect("write gap");
                prev = v;
            }
        }
    }

    fn read<'a>(
        &self,
        slice: Box<dyn Slice + 'a>,
        _term_index: TermIndex,
        info: &TermBlockInformation,
    ) -> Box<dyn Iterator<Item = DocId> + Send + 'a> {
        let mut buffer = Vec::new();
        self.decode_block(slice.data(), info, &mut buffer);
        Box::new(buffer.into_iter())
    }

    fn decode_into_bytes(
        &self,
        data: &[u8],
        _term_index: TermIndex,
        info: &TermBlockInformation,
        buffer: &mut Vec<DocId>,
    ) {
        self.decode_block(data, info, buffer);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::compress::TermBlockInformation;

    fn make_info(min: DocId, max: DocId, len: usize) -> TermBlockInformation {
        TermBlockInformation {
            docid_position_range: (0, 0),
            impact_position_range: (0, 0),
            length: len,
            max_value: 1.0,
            min_doc_id: min,
            max_doc_id: max,
        }
    }

    #[test]
    fn test_pfor_roundtrip() {
        let mut doc_ids: Vec<DocId> = (0..128).map(|i| i * 3).collect();
        doc_ids[50] += 100000; // outlier
        for i in 51..128 {
            doc_ids[i] = doc_ids[i].max(doc_ids[i - 1] + 1);
        }
        let info = make_info(doc_ids[0], *doc_ids.last().unwrap(), 128);

        let pfor = PForCompressor {};
        let mut buf = Vec::new();
        pfor.write(&mut buf, &doc_ids, 0, &info);

        let mut decoded = Vec::new();
        pfor.decode_block(&buf, &info, &mut decoded);
        assert_eq!(decoded, doc_ids);
    }

    #[test]
    fn test_pfor_better_than_bitpacking_with_outliers() {
        let mut doc_ids: Vec<DocId> = (0..128).map(|i| i * 2 + 1000).collect();
        doc_ids[10] += 1000000;
        doc_ids[50] += 2000000;
        for i in 11..128 {
            doc_ids[i] = doc_ids[i].max(doc_ids[i - 1] + 1);
        }
        let info = make_info(doc_ids[0], *doc_ids.last().unwrap(), 128);

        let bp = BitPackingCompressor {};
        let pfor = PForCompressor {};
        let mut bp_buf = Vec::new();
        bp.write(&mut bp_buf, &doc_ids, 0, &info);
        let mut pfor_buf = Vec::new();
        pfor.write(&mut pfor_buf, &doc_ids, 0, &info);

        eprintln!("BP: {} bytes, PFOR: {} bytes", bp_buf.len(), pfor_buf.len());
        assert!(pfor_buf.len() <= bp_buf.len());
    }

    #[test]
    fn test_pfor_tail_block() {
        let doc_ids: Vec<DocId> = vec![100, 200, 350, 400, 999];
        let info = make_info(100, 999, 5);

        let pfor = PForCompressor {};
        let mut buf = Vec::new();
        pfor.write(&mut buf, &doc_ids, 0, &info);

        let mut decoded = Vec::new();
        pfor.decode_block(&buf, &info, &mut decoded);
        assert_eq!(decoded, doc_ids);
    }
}
