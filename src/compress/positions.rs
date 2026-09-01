//! Position-delta compression for the compressed index's `positions.dat`.
//!
//! Encoding unit is one block's flattened delta stream (Lucene `.pos`
//! model): concatenate, over the block's postings in order, each posting's
//! position deltas (first position absolute, then gaps). Run boundaries are
//! implied by the block's tf values (recovered from the impact stream at
//! read time via [`super::TermBlockInformation::num_positions`]), not
//! stored by the codec itself.

use std::io::Write;

use bitpacking::{BitPacker, BitPacker4x};
use serde::{Deserialize, Serialize};

use crate::utils::vbyte::{read_vbyte_u32, write_vbyte_u32};

/// Codec for one block's flattened position-delta stream (per posting:
/// first position absolute, then gaps; runs concatenated in posting order —
/// run boundaries are implied by the block's tf values).
#[typetag::serde(tag = "type")]
pub trait PositionsCompressor: Send + Sync {
    /// Human-readable codec name, used in manifest / diagnostics summaries.
    ///
    /// Same `type_name` trick as [`super::Compressor::codec_name`]: stays
    /// object-safe because `Self` only appears inside the body.
    fn codec_name(&self) -> &'static str {
        let full = std::any::type_name::<Self>();
        full.rsplit("::").next().unwrap_or(full)
    }

    /// Writes the block's flattened delta stream.
    fn write(&self, writer: &mut dyn std::io::Write, deltas: &[u32]);

    /// Decodes `num_positions` deltas from `data` into `buffer` (cleared
    /// then filled). `num_positions` is the sum of the block's tfs,
    /// carried alongside the byte range in `TermBlockInformation` since the
    /// codec has no other way to know how many deltas a chunked format
    /// encodes.
    fn decode_into(&self, data: &[u8], num_positions: usize, buffer: &mut Vec<u32>);
}

/// Byte-aligned baseline: vbyte every delta. Simplest decode, and the
/// reference implementation other codecs are tested against.
#[derive(Serialize, Deserialize, Clone, Copy)]
pub struct VBytePositions;

#[typetag::serde]
impl PositionsCompressor for VBytePositions {
    fn write(&self, writer: &mut dyn Write, deltas: &[u32]) {
        for &d in deltas {
            write_vbyte_u32(writer, d).expect("write vbyte position delta");
        }
    }

    fn decode_into(&self, data: &[u8], num_positions: usize, buffer: &mut Vec<u32>) {
        buffer.clear();
        buffer.reserve(num_positions);
        let mut offset = 0usize;
        for _ in 0..num_positions {
            buffer.push(read_vbyte_u32(data, &mut offset));
        }
    }
}

const POSITIONS_CHUNK_LEN: usize = 128; // BitPacker4x::BLOCK_LEN

/// Marker byte for the final partial chunk (< 128 deltas), stored as plain
/// vbytes instead of a bitpacked chunk. Reuses the same convention as
/// [`super::docid::BitPackingCompressor`]'s tail marker (kept as a local
/// const rather than importing that module's private one).
const POSITIONS_TAIL_MARKER: u8 = 0xFF;

/// Default codec: FOR-bitpacked chunks of 128 deltas via [`BitPacker4x`]'s
/// plain (non-`_sorted`) `compress`/`decompress` -- position deltas are not
/// sorted, unlike doc-id gaps. Format: full chunks are `[num_bits: u8]`
/// followed by the packed bytes; the final partial chunk (< 128 deltas, if
/// any) is `[0xFF marker: u8]` followed by vbyte-encoded deltas.
#[derive(Serialize, Deserialize, Clone, Copy)]
pub struct BitPackingPositions;

#[typetag::serde]
impl PositionsCompressor for BitPackingPositions {
    fn write(&self, writer: &mut dyn Write, deltas: &[u32]) {
        let bitpacker = BitPacker4x::new();
        let mut chunks = deltas.chunks_exact(POSITIONS_CHUNK_LEN);
        for chunk in &mut chunks {
            let num_bits = bitpacker.num_bits(chunk);
            writer.write_all(&[num_bits]).expect("write num_bits");
            if num_bits > 0 {
                let mut compressed = vec![0u8; POSITIONS_CHUNK_LEN * 4];
                let written = bitpacker.compress(chunk, &mut compressed, num_bits);
                writer
                    .write_all(&compressed[..written])
                    .expect("write packed positions");
            }
        }

        let remainder = chunks.remainder();
        if !remainder.is_empty() {
            writer
                .write_all(&[POSITIONS_TAIL_MARKER])
                .expect("write tail marker");
            for &d in remainder {
                write_vbyte_u32(writer, d).expect("write vbyte position delta");
            }
        }
    }

    fn decode_into(&self, data: &[u8], num_positions: usize, buffer: &mut Vec<u32>) {
        buffer.clear();
        buffer.reserve(num_positions);

        let bitpacker = BitPacker4x::new();
        let mut offset = 0usize;
        let full_chunks = num_positions / POSITIONS_CHUNK_LEN;
        let tail_len = num_positions % POSITIONS_CHUNK_LEN;

        for _ in 0..full_chunks {
            let num_bits = data[offset];
            offset += 1;
            let mut chunk = [0u32; POSITIONS_CHUNK_LEN];
            if num_bits > 0 {
                offset += bitpacker.decompress(&data[offset..], &mut chunk, num_bits);
            }
            buffer.extend_from_slice(&chunk);
        }

        if tail_len > 0 {
            debug_assert_eq!(
                data[offset], POSITIONS_TAIL_MARKER,
                "expected tail marker at end of bitpacked positions chunk stream"
            );
            offset += 1;
            for _ in 0..tail_len {
                buffer.push(read_vbyte_u32(data, &mut offset));
            }
        }
    }
}
