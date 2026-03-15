//! Compression schemes for impact values.
//!
//! - [`Quantizer`]: Fixed-range uniform quantization to N bits
//! - [`GlobalQuantizerFactory`]: Auto-ranging quantizer using global min/max
//! - [`Identity`]: No compression (stores raw f32 values)

use core::f32;
use std::io::Write;

use bitpacking::{BitPacker, BitPacker4x};
use bitstream_io::{BigEndian, BitRead, BitReader, BitWrite, BitWriter};
use byteorder::{ReadBytesExt, WriteBytesExt};

use super::{Compressor, ImpactCompressor, ImpactCompressorFactory, TermBlockInformation};
use crate::{
    base::{ImpactValue, TermIndex},
    index::SparseIndexView,
    utils::buffer::{Slice, SliceReader},
};
use serde::{Deserialize, Serialize};

const BLOCK_LEN: usize = 128;

// ---
// --- Quantizer
// ---

/// Uniform quantizer that maps floating-point impact values to N-bit integers.
///
/// Values are linearly mapped from `[min, max]` into `2^nbits` levels.
#[derive(Serialize, Deserialize, Clone, Copy)]
pub struct Quantizer {
    /// Number of bits per quantized value.
    pub nbits: u32,
    /// Number of quantization levels (`2^nbits`).
    pub levels: u32,
    /// Quantization step size.
    pub step: ImpactValue,
    /// Minimum value of the quantization range.
    pub min: ImpactValue,
    /// Maximum value of the quantization range.
    pub max: ImpactValue,
}

impl Quantizer {
    /// Creates a quantizer with the given bit width and value range.
    pub fn new(nbits: u32, min: ImpactValue, max: ImpactValue) -> Self {
        let levels = 2 << (nbits - 1);
        Self {
            nbits: nbits,
            levels: levels,
            min: min,
            max: max,
            step: (max - min) / ((levels + 1) as f32),
        }
    }
}

/// Factory that creates a [`Quantizer`] with min/max determined from the index.
#[derive(Clone)]
pub struct GlobalQuantizerFactory {
    /// Number of bits for quantization.
    pub nbits: u32,
}

impl ImpactCompressorFactory for GlobalQuantizerFactory {
    fn create(&self, index: &dyn crate::index::SparseIndexView) -> Box<dyn ImpactCompressor> {
        log::info!(
            "Computing global minimum and maximum impact (quantizer) over {} terms",
            index.len()
        );
        let mut min = ImpactValue::INFINITY;
        let mut max = -ImpactValue::INFINITY;

        // Compute the maximum over all terms
        for term_ix in 0..index.len() {
            let (term_min, term_max) = index.value_range(term_ix);
            min = min.min(term_min);
            max = max.max(term_max);
        }
        log::info!("Quantizer bounds: {}-{}", min, max);
        Box::new(Quantizer::new(self.nbits, min, max))
    }

    fn clone(&self) -> Box<dyn ImpactCompressorFactory> {
        Box::new(Clone::clone(self))
    }
}

#[typetag::serde]
impl ImpactCompressor for Quantizer {}

impl ImpactCompressorFactory for Quantizer {
    fn create(&self, _index: &dyn crate::index::SparseIndexView) -> Box<dyn ImpactCompressor> {
        Box::new(Clone::clone(self))
    }

    fn clone(&self) -> Box<dyn ImpactCompressorFactory> {
        Box::new(Clone::clone(self))
    }
}

impl<'a> Compressor<ImpactValue> for Quantizer {
    fn write(
        &self,
        writer: &mut dyn Write,
        values: &[ImpactValue],
        _term_index: TermIndex,
        _info: &TermBlockInformation,
    ) {
        let mut bit_writer = BitWriter::endian(writer, BigEndian);

        for x in values {
            let value = ((*x - self.min) / self.step).trunc() as u32;

            let quantized = value.max(0).min(self.levels - 1);
            bit_writer
                .write(self.nbits, quantized)
                .expect("Cannot write bits");
        }

        bit_writer
            .byte_align()
            .expect("Could not write padding bits");
    }

    fn read<'b>(
        &self,
        slice: Box<dyn Slice + 'b>,
        _term_index: TermIndex,
        info: &TermBlockInformation,
    ) -> Box<dyn Iterator<Item = ImpactValue> + Send + 'b> {
        // Bulk-decode all quantized values at once to avoid per-posting BitReader overhead
        let slice_reader = Box::new(SliceReader::new(slice));
        let mut bit_reader = BitReader::endian(slice_reader, BigEndian);
        let min = self.min;
        let step = self.step;

        let values: Vec<ImpactValue> = (0..info.length)
            .map(|_| {
                let quantized = bit_reader.read::<u32>(self.nbits).unwrap();
                (quantized as ImpactValue) * step + min + step / 2.
            })
            .collect();

        Box::new(values.into_iter())
    }

    fn decode_into_bytes(
        &self,
        data: &[u8],
        _term_index: TermIndex,
        info: &TermBlockInformation,
        buffer: &mut Vec<ImpactValue>,
    ) {
        let min = self.min;
        let step = self.step;
        let half_step = step / 2.0;

        buffer.clear();
        buffer.reserve(info.length);

        // Fast path for byte-aligned bit widths: read directly from bytes
        match self.nbits {
            16 => {
                for i in 0..info.length {
                    let offset = i * 2;
                    let quantized = u16::from_be_bytes([data[offset], data[offset + 1]]) as u32;
                    buffer.push((quantized as ImpactValue) * step + min + half_step);
                }
            }
            8 => {
                for i in 0..info.length {
                    let quantized = data[i] as u32;
                    buffer.push((quantized as ImpactValue) * step + min + half_step);
                }
            }
            _ => {
                // Generic path via BitReader for arbitrary bit widths
                let mut bit_reader = BitReader::endian(data, BigEndian);
                for _ in 0..info.length {
                    let quantized = bit_reader.read::<u32>(self.nbits).unwrap();
                    buffer.push((quantized as ImpactValue) * step + min + half_step);
                }
            }
        }
    }
}

// ---
// --- Identity transform
// ---

/// Identity compressor that stores impact values as raw f32 (no compression).
#[derive(Serialize, Deserialize, Clone, Copy)]
pub struct Identity {}

#[typetag::serde]
impl ImpactCompressor for Identity {}

impl ImpactCompressorFactory for Identity {
    fn create(&self, _index: &dyn crate::index::SparseIndexView) -> Box<dyn ImpactCompressor> {
        Box::new(Clone::clone(self))
    }

    fn clone(&self) -> Box<dyn ImpactCompressorFactory> {
        Box::new(Clone::clone(self))
    }
}

impl<'a> Compressor<ImpactValue> for Identity {
    fn write(
        &self,
        writer: &mut dyn Write,
        values: &[ImpactValue],
        _term_index: TermIndex,
        _info: &TermBlockInformation,
    ) {
        for x in values {
            writer
                .write_f32::<byteorder::BigEndian>(*x)
                .expect("cannot write");
        }
    }

    fn read<'b>(
        &self,
        slice: Box<dyn Slice + 'b>,
        _term_index: TermIndex,
        info: &TermBlockInformation,
    ) -> Box<dyn Iterator<Item = ImpactValue> + Send + 'b> {
        Box::new(IdentityIterator::<'b> {
            index: 0,
            count: info.length,
            slice,
        })
    }
}

struct IdentityIterator<'a> {
    index: usize,
    count: usize,
    slice: Box<dyn Slice + 'a>,
}

impl<'a> Iterator for IdentityIterator<'a> {
    type Item = ImpactValue;

    fn next(&mut self) -> Option<Self::Item> {
        if self.index < self.count {
            let data = self.slice.as_ref().data();
            let mut view = &data[self.index * 4..self.index * 4 + 4];
            self.index += 1;
            Some(view.read_f32::<byteorder::BigEndian>().expect("read error"))
        } else {
            None
        }
    }
}

// ---
// --- BitPacked integer compressor (for raw TF counts)
// ---

/// Compresses impact values as raw integers with SIMD bitpacking.
///
/// For BM25 indices where values are integer term frequencies (typically 1-5),
/// this uses ~2-3 bits per value vs 8 bits for quantized floats.
///
/// Format per full block (128 values):
/// - `[num_bits: u8]`
/// - `[bitpacked 128 u32 values: 16 * num_bits bytes]`
///
/// Tail blocks (< 128): `[0xFF marker] [u32 LE values]`
///
/// Values are stored as `round(value)` — lossless for integer TF counts.
#[derive(Serialize, Deserialize, Clone, Copy)]
pub struct BitPackedIntCompressor {}

#[typetag::serde]
impl ImpactCompressor for BitPackedIntCompressor {}

impl ImpactCompressorFactory for BitPackedIntCompressor {
    fn create(&self, _index: &dyn SparseIndexView) -> Box<dyn ImpactCompressor> {
        Box::new(BitPackedIntCompressor {})
    }
    fn clone(&self) -> Box<dyn ImpactCompressorFactory> {
        Box::new(Clone::clone(self))
    }
}

impl Compressor<ImpactValue> for BitPackedIntCompressor {
    fn write(
        &self,
        writer: &mut dyn Write,
        values: &[ImpactValue],
        _term_index: TermIndex,
        _info: &TermBlockInformation,
    ) {
        let ints: Vec<u32> = values.iter().map(|&v| v.round() as u32).collect();

        if ints.len() == BLOCK_LEN {
            let bitpacker = BitPacker4x::new();
            let num_bits = bitpacker.num_bits(&ints);
            writer.write_all(&[num_bits]).expect("write num_bits");
            if num_bits > 0 {
                let mut compressed = vec![0u8; BLOCK_LEN * 4];
                let written = bitpacker.compress(&ints, &mut compressed, num_bits);
                writer
                    .write_all(&compressed[..written])
                    .expect("write packed");
            }
        } else {
            writer.write_all(&[0xFF]).expect("write marker");
            for &v in &ints {
                writer.write_all(&v.to_le_bytes()).expect("write val");
            }
        }
    }

    fn read<'a>(
        &self,
        slice: Box<dyn Slice + 'a>,
        _term_index: TermIndex,
        info: &TermBlockInformation,
    ) -> Box<dyn Iterator<Item = ImpactValue> + Send + 'a> {
        let mut buffer = Vec::new();
        self.decode_block(slice.data(), info, &mut buffer);
        Box::new(buffer.into_iter())
    }

    fn decode_into_bytes(
        &self,
        data: &[u8],
        _term_index: TermIndex,
        info: &TermBlockInformation,
        buffer: &mut Vec<ImpactValue>,
    ) {
        self.decode_block(data, info, buffer);
    }
}

impl BitPackedIntCompressor {
    fn decode_block(
        &self,
        data: &[u8],
        info: &TermBlockInformation,
        buffer: &mut Vec<ImpactValue>,
    ) {
        buffer.clear();
        let marker = data[0];

        if marker != 0xFF && info.length == BLOCK_LEN {
            let num_bits = marker;
            let mut decompressed = [0u32; BLOCK_LEN];
            if num_bits > 0 {
                let bitpacker = BitPacker4x::new();
                bitpacker.decompress(&data[1..], &mut decompressed, num_bits);
            }
            buffer.reserve(BLOCK_LEN);
            for &v in &decompressed {
                buffer.push(v as ImpactValue);
            }
        } else {
            buffer.reserve(info.length);
            for i in 0..info.length {
                let offset = 1 + i * 4;
                let v = u32::from_le_bytes([
                    data[offset],
                    data[offset + 1],
                    data[offset + 2],
                    data[offset + 3],
                ]);
                buffer.push(v as ImpactValue);
            }
        }
    }
}

// ---
// --- Quantized + BitPacked compressor (for neural IR like SPLADE)
// ---

/// Quantizes float impacts to N-bit integers, then compresses with adaptive
/// SIMD bitpacking. Combines quantization with adaptive compression.
///
/// For SPLADE, most quantized values are small (0-3) with occasional spikes.
/// Adaptive bitpacking uses ~3-4 bits/value instead of fixed N bits,
/// reducing size by 2-3x compared to fixed-width quantization.
///
/// Use this for neural IR models with continuous float impact values.
/// For BM25 with integer TF counts, use [`BitPackedIntCompressor`] instead.
#[derive(Serialize, Deserialize, Clone, Copy)]
pub struct QuantizedBitPackedCompressor {
    pub nbits: u32,
    pub step: ImpactValue,
    pub min: ImpactValue,
    pub max: ImpactValue,
}

#[typetag::serde]
impl ImpactCompressor for QuantizedBitPackedCompressor {}

/// Factory that creates a [`QuantizedBitPackedCompressor`] with min/max from the index.
#[derive(Clone)]
pub struct QuantizedBitPackedFactory {
    pub nbits: u32,
}

impl ImpactCompressorFactory for QuantizedBitPackedFactory {
    fn create(&self, index: &dyn SparseIndexView) -> Box<dyn ImpactCompressor> {
        let mut min = ImpactValue::INFINITY;
        let mut max = -ImpactValue::INFINITY;
        for term_ix in 0..index.len() {
            let (term_min, term_max) = index.value_range(term_ix);
            min = min.min(term_min);
            max = max.max(term_max);
        }
        let levels = 1u32 << self.nbits;
        let step = (max - min) / (levels as f32 + 1.0);
        Box::new(QuantizedBitPackedCompressor {
            nbits: self.nbits,
            step,
            min,
            max,
        })
    }

    fn clone(&self) -> Box<dyn ImpactCompressorFactory> {
        Box::new(Clone::clone(self))
    }
}

impl Compressor<ImpactValue> for QuantizedBitPackedCompressor {
    fn write(
        &self,
        writer: &mut dyn Write,
        values: &[ImpactValue],
        _term_index: TermIndex,
        _info: &TermBlockInformation,
    ) {
        let levels = (1u32 << self.nbits) - 1;
        let ints: Vec<u32> = values
            .iter()
            .map(|&v| {
                let q = ((v - self.min) / self.step).trunc() as u32;
                q.min(levels)
            })
            .collect();

        if ints.len() == BLOCK_LEN {
            let bitpacker = BitPacker4x::new();
            let num_bits = bitpacker.num_bits(&ints);
            writer.write_all(&[num_bits]).expect("write num_bits");
            if num_bits > 0 {
                let mut compressed = vec![0u8; BLOCK_LEN * 4];
                let written = bitpacker.compress(&ints, &mut compressed, num_bits);
                writer
                    .write_all(&compressed[..written])
                    .expect("write packed");
            }
        } else {
            writer.write_all(&[0xFF]).expect("write marker");
            for &v in &ints {
                writer.write_all(&v.to_le_bytes()).expect("write val");
            }
        }
    }

    fn read<'a>(
        &self,
        slice: Box<dyn Slice + 'a>,
        _term_index: TermIndex,
        info: &TermBlockInformation,
    ) -> Box<dyn Iterator<Item = ImpactValue> + Send + 'a> {
        let mut buffer = Vec::new();
        self.decode_block(slice.data(), info, &mut buffer);
        Box::new(buffer.into_iter())
    }

    fn decode_into_bytes(
        &self,
        data: &[u8],
        _term_index: TermIndex,
        info: &TermBlockInformation,
        buffer: &mut Vec<ImpactValue>,
    ) {
        self.decode_block(data, info, buffer);
    }
}

impl QuantizedBitPackedCompressor {
    fn decode_block(
        &self,
        data: &[u8],
        info: &TermBlockInformation,
        buffer: &mut Vec<ImpactValue>,
    ) {
        buffer.clear();
        let step = self.step;
        let min = self.min;
        let half_step = step / 2.0;
        let marker = data[0];

        if marker != 0xFF && info.length == BLOCK_LEN {
            let num_bits = marker;
            let mut decompressed = [0u32; BLOCK_LEN];
            if num_bits > 0 {
                let bitpacker = BitPacker4x::new();
                bitpacker.decompress(&data[1..], &mut decompressed, num_bits);
            }
            buffer.reserve(BLOCK_LEN);
            for &v in &decompressed {
                buffer.push((v as ImpactValue) * step + min + half_step);
            }
        } else {
            buffer.reserve(info.length);
            for i in 0..info.length {
                let offset = 1 + i * 4;
                let v = u32::from_le_bytes([
                    data[offset],
                    data[offset + 1],
                    data[offset + 2],
                    data[offset + 3],
                ]);
                buffer.push((v as ImpactValue) * step + min + half_step);
            }
        }
    }
}
