//! Portable Elias-Fano codec, byte-compatible with `sucds` 0.5.
//!
//! `sucds` only compiles on 64-bit targets. This module reads and writes the
//! same serialized `EliasFano` structure with explicit `u64` arithmetic, so
//! that 32-bit targets (WASM) can use [`super::docid::EliasFanoCompressor`]
//! and exchange index files with native builds. Native builds keep using
//! `sucds`; this module is compiled everywhere so that tests can check it
//! against `sucds` byte for byte.
//!
//! Serialized layout (all integers little-endian, `usize` as 8 bytes):
//!
//! ```text
//! high_bits     BitVector    (Vec<u64> words, u64 len)
//! high_bits_d1  DArrayIndex  (Vec<i64>, Vec<u16>, Vec<u64>, u64, bool)
//! high_bits_d0  Option<DArrayIndex>  (bool, then the index if true)
//! low_bits      BitVector
//! low_len       u64
//! universe      u64
//! ```

use std::io::{self, Read, Write};

/// `DArrayIndex` parameters (see `sucds::darray`)
const BLOCK_LEN: usize = 1024;
const SUBBLOCK_LEN: usize = 32;
const MAX_IN_BLOCK_DISTANCE: u64 = 1 << 16;

fn write_u64(w: &mut dyn Write, x: u64) -> io::Result<()> {
    w.write_all(&x.to_le_bytes())
}

fn read_u64(r: &mut impl Read) -> io::Result<u64> {
    let mut buf = [0u8; 8];
    r.read_exact(&mut buf)?;
    Ok(u64::from_le_bytes(buf))
}

fn read_len(r: &mut impl Read) -> io::Result<usize> {
    usize::try_from(read_u64(r)?)
        .map_err(|_| io::Error::new(io::ErrorKind::InvalidData, "length overflows usize"))
}

fn read_bool(r: &mut impl Read) -> io::Result<bool> {
    let mut buf = [0u8; 1];
    r.read_exact(&mut buf)?;
    Ok(buf[0] != 0)
}

/// Skips `n` bytes of a reader
fn skip(r: &mut impl Read, n: u64) -> io::Result<()> {
    let skipped = io::copy(&mut r.take(n), &mut io::sink())?;
    if skipped != n {
        return Err(io::ErrorKind::UnexpectedEof.into());
    }
    Ok(())
}

/// A bit vector stored as little-endian 64-bit words (LSB first)
struct Bits {
    words: Vec<u64>,
    len: u64,
}

impl Bits {
    fn with_len(len: u64) -> Self {
        Self {
            words: vec![0; len.div_ceil(64) as usize],
            len,
        }
    }

    fn set(&mut self, pos: u64) {
        self.words[(pos / 64) as usize] |= 1 << (pos % 64);
    }

    /// Appends the `len` (< 64) low bits of `bits`
    fn push_bits(&mut self, bits: u64, len: u64) {
        if len == 0 {
            return;
        }
        let pos_in_word = self.len % 64;
        if pos_in_word == 0 {
            self.words.push(bits);
        } else {
            *self.words.last_mut().unwrap() |= bits << pos_in_word;
            if len > 64 - pos_in_word {
                self.words.push(bits >> (64 - pos_in_word));
            }
        }
        self.len += len;
    }

    /// Reads `len` (< 64) bits at `pos`
    fn get_bits(&self, pos: u64, len: u64) -> u64 {
        if len == 0 {
            return 0;
        }
        let (block, shift) = ((pos / 64) as usize, pos % 64);
        let mask = (1u64 << len) - 1;
        if shift + len <= 64 {
            (self.words[block] >> shift) & mask
        } else {
            ((self.words[block] >> shift) | (self.words[block + 1] << (64 - shift))) & mask
        }
    }

    fn write(&self, w: &mut dyn Write) -> io::Result<()> {
        write_u64(w, self.words.len() as u64)?;
        for &word in &self.words {
            write_u64(w, word)?;
        }
        write_u64(w, self.len)
    }

    fn read(r: &mut impl Read) -> io::Result<Self> {
        let n = read_len(r)?;
        let mut words = Vec::with_capacity(n);
        for _ in 0..n {
            words.push(read_u64(r)?);
        }
        let len = read_u64(r)?;
        if len > 64 * n as u64 {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                "bit vector shorter than its length",
            ));
        }
        Ok(Self { words, len })
    }
}

/// Writes the `DArrayIndex` (select over ones) of `bits`, as built by
/// `sucds::darray::DArrayIndex::new(bits, true)`
fn write_darray_index(w: &mut dyn Write, bits: &Bits) -> io::Result<()> {
    let mut block_inventory: Vec<i64> = Vec::new();
    let mut subblock_inventory: Vec<u16> = Vec::new();
    let mut overflow_positions: Vec<u64> = Vec::new();
    let mut num_positions = 0u64;

    let mut flush = |block: &mut Vec<u64>| {
        let (first, last) = (block[0], block[block.len() - 1]);
        if last - first < MAX_IN_BLOCK_DISTANCE {
            block_inventory.push(first as i64);
            for i in (0..block.len()).step_by(SUBBLOCK_LEN) {
                subblock_inventory.push((block[i] - first) as u16);
            }
        } else {
            block_inventory.push(-((overflow_positions.len() + 1) as i64));
            overflow_positions.extend_from_slice(block);
            for _ in (0..block.len()).step_by(SUBBLOCK_LEN) {
                subblock_inventory.push(u16::MAX);
            }
        }
        block.clear();
    };

    let mut block = Vec::with_capacity(BLOCK_LEN);
    for (word_ix, &word) in bits.words.iter().enumerate() {
        let mut word = word;
        while word != 0 {
            let pos = word_ix as u64 * 64 + word.trailing_zeros() as u64;
            if pos >= bits.len {
                break;
            }
            word &= word - 1;
            block.push(pos);
            num_positions += 1;
            if block.len() == BLOCK_LEN {
                flush(&mut block);
            }
        }
    }
    if !block.is_empty() {
        flush(&mut block);
    }

    write_u64(w, block_inventory.len() as u64)?;
    for &x in &block_inventory {
        w.write_all(&x.to_le_bytes())?;
    }
    write_u64(w, subblock_inventory.len() as u64)?;
    for &x in &subblock_inventory {
        w.write_all(&x.to_le_bytes())?;
    }
    write_u64(w, overflow_positions.len() as u64)?;
    for &x in &overflow_positions {
        write_u64(w, x)?;
    }
    write_u64(w, num_positions)?;
    w.write_all(&[1u8]) // over_one
}

/// Skips a serialized `DArrayIndex`, returning its number of positions
fn skip_darray_index(r: &mut impl Read) -> io::Result<u64> {
    let n = read_u64(r)?;
    skip(r, n * 8)?;
    let n = read_u64(r)?;
    skip(r, n * 2)?;
    let n = read_u64(r)?;
    skip(r, n * 8)?;
    let num_positions = read_u64(r)?;
    read_bool(r)?;
    Ok(num_positions)
}

/// Serializes the non-decreasing `values` (all `< universe`) exactly as
/// `sucds::EliasFanoBuilder::new(universe, values.len())` + `push` +
/// `build` + `serialize_into` would.
pub fn write(w: &mut dyn Write, values: &[u64], universe: u64) -> io::Result<()> {
    let n = values.len() as u64;
    if n == 0 {
        return Err(io::Error::new(
            io::ErrorKind::InvalidInput,
            "Elias-Fano needs at least one value",
        ));
    }
    let low_len = match universe / n {
        0 => 0,
        x => 63 - x.leading_zeros() as u64,
    };
    let low_mask = (1u64 << low_len) - 1;

    let mut high_bits = Bits::with_len((n + 1) + (universe >> low_len) + 1);
    let mut low_bits = Bits {
        words: Vec::new(),
        len: 0,
    };
    let mut last = 0;
    for (i, &v) in values.iter().enumerate() {
        if v < last || v >= universe {
            return Err(io::Error::new(
                io::ErrorKind::InvalidInput,
                "Elias-Fano values must be non-decreasing and below the universe",
            ));
        }
        last = v;
        low_bits.push_bits(v & low_mask, low_len);
        high_bits.set((v >> low_len) + i as u64);
    }

    high_bits.write(w)?;
    write_darray_index(w, &high_bits)?;
    w.write_all(&[0u8])?; // no high_bits_d0
    low_bits.write(w)?;
    write_u64(w, low_len)?;
    write_u64(w, universe)
}

/// A deserialized Elias-Fano sequence, iterated from the start
pub struct EliasFano {
    high_bits: Bits,
    low_bits: Bits,
    low_len: u64,
    len: u64,
}

impl EliasFano {
    pub fn read(data: &[u8]) -> io::Result<Self> {
        let mut r = data;
        let high_bits = Bits::read(&mut r)?;
        let len = skip_darray_index(&mut r)?;
        if read_bool(&mut r)? {
            skip_darray_index(&mut r)?;
        }
        let low_bits = Bits::read(&mut r)?;
        let low_len = read_u64(&mut r)?;
        let _universe = read_u64(&mut r)?;
        if low_len >= 64 || low_bits.len < len * low_len {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                "inconsistent Elias-Fano low bits",
            ));
        }
        Ok(Self {
            high_bits,
            low_bits,
            low_len,
            len,
        })
    }

    pub fn len(&self) -> u64 {
        self.len
    }

    pub fn iter(&self) -> Iter<'_> {
        Iter {
            ef: self,
            k: 0,
            word_ix: 0,
            word: self.high_bits.words.first().copied().unwrap_or(0),
        }
    }
}

/// Iterates over the values of an [`EliasFano`] sequence
pub struct Iter<'a> {
    ef: &'a EliasFano,
    k: u64,
    /// Current high-bits word, with the ones already consumed cleared
    word_ix: usize,
    word: u64,
}

impl Iterator for Iter<'_> {
    type Item = u64;

    #[inline]
    fn next(&mut self) -> Option<u64> {
        if self.k == self.ef.len {
            return None;
        }
        while self.word == 0 {
            self.word_ix += 1;
            // A truncated vector has fewer ones than `len`: stop early
            self.word = *self.ef.high_bits.words.get(self.word_ix)?;
        }
        let high = self.word_ix as u64 * 64 + self.word.trailing_zeros() as u64;
        self.word &= self.word - 1;
        let low = self
            .ef
            .low_bits
            .get_bits(self.k * self.ef.low_len, self.ef.low_len);
        let value = ((high - self.k) << self.ef.low_len) | low;
        self.k += 1;
        Some(value)
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        let remaining = (self.ef.len - self.k) as usize;
        (0, Some(remaining))
    }
}
