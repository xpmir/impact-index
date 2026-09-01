//! Variable-length encoding for `u32` values (position deltas).
//!
//! Same convention as `write_vint`/`read_vint` in `src/compress/mod.rs`
//! (low 7 bits first, high bit of each byte = "more bytes follow"),
//! specialized to `u32` since positions never exceed a document's token
//! count. Used by the forward-index builder now, and by compressed
//! position codecs later.

use std::io::{self, Write};

/// Writes `v` as a vbyte (1-5 bytes): low 7 bits first, 0x80 = continue.
/// Returns the number of bytes written.
pub(crate) fn write_vbyte_u32(writer: &mut dyn Write, mut v: u32) -> io::Result<usize> {
    let mut n = 0;
    while v >= 0x80 {
        writer.write_all(&[(v as u8) | 0x80])?;
        v >>= 7;
        n += 1;
    }
    writer.write_all(&[v as u8])?;
    Ok(n + 1)
}

/// Reads a vbyte-encoded `u32` from `data` starting at `*offset`, advancing
/// `offset` past the bytes consumed.
pub(crate) fn read_vbyte_u32(data: &[u8], offset: &mut usize) -> u32 {
    let mut result: u32 = 0;
    let mut shift = 0;
    loop {
        let byte = data[*offset];
        *offset += 1;
        result |= ((byte & 0x7F) as u32) << shift;
        if byte & 0x80 == 0 {
            return result;
        }
        shift += 7;
    }
}
