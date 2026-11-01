//! File copy helpers that avoid `copy_file_range`.
//!
//! On Linux, `std::fs::copy` (and `std::io::copy` between two files) use
//! `copy_file_range`, which NFS v4.2 turns into a server-side COPY. Some
//! servers never complete that operation (the async-copy callback never
//! arrives), leaving the process blocked in `nfs42_proc_copy` forever. These
//! helpers copy through a user-space buffer instead.

use std::fs::File;
use std::io::{self, BufReader, BufWriter, Read, Write};
use std::path::Path;

/// Copies `src` to `dst` with plain reads and writes (no `copy_file_range`).
pub(crate) fn copy_file(src: &Path, dst: &Path) -> io::Result<u64> {
    let mut reader = BufReader::with_capacity(1 << 20, File::open(src)?);
    let mut writer = BufWriter::with_capacity(1 << 20, File::create(dst)?);
    let mut buf = vec![0u8; 1 << 20];
    let mut total = 0u64;
    loop {
        let n = match reader.read(&mut buf) {
            Ok(0) => break,
            Ok(n) => n,
            Err(e) if e.kind() == io::ErrorKind::Interrupted => continue,
            Err(e) => return Err(e),
        };
        writer.write_all(&buf[..n])?;
        total += n as u64;
    }
    writer.flush()?;
    Ok(total)
}

/// Hard-links `src` to `dst`, falling back to [`copy_file`] (e.g. across
/// filesystems).
pub(crate) fn link_or_copy(src: &Path, dst: &Path) -> io::Result<()> {
    if std::fs::hard_link(src, dst).is_err() {
        copy_file(src, dst)?;
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn copy_and_link() {
        let dir = std::env::temp_dir().join(format!("impact-fs-test-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let src = dir.join("a");
        let data: Vec<u8> = (0..3_000_000u32).map(|i| (i % 251) as u8).collect();
        std::fs::write(&src, &data).unwrap();

        let dst = dir.join("b");
        assert_eq!(copy_file(&src, &dst).unwrap(), data.len() as u64);
        assert_eq!(std::fs::read(&dst).unwrap(), data);

        let linked = dir.join("c");
        link_or_copy(&src, &linked).unwrap();
        assert_eq!(std::fs::read(&linked).unwrap(), data);
        std::fs::remove_dir_all(&dir).unwrap();
    }
}
