//! File copy helpers, with a global [`CopyMode`].
//!
//! On Linux, `std::fs::copy` (and `std::io::copy` between two files) use
//! `copy_file_range`, which NFS v4.2 turns into a server-side COPY. Some
//! servers never complete that operation (the async-copy callback never
//! arrives), leaving the process blocked in `nfs42_proc_copy` forever.
//!
//! Every file copy in the crate goes through [`copy_file`] or
//! [`link_or_copy`] (`std::fs::copy` and `std::io::copy` are forbidden by
//! `clippy.toml`), which follow the process-wide [`CopyMode`]:
//! - [`CopyMode::Auto`] (default): a user-space copy when the source or the
//!   destination is on a network filesystem (Linux only), the native copy
//!   otherwise;
//! - [`CopyMode::Buffered`]: always a user-space copy;
//! - [`CopyMode::Native`]: always `std::fs::copy` (`copy_file_range`,
//!   reflinks, `clonefile` on macOS).
//!
//! The mode is set with [`set_copy_mode`] or, if that is never called, from
//! the `IMPACT_INDEX_COPY_MODE` environment variable (`auto`, `buffered` or
//! `native`), read on first use.

use std::fs::File;
use std::io::{self, BufReader, BufWriter, Read, Write};
use std::path::Path;
use std::str::FromStr;
use std::sync::atomic::{AtomicU8, Ordering};

/// Environment variable giving the initial [`CopyMode`].
pub const COPY_MODE_ENV: &str = "IMPACT_INDEX_COPY_MODE";

/// How files are copied (see the [module documentation](self)).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum CopyMode {
    /// User-space copy on network filesystems, native copy elsewhere
    #[default]
    Auto,
    /// Always copy through a user-space buffer (never `copy_file_range`)
    Buffered,
    /// Always use `std::fs::copy`
    Native,
}

impl CopyMode {
    fn to_u8(self) -> u8 {
        match self {
            CopyMode::Auto => 1,
            CopyMode::Buffered => 2,
            CopyMode::Native => 3,
        }
    }

    fn from_u8(v: u8) -> Option<Self> {
        match v {
            1 => Some(CopyMode::Auto),
            2 => Some(CopyMode::Buffered),
            3 => Some(CopyMode::Native),
            _ => None,
        }
    }

    /// Name of the mode, as accepted by [`FromStr`]
    pub fn name(self) -> &'static str {
        match self {
            CopyMode::Auto => "auto",
            CopyMode::Buffered => "buffered",
            CopyMode::Native => "native",
        }
    }
}

impl std::fmt::Display for CopyMode {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(self.name())
    }
}

impl FromStr for CopyMode {
    type Err = String;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s.trim().to_ascii_lowercase().as_str() {
            "auto" => Ok(CopyMode::Auto),
            "buffered" => Ok(CopyMode::Buffered),
            "native" => Ok(CopyMode::Native),
            other => Err(format!(
                "unknown copy mode '{other}' (expected auto, buffered or native)"
            )),
        }
    }
}

/// 0 = not yet initialised (read the environment on first use)
static COPY_MODE: AtomicU8 = AtomicU8::new(0);

/// Sets the process-wide copy mode (overrides `IMPACT_INDEX_COPY_MODE`).
pub fn set_copy_mode(mode: CopyMode) {
    COPY_MODE.store(mode.to_u8(), Ordering::Relaxed);
}

/// Returns the process-wide copy mode.
pub fn copy_mode() -> CopyMode {
    if let Some(mode) = CopyMode::from_u8(COPY_MODE.load(Ordering::Relaxed)) {
        return mode;
    }
    let mode = mode_from_env(std::env::var(COPY_MODE_ENV).ok().as_deref());
    // Keep a mode set concurrently by `set_copy_mode`
    match COPY_MODE.compare_exchange(0, mode.to_u8(), Ordering::Relaxed, Ordering::Relaxed) {
        Ok(_) => mode,
        Err(current) => CopyMode::from_u8(current).unwrap_or(mode),
    }
}

fn mode_from_env(value: Option<&str>) -> CopyMode {
    match value {
        None => CopyMode::default(),
        Some(v) if v.trim().is_empty() => CopyMode::default(),
        Some(v) => v.parse().unwrap_or_else(|e| {
            log::warn!("{COPY_MODE_ENV}: {e}; using auto");
            CopyMode::default()
        }),
    }
}

/// Whether `path` (or, if it does not exist, its closest existing ancestor)
/// is on a network filesystem.
#[cfg(target_os = "linux")]
pub(crate) fn is_network_fs(path: &Path) -> bool {
    use std::os::unix::ffi::OsStrExt;

    let Some(existing) = path.ancestors().find(|p| p.exists()) else {
        return false;
    };
    let existing = if existing.as_os_str().is_empty() {
        Path::new(".")
    } else {
        existing
    };
    let Ok(c_path) = std::ffi::CString::new(existing.as_os_str().as_bytes()) else {
        return false;
    };
    let mut stat = std::mem::MaybeUninit::<libc::statfs>::uninit();
    // SAFETY: `c_path` is a valid NUL-terminated string and `stat` points
    // to writable memory of the right size.
    if unsafe { libc::statfs(c_path.as_ptr(), stat.as_mut_ptr()) } != 0 {
        // Unknown: be safe
        return true;
    }
    // SAFETY: statfs succeeded, so `stat` is initialised
    is_network_magic(unsafe { stat.assume_init() }.f_type as i64)
}

/// Whether a `statfs` filesystem type (`f_type`) is a network filesystem.
#[cfg_attr(not(target_os = "linux"), allow(dead_code))]
fn is_network_magic(f_type: i64) -> bool {
    // From linux/magic.h (and the Lustre, GPFS sources)
    const NETWORK_FS: &[u32] = &[
        0x6969,     // NFS
        0x517B,     // SMB
        0xFF534D42, // CIFS
        0xFE534D42, // SMB2
        0x0BD00BD0, // Lustre
        0x00C36400, // Ceph
        0x47504653, // GPFS
        0x5346414F, // AFS
        0x01021997, // 9P
    ];
    // f_type is a signed 32-bit integer on some architectures (CIFS is then
    // negative): compare the low 32 bits
    NETWORK_FS.contains(&(f_type as u32))
}

/// Network filesystems are only detected on Linux (the only platform where
/// `std::fs::copy` asks the server to copy).
#[cfg(not(target_os = "linux"))]
pub(crate) fn is_network_fs(_path: &Path) -> bool {
    false
}

/// Whether copying `src` to `dst` should go through a user-space buffer.
fn use_buffered(src: &Path, dst: &Path) -> bool {
    match copy_mode() {
        CopyMode::Buffered => true,
        CopyMode::Native => false,
        CopyMode::Auto => is_network_fs(src) || is_network_fs(dst),
    }
}

/// Copies `src` to `dst` with plain reads and writes (no `copy_file_range`).
fn copy_buffered(src: &Path, dst: &Path) -> io::Result<u64> {
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

/// Copies `src` to `dst` following the [`CopyMode`]; returns the number of
/// bytes copied.
pub(crate) fn copy_file(src: &Path, dst: &Path) -> io::Result<u64> {
    if use_buffered(src, dst) {
        copy_buffered(src, dst)
    } else {
        #[allow(clippy::disallowed_methods)]
        std::fs::copy(src, dst)
    }
}

/// Hard-links `src` to `dst`, falling back to [`copy_file`] (e.g. across
/// filesystems).
///
/// An existing `dst` is replaced (it is removed first: copying over a hard
/// link of `src` would truncate `src` itself); nothing is done when `dst` is
/// `src`.
pub(crate) fn link_or_copy(src: &Path, dst: &Path) -> io::Result<()> {
    if dst.exists() {
        if src.canonicalize()? == dst.canonicalize()? {
            return Ok(());
        }
        std::fs::remove_file(dst)?;
    }
    if std::fs::hard_link(src, dst).is_err() {
        copy_file(src, dst)?;
    }
    Ok(())
}

/// Recursively copies the directory `src` into `dst` (created if needed),
/// with [`copy_file`].
pub(crate) fn copy_dir_all(src: &Path, dst: &Path) -> io::Result<()> {
    std::fs::create_dir_all(dst)?;
    for entry in std::fs::read_dir(src)? {
        let entry = entry?;
        let dst_path = dst.join(entry.file_name());
        if entry.file_type()?.is_dir() {
            copy_dir_all(&entry.path(), &dst_path)?;
        } else {
            copy_file(&entry.path(), &dst_path)?;
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Mutex;

    /// Tests changing the global mode must not interleave
    static MODE_LOCK: Mutex<()> = Mutex::new(());

    fn test_dir(name: &str) -> std::path::PathBuf {
        let dir =
            std::env::temp_dir().join(format!("impact-fs-test-{}-{}", name, std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        dir
    }

    #[test]
    fn copy_and_link() {
        let _guard = MODE_LOCK.lock().unwrap();
        let dir = test_dir("copy");
        let src = dir.join("a");
        let data: Vec<u8> = (0..3_000_000u32).map(|i| (i % 251) as u8).collect();
        std::fs::write(&src, &data).unwrap();

        for mode in [CopyMode::Auto, CopyMode::Buffered, CopyMode::Native] {
            set_copy_mode(mode);
            let dst = dir.join(format!("b-{mode}"));
            assert_eq!(copy_file(&src, &dst).unwrap(), data.len() as u64);
            assert_eq!(std::fs::read(&dst).unwrap(), data);
        }
        set_copy_mode(CopyMode::Auto);

        let linked = dir.join("c");
        link_or_copy(&src, &linked).unwrap();
        assert_eq!(std::fs::read(&linked).unwrap(), data);

        // Replacing an existing (stale) destination, a hard link of the
        // source, and the source itself
        let stale = dir.join("d");
        std::fs::write(&stale, b"stale").unwrap();
        link_or_copy(&src, &stale).unwrap();
        assert_eq!(std::fs::read(&stale).unwrap(), data);
        link_or_copy(&src, &linked).unwrap();
        link_or_copy(&src, &src).unwrap();
        assert_eq!(std::fs::read(&linked).unwrap(), data);
        assert_eq!(std::fs::read(&src).unwrap(), data);
        std::fs::remove_dir_all(&dir).unwrap();
    }

    #[test]
    fn copy_dir() {
        let _guard = MODE_LOCK.lock().unwrap();
        let dir = test_dir("dir");
        let src = dir.join("src");
        std::fs::create_dir_all(src.join("sub")).unwrap();
        std::fs::write(src.join("x"), b"x").unwrap();
        std::fs::write(src.join("sub").join("y"), b"y").unwrap();
        for mode in [CopyMode::Buffered, CopyMode::Native] {
            set_copy_mode(mode);
            let dst = dir.join(format!("dst-{mode}"));
            copy_dir_all(&src, &dst).unwrap();
            assert_eq!(std::fs::read(dst.join("x")).unwrap(), b"x");
            assert_eq!(std::fs::read(dst.join("sub").join("y")).unwrap(), b"y");
        }
        set_copy_mode(CopyMode::Auto);
        std::fs::remove_dir_all(&dir).unwrap();
    }

    #[test]
    fn modes() {
        let _guard = MODE_LOCK.lock().unwrap();
        for mode in [CopyMode::Auto, CopyMode::Buffered, CopyMode::Native] {
            assert_eq!(mode.name().parse::<CopyMode>().unwrap(), mode);
            set_copy_mode(mode);
            assert_eq!(copy_mode(), mode);
        }
        set_copy_mode(CopyMode::Auto);
        assert_eq!(
            " Buffered ".parse::<CopyMode>().unwrap(),
            CopyMode::Buffered
        );
        assert!("fast".parse::<CopyMode>().is_err());

        assert_eq!(mode_from_env(None), CopyMode::Auto);
        assert_eq!(mode_from_env(Some("")), CopyMode::Auto);
        assert_eq!(mode_from_env(Some("native")), CopyMode::Native);
        assert_eq!(mode_from_env(Some("BUFFERED")), CopyMode::Buffered);
        assert_eq!(mode_from_env(Some("nonsense")), CopyMode::Auto);
    }

    #[test]
    fn network_magic() {
        assert!(is_network_magic(0x6969)); // NFS
        assert!(is_network_magic(0xFF534D42)); // CIFS, unsigned f_type
        assert!(is_network_magic(0xFF534D42u32 as i32 as i64)); // signed
        assert!(!is_network_magic(0xEF53)); // ext4
        assert!(!is_network_magic(0x9123683E)); // btrfs
        assert!(!is_network_magic(0x01021994)); // tmpfs
    }

    #[test]
    fn local_fs_is_not_network() {
        // The temporary directory is local on CI and developer machines
        let dir = test_dir("netfs");
        assert!(!is_network_fs(&dir));
        assert!(!is_network_fs(&dir.join("does-not-exist").join("file")));
        std::fs::remove_dir_all(&dir).unwrap();
    }
}
