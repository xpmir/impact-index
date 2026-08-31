//! Index directory manifest: format versioning and migration.
//!
//! Every index directory (forward, compressed, split) carries a
//! `manifest.json` recording the on-disk format version, the kind of
//! index, when it was built, and a human-readable summary of the build
//! parameters (block size, codecs, analyzer). This lets [`crate::base::load_index`]
//! detect a format change made in a later release: instead of silently
//! misreading bytes (as `CompressedIndexInformation::read_binary` used to)
//! or panicking with an opaque message, it raises a clear, actionable
//! error telling the caller whether to upgrade the library or run the
//! migration tool.
//!
//! Index directories that predate this feature have no `manifest.json` at
//! all. These are treated as "legacy" and always keep loading normally
//! (see [`check_index_manifest`]) -- a missing manifest is never an error,
//! only a version *mismatch* is.
//!
//! # Extending for a new format version
//!
//! When a future change (e.g. P1a's per-block `min_dl` statistics, or
//! P2's full re-transform) alters the on-disk layout:
//!
//! 1. Bump [`CURRENT_FORMAT_VERSION`].
//! 2. Write a `migrate_vN_to_vN1(path: &Path) -> io::Result<()>` function
//!    that rewrites a directory at version `N` into version `N + 1`.
//! 3. Append `(N, migrate_vN_to_vN1)` to the `steps` table inside
//!    [`update_index`]. Nothing else about `update_index`'s chaining logic
//!    needs to change -- it walks the steps one version at a time so
//!    directories more than one version behind still migrate correctly.
//! 4. If readers can still cheaply understand the *old* bytes, keep doing
//!    so (as `read_binary` does for one version back); otherwise, loading
//!    an un-migrated directory should fail with the actionable error from
//!    [`check_format_version`], pointing the user at `update_index`.

use std::fs::File;
use std::io::{self, BufReader, BufWriter};
use std::path::{Path, PathBuf};
use std::time::{SystemTime, UNIX_EPOCH};

use serde::{Deserialize, Serialize};

use crate::base::BUILDER_INDEX_CBOR;

/// Name of the manifest file written into every index directory.
pub const MANIFEST_FILENAME: &str = "manifest.json";

/// Current on-disk directory format version understood by this build.
///
/// This tracks the *directory layout contract* (which files exist, what
/// the manifest promises), independently of the low-level binary
/// magic/version numbers used by individual files (e.g.
/// `FORWARD_INDEX_VERSION` in `index.rs`, `COMPRESSED_INDEX_VERSION` in
/// `compress/mod.rs`). Those stay as fine-grained safety checks on their
/// own file formats; this is the version [`update_index`] migrates.
pub const CURRENT_FORMAT_VERSION: u32 = 1;

/// Kind of index directory described by a manifest.
#[derive(Serialize, Deserialize, Debug, Clone, Copy, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum IndexKind {
    /// Raw forward index (`postings.dat` + `information.{cbor,bin}`).
    Forward,
    /// Block-compressed index (`docids.dat` + `impacts.dat` + `index.bin`).
    Compressed,
    /// Quantile-split index wrapping an inner (typically compressed) index.
    Split,
}

impl std::fmt::Display for IndexKind {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let s = match self {
            IndexKind::Forward => "forward",
            IndexKind::Compressed => "compressed",
            IndexKind::Split => "split",
        };
        write!(f, "{}", s)
    }
}

/// Best-effort, human-readable build parameters. Every field is optional
/// and purely informational -- nothing here is consulted when deciding
/// whether an index can be loaded.
#[derive(Serialize, Deserialize, Debug, Clone, Default)]
pub struct BuilderInfo {
    /// `CARGO_PKG_VERSION` of the impact-index crate that wrote the index.
    pub library_version: String,

    /// Postings-per-block, for index kinds that use fixed-size blocks.
    #[serde(skip_serializing_if = "Option::is_none", default)]
    pub block_size: Option<usize>,

    /// Human-readable codec summary (e.g. doc-id / impact compressors).
    #[serde(skip_serializing_if = "Option::is_none", default)]
    pub codecs: Option<String>,

    /// Human-readable analyzer summary (stemmer, stop words).
    #[serde(skip_serializing_if = "Option::is_none", default)]
    pub analyzer: Option<String>,
}

impl BuilderInfo {
    /// Creates a new `BuilderInfo` stamped with the current crate version.
    pub fn new() -> Self {
        Self {
            library_version: env!("CARGO_PKG_VERSION").to_string(),
            block_size: None,
            codecs: None,
            analyzer: None,
        }
    }

    pub fn with_block_size(mut self, block_size: usize) -> Self {
        self.block_size = Some(block_size);
        self
    }

    pub fn with_codecs(mut self, codecs: impl Into<String>) -> Self {
        self.codecs = Some(codecs.into());
        self
    }

    pub fn with_analyzer(mut self, analyzer: impl Into<String>) -> Self {
        self.analyzer = Some(analyzer.into());
        self
    }
}

/// On-disk `manifest.json` contents.
#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct Manifest {
    /// Directory format version (see [`CURRENT_FORMAT_VERSION`]).
    pub format_version: u32,
    /// Kind of index stored in this directory.
    pub index_kind: IndexKind,
    /// ISO-8601 creation timestamp (UTC), e.g. `2026-08-31T12:00:00Z`.
    pub created: String,
    /// Best-effort build parameters, for human inspection.
    #[serde(default)]
    pub builder: BuilderInfo,
}

impl Manifest {
    /// Creates a manifest for the current format version, stamped "now".
    pub fn new(index_kind: IndexKind, builder: BuilderInfo) -> Self {
        Self {
            format_version: CURRENT_FORMAT_VERSION,
            index_kind,
            created: current_iso_date(),
            builder,
        }
    }
}

/// Returns the current UTC time as an ISO-8601 string, without pulling in
/// a datetime dependency (the crate has none).
fn current_iso_date() -> String {
    let secs = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0);
    let days = (secs / 86_400) as i64;
    let time_of_day = secs % 86_400;
    let (y, m, d) = civil_from_days(days);
    format!(
        "{:04}-{:02}-{:02}T{:02}:{:02}:{:02}Z",
        y,
        m,
        d,
        time_of_day / 3600,
        (time_of_day % 3600) / 60,
        time_of_day % 60
    )
}

/// Converts a day count since the Unix epoch (1970-01-01) into a
/// (year, month, day) civil calendar date. Proleptic Gregorian calendar,
/// after Howard Hinnant's `civil_from_days` algorithm (public domain).
fn civil_from_days(z: i64) -> (i64, u32, u32) {
    let z = z + 719_468;
    let era = if z >= 0 { z } else { z - 146_096 } / 146_097;
    let doe = (z - era * 146_097) as u64; // [0, 146096]
    let yoe = (doe - doe / 1460 + doe / 36524 - doe / 146096) / 365; // [0, 399]
    let y = yoe as i64 + era * 400;
    let doy = doe - (365 * yoe + yoe / 4 - yoe / 100); // [0, 365]
    let mp = (5 * doy + 2) / 153; // [0, 11]
    let d = (doy - (153 * mp + 2) / 5 + 1) as u32; // [1, 31]
    let m = if mp < 10 { mp + 3 } else { mp - 9 } as u32; // [1, 12]
    let y = if m <= 2 { y + 1 } else { y };
    (y, m, d)
}

/// Formats the "newer than supported" actionable error message.
fn newer_message(found: u32, supported: u32) -> String {
    format!(
        "index format v{} is newer than this library supports (v{}) — upgrade impact-index",
        found, supported
    )
}

/// Formats the "older than supported, needs migration" actionable error message.
fn older_message(found: u32, supported: u32) -> String {
    format!(
        "index format v{}, this version requires v{} — run Index.update(path) (Python) or impact_index::update_index (Rust) to migrate",
        found, supported
    )
}

/// Checks a format version found on disk against the version supported by
/// this build, returning an actionable [`io::Error`] on mismatch.
pub fn check_format_version(found: u32, supported: u32) -> io::Result<()> {
    use std::cmp::Ordering;
    match found.cmp(&supported) {
        Ordering::Equal => Ok(()),
        Ordering::Greater => Err(io::Error::new(
            io::ErrorKind::InvalidData,
            newer_message(found, supported),
        )),
        Ordering::Less => Err(io::Error::new(
            io::ErrorKind::InvalidData,
            older_message(found, supported),
        )),
    }
}

/// Reads `manifest.json` from `path`, if present.
///
/// Returns `Ok(None)` when there is no manifest at all (a legacy index
/// directory). Returns `Err` only for I/O errors or a malformed file.
pub fn read_manifest(path: &Path) -> io::Result<Option<Manifest>> {
    let manifest_path = path.join(MANIFEST_FILENAME);
    if !manifest_path.exists() {
        return Ok(None);
    }
    let file = File::open(&manifest_path)?;
    let manifest: Manifest = serde_json::from_reader(BufReader::new(file)).map_err(|e| {
        io::Error::new(
            io::ErrorKind::InvalidData,
            format!("Malformed manifest at {}: {}", manifest_path.display(), e),
        )
    })?;
    Ok(Some(manifest))
}

/// Writes a `manifest.json` describing `index_kind` into `path`, stamped
/// with the current format version and "now".
pub fn write_manifest(path: &Path, index_kind: IndexKind, builder: BuilderInfo) -> io::Result<()> {
    write_manifest_raw(path, &Manifest::new(index_kind, builder))
}

/// Writes an already-constructed [`Manifest`] verbatim, without forcing
/// `format_version` to [`CURRENT_FORMAT_VERSION`]. This is the primitive
/// [`write_manifest`] builds on; it is also useful for tests that need to
/// simulate a manifest written by a different library version.
pub fn write_manifest_raw(path: &Path, manifest: &Manifest) -> io::Result<()> {
    let manifest_path = path.join(MANIFEST_FILENAME);
    let file = File::create(&manifest_path)?;
    serde_json::to_writer_pretty(BufWriter::new(file), manifest)
        .map_err(|e| io::Error::new(io::ErrorKind::Other, e.to_string()))
}

/// Verifies that an index directory's manifest (if present) is compatible
/// with this build.
///
/// - `Ok(Some(manifest))`: a manifest is present and matches
///   [`CURRENT_FORMAT_VERSION`] -- load normally.
/// - `Ok(None)`: no manifest at all. This is a legacy directory (predates
///   manifests entirely) and must keep loading -- callers should treat it
///   as "current format" and may opportunistically stamp a manifest.
/// - `Err`: a manifest is present but its `format_version` does not match
///   -- the error message is actionable (upgrade the library, or run the
///   migration).
pub fn check_index_manifest(path: &Path) -> io::Result<Option<Manifest>> {
    match read_manifest(path)? {
        None => Ok(None),
        Some(manifest) => {
            check_format_version(manifest.format_version, CURRENT_FORMAT_VERSION)?;
            Ok(Some(manifest))
        }
    }
}

/// Best-effort detection of the kind of index directory at `path`, used
/// when stamping a manifest for a legacy (manifest-less) directory.
pub fn detect_index_kind(path: &Path) -> IndexKind {
    if path.join(BUILDER_INDEX_CBOR).exists() {
        IndexKind::Forward
    } else if path.join("inner").is_dir() {
        IndexKind::Split
    } else {
        IndexKind::Compressed
    }
}

/// Recursively copies a directory tree. Used by [`update_index`] when
/// migrating into a separate destination rather than in place.
fn copy_dir_all(src: &Path, dst: &Path) -> io::Result<()> {
    std::fs::create_dir_all(dst)?;
    for entry in std::fs::read_dir(src)? {
        let entry = entry?;
        let ty = entry.file_type()?;
        let dst_path = dst.join(entry.file_name());
        if ty.is_dir() {
            copy_dir_all(&entry.path(), &dst_path)?;
        } else {
            std::fs::copy(entry.path(), dst_path)?;
        }
    }
    Ok(())
}

/// v0 (legacy / absent manifest) -> v1.
///
/// v1 is the format every reader in this build already understands, so
/// this step performs no data rewrite at all -- it only stamps the
/// manifest that future loaders will require. This is the template for
/// later steps that *do* need to rewrite data (e.g. recomputing per-block
/// statistics for P1a): read what's needed from the existing files,
/// write the new/changed files, then write the manifest last so a step
/// that fails partway never leaves a directory claiming a version it
/// doesn't actually have.
fn migrate_legacy_to_v1(path: &Path) -> io::Result<()> {
    let kind = detect_index_kind(path);
    write_manifest(path, kind, BuilderInfo::new())
}

/// Migrates the index directory at `path` to [`CURRENT_FORMAT_VERSION`].
///
/// If `dest` is `Some`, the directory is first copied there and the
/// migration runs on the copy, leaving `path` untouched; if `dest` is
/// `None` (or equal to `path`), the migration runs in place.
///
/// Returns the path that now holds the migrated index (`dest` or `path`).
///
/// # Migration chain
///
/// Migrations run one version-step at a time via the `steps` table below,
/// keyed by the version each step starts from. To support a directory
/// more than one version behind, just register every intermediate step --
/// `update_index` walks them in order until it reaches
/// [`CURRENT_FORMAT_VERSION`]. See the module docs for how to add a step.
pub fn update_index(path: &Path, dest: Option<&Path>) -> io::Result<PathBuf> {
    let target: PathBuf = match dest {
        Some(d) if d != path => {
            copy_dir_all(path, d)?;
            d.to_path_buf()
        }
        Some(d) => d.to_path_buf(),
        None => path.to_path_buf(),
    };

    // Absent manifest == legacy == version 0 for migration purposes. This
    // is distinct from `check_index_manifest`'s "no manifest -> keep
    // loading" behavior at load time: here we're explicitly asked to
    // bring the directory up to date.
    let mut current_version = match read_manifest(&target)? {
        Some(m) => m.format_version,
        None => 0,
    };

    // Ordered chain of migration steps, keyed by the version they start
    // from. Add new entries here as the format evolves -- e.g. once P1a
    // lands: `(1, migrate_v1_to_v2)`.
    let steps: &[(u32, fn(&Path) -> io::Result<()>)] = &[(0, migrate_legacy_to_v1)];

    while current_version != CURRENT_FORMAT_VERSION {
        if current_version > CURRENT_FORMAT_VERSION {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                newer_message(current_version, CURRENT_FORMAT_VERSION),
            ));
        }
        match steps.iter().find(|(from, _)| *from == current_version) {
            Some((_, step)) => {
                step(&target)?;
                current_version += 1;
            }
            None => {
                return Err(io::Error::new(
                    io::ErrorKind::InvalidData,
                    format!(
                        "no migration registered from index format v{} to v{}",
                        current_version,
                        current_version + 1
                    ),
                ));
            }
        }
    }

    Ok(target)
}

// Unit tests for this module live in `tests/manifest.rs` as integration
// tests: the crate's `[lib] test = false` (see `Cargo.toml`) means
// `#[cfg(test)]` modules inside `src/` are never compiled or run by
// `cargo test` -- only the top-level `tests/*.rs` binaries are.
