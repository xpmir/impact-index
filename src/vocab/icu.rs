//! ICU4X word segmentation data, downloaded on demand.
//!
//! The word-break rules are built into the binary (~50 KB), but the
//! dictionaries used for scripts written without spaces (Han, Kana, Thai,
//! Lao, Khmer, Burmese) would add ~3 MB, so they are loaded at runtime from
//! the official ICU4X data blob (`full.postcard`, a release asset of
//! [ICU4X](https://github.com/unicode-org/icu4x/releases)), downloaded into
//! the cache directory (see [`super::lucene::morph::cache_dir`]) when first
//! needed. Its segmentation is identical to the data built into
//! `icu_segmenter` of the same version.
//!
//! When bumping `icu_segmenter`, bump [`ICU4X_VERSION`] and [`SHA256`]
//! together (the digest is shown on the release page).

use std::io::{Error, ErrorKind};
use std::path::{Path, PathBuf};
use std::sync::{Mutex, OnceLock};

use icu_segmenter::{WordSegmenter, WordSegmenterBorrowed};
use sha2::{Digest, Sha256};

/// ICU4X release whose data blob is used (same as `icu_segmenter`)
pub const ICU4X_VERSION: &str = "2.4.0";

/// SHA-256 of the data blob of [`ICU4X_VERSION`]
pub const SHA256: &str = "73e5e956536fbc571f484daa514cf1eee346c95fc41759478219de23777a0e88";

const FILE_NAME: &str = "full.postcard";

fn url() -> String {
    format!(
        "https://github.com/unicode-org/icu4x/releases/download/icu%40{}/{}",
        ICU4X_VERSION, FILE_NAME
    )
}

/// Path set with [`set_data_path`]
static DATA_PATH: Mutex<Option<PathBuf>> = Mutex::new(None);

/// Loaded segmenter (process-wide: the data is the same for every analyzer)
static SEGMENTER: OnceLock<WordSegmenter> = OnceLock::new();

/// Serializes loading (so that the data is downloaded once)
static LOADING: Mutex<()> = Mutex::new(());

/// Sets the path of the ICU4X data blob, instead of the cache (e.g. for a
/// machine without network access). Must be called before the first ICU
/// tokenization.
pub fn set_data_path(path: PathBuf) {
    *DATA_PATH.lock().unwrap() = Some(path);
}

/// Path of the data blob in the cache
fn cached_path() -> PathBuf {
    super::lucene::morph::cache_dir()
        .join(format!("icu4x-{}", ICU4X_VERSION))
        .join(FILE_NAME)
}

fn sha256_hex(bytes: &[u8]) -> String {
    Sha256::digest(bytes)
        .iter()
        .map(|b| format!("{:02x}", b))
        .collect()
}

/// Returns the path of the ICU4X data blob in the cache, downloading it if
/// needed
pub fn download_data() -> std::io::Result<PathBuf> {
    let target = cached_path();
    if target.exists() {
        return Ok(target);
    }
    let url = url();
    log::info!("Downloading the ICU4X segmentation data from {}", url);
    let response = ureq::get(&url)
        .call()
        .map_err(|e| Error::other(format!("cannot download {}: {}", url, e)))?;
    let mut bytes = Vec::new();
    std::io::copy(&mut response.into_reader(), &mut bytes)?;
    let digest = sha256_hex(&bytes);
    if digest != SHA256 {
        return Err(Error::new(
            ErrorKind::InvalidData,
            format!(
                "{} has SHA-256 {}, expected {} (incomplete download?)",
                url, digest, SHA256
            ),
        ));
    }
    // Written to a temporary file, renamed when complete
    let dir = target.parent().expect("cache path has a parent");
    std::fs::create_dir_all(dir)?;
    let tmp = dir.join(format!(".{}.{}", FILE_NAME, std::process::id()));
    std::fs::write(&tmp, &bytes)?;
    std::fs::rename(&tmp, &target)?;
    Ok(target)
}

fn load(path: &Path) -> std::io::Result<WordSegmenter> {
    let invalid = |e: icu_provider::DataError| {
        Error::new(
            ErrorKind::InvalidData,
            format!("invalid ICU4X data {}: {}", path.display(), e),
        )
    };
    let bytes = std::fs::read(path).map_err(|e| {
        Error::new(
            e.kind(),
            format!("cannot read ICU4X data {}: {}", path.display(), e),
        )
    })?;
    let provider = icu_provider_blob::BlobDataProvider::try_new_from_blob(bytes.into_boxed_slice())
        .map_err(invalid)?;
    let mut segmenter =
        WordSegmenter::try_new_for_non_complex_scripts(Default::default()).map_err(invalid)?;
    segmenter
        .load_dictionary_with_buffer_provider(&provider)
        .map_err(invalid)?;
    Ok(segmenter)
}

/// Dictionary-based word segmenter (UAX#29 extended with dictionary
/// segmentation of Han/Kana/Thai/Lao/Khmer/Burmese), loading its data (from
/// the path set with [`set_data_path`], or else the cache, downloading it if
/// needed) on first use
pub fn word_segmenter() -> std::io::Result<WordSegmenterBorrowed<'static>> {
    if let Some(segmenter) = SEGMENTER.get() {
        return Ok(segmenter.as_borrowed());
    }
    let _guard = LOADING.lock().unwrap_or_else(|e| e.into_inner());
    if SEGMENTER.get().is_none() {
        let path = match DATA_PATH.lock().unwrap().clone() {
            Some(path) => path,
            None => download_data()?,
        };
        let _ = SEGMENTER.set(load(&path)?);
    }
    Ok(SEGMENTER.get().expect("segmenter just set").as_borrowed())
}
