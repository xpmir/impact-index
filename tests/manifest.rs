//! Tests for the index directory manifest: version stamping, the
//! load-time compatibility check, and the migration entry point
//! (`impact_index::manifest::update_index`).

use std::collections::HashSet;

use helpers::index::TestIndex;
use impact_index::base::{load_index, DocId};
use impact_index::builder::BuilderOptions;
use impact_index::compress::{docid::EliasFanoCompressor, impact::Identity, CompressionTransform};
use impact_index::manifest::{
    check_index_manifest, read_manifest, update_index, write_manifest_raw, BuilderInfo, IndexKind,
    Manifest, CURRENT_FORMAT_VERSION, MANIFEST_FILENAME,
};
use impact_index::transforms::split::SplitIndexTransform;
use impact_index::transforms::IndexTransform;
use temp_dir::TempDir;

fn build_test_index() -> TestIndex {
    TestIndex::new(
        50,
        200,
        5.,
        10,
        Some(7),
        BuilderOptions {
            checkpoint_frequency: 0,
            in_memory_threshold: 16,
            checkpoint_flush_ratio: 0.5,
        },
        &HashSet::<DocId>::from([]),
    )
}

fn overwrite_manifest(dir: &std::path::Path, manifest: &Manifest) {
    write_manifest_raw(dir, manifest).expect("write manifest");
}

/// Building a forward index writes a manifest.json describing it.
#[test]
fn test_forward_index_writes_manifest() {
    let data = build_test_index();
    let manifest = read_manifest(data.indexer.folder())
        .expect("io error")
        .expect("manifest.json should have been written by Indexer::build()");

    assert_eq!(manifest.format_version, CURRENT_FORMAT_VERSION);
    assert_eq!(manifest.index_kind, IndexKind::Forward);
    assert_eq!(manifest.builder.block_size, Some(16));
    assert!(!manifest.builder.library_version.is_empty());
}

/// Compressing an index writes a manifest.json describing the compressed
/// directory (distinct from the forward index it was built from).
#[test]
fn test_compressed_index_writes_manifest() {
    let mut data = build_test_index();
    let index = data.indexer.to_index(true);

    let dir = TempDir::new().expect("tmp dir");
    let transform = CompressionTransform {
        max_block_size: 64,
        doc_ids_compressor_factory: Box::new(EliasFanoCompressor {}),
        impacts_compressor_factory: Box::new(Identity {}),
    };
    transform.process(dir.path(), &index).expect("compress");

    let manifest = read_manifest(dir.path())
        .expect("io error")
        .expect("manifest.json should have been written by CompressionTransform");
    assert_eq!(manifest.format_version, CURRENT_FORMAT_VERSION);
    assert_eq!(manifest.index_kind, IndexKind::Compressed);
    assert_eq!(manifest.builder.block_size, Some(64));
    assert!(manifest
        .builder
        .codecs
        .as_deref()
        .unwrap_or("")
        .contains("EliasFano"));

    // The index still loads normally.
    let _loaded = load_index(dir.path(), true);
}

/// Splitting an index writes a manifest describing the split wrapper,
/// while the inner compressed index keeps its own manifest.
#[test]
fn test_split_index_writes_manifest() {
    let mut data = build_test_index();
    let index = data.indexer.to_index(true);

    let dir = TempDir::new().expect("tmp dir");
    let sink = Box::new(CompressionTransform {
        max_block_size: 64,
        doc_ids_compressor_factory: Box::new(EliasFanoCompressor {}),
        impacts_compressor_factory: Box::new(Identity {}),
    });
    let transform = SplitIndexTransform {
        sink,
        quantiles: vec![0.9],
    };
    transform.process(dir.path(), &index).expect("split");

    let outer = read_manifest(dir.path())
        .expect("io error")
        .expect("split manifest missing");
    assert_eq!(outer.index_kind, IndexKind::Split);

    let inner = read_manifest(&dir.path().join("inner"))
        .expect("io error")
        .expect("inner compressed manifest missing");
    assert_eq!(inner.index_kind, IndexKind::Compressed);

    let _loaded = load_index(dir.path(), true);
}

/// A manifest claiming a newer format version than this build supports
/// must produce the actionable "upgrade" error, not a silent misread.
#[test]
fn test_newer_format_version_error_message() {
    let dir = TempDir::new().expect("tmp dir");
    let manifest = Manifest::new(IndexKind::Compressed, BuilderInfo::new());
    let mut manifest = manifest;
    manifest.format_version = CURRENT_FORMAT_VERSION + 7;
    overwrite_manifest(dir.path(), &manifest);

    let err = check_index_manifest(dir.path()).expect_err("expected a version-mismatch error");
    let msg = err.to_string();
    assert!(
        msg.contains(&format!("v{}", CURRENT_FORMAT_VERSION + 7)),
        "message should mention the found version: {msg}"
    );
    assert!(
        msg.contains("newer than this library supports"),
        "message should explain the mismatch: {msg}"
    );
    assert!(
        msg.contains("upgrade impact-index"),
        "message should say how to fix it: {msg}"
    );
}

/// A manifest claiming an older format version must produce the
/// actionable "run the migration" error.
#[test]
fn test_older_format_version_error_message() {
    let dir = TempDir::new().expect("tmp dir");
    let mut manifest = Manifest::new(IndexKind::Forward, BuilderInfo::new());
    manifest.format_version = 0;
    overwrite_manifest(dir.path(), &manifest);

    let err = check_index_manifest(dir.path()).expect_err("expected a version-mismatch error");
    let msg = err.to_string();
    assert!(msg.contains("v0"), "message should mention v0: {msg}");
    assert!(
        msg.contains(&format!("v{}", CURRENT_FORMAT_VERSION)),
        "message should mention the required version: {msg}"
    );
    assert!(
        msg.contains("Index.update(path)") && msg.contains("impact_index::update_index"),
        "message should point at both migration entry points: {msg}"
    );
}

/// `load_index` must surface the same actionable error (as a panic
/// message, per this crate's existing error-handling convention) rather
/// than crashing opaquely or silently misreading the directory.
#[test]
fn test_load_index_panics_with_actionable_message_on_mismatch() {
    let mut data = build_test_index();
    let index = data.indexer.to_index(true);
    let dir = TempDir::new().expect("tmp dir");
    let transform = CompressionTransform {
        max_block_size: 64,
        doc_ids_compressor_factory: Box::new(EliasFanoCompressor {}),
        impacts_compressor_factory: Box::new(Identity {}),
    };
    transform.process(dir.path(), &index).expect("compress");

    let mut manifest = read_manifest(dir.path()).unwrap().unwrap();
    manifest.format_version = CURRENT_FORMAT_VERSION + 1;
    overwrite_manifest(dir.path(), &manifest);

    let path = dir.path().to_path_buf();
    let prev_hook = std::panic::take_hook();
    std::panic::set_hook(Box::new(|_| {})); // silence the default panic backtrace print
    let result = std::panic::catch_unwind(move || load_index(&path, true));
    std::panic::set_hook(prev_hook);

    let payload = match result {
        Ok(_) => panic!("load_index should panic on a version mismatch"),
        Err(payload) => payload,
    };
    let msg = payload
        .downcast_ref::<String>()
        .cloned()
        .or_else(|| payload.downcast_ref::<&str>().map(|s| s.to_string()))
        .unwrap_or_default();
    assert!(
        msg.contains("upgrade impact-index"),
        "panic message should be actionable: {msg}"
    );
}

/// A directory with no manifest.json at all (every index built before
/// this feature existed) must keep loading normally.
#[test]
fn test_manifest_less_directory_still_loads() {
    let mut data = build_test_index();
    let index = data.indexer.to_index(true);
    let dir = TempDir::new().expect("tmp dir");
    let transform = CompressionTransform {
        max_block_size: 64,
        doc_ids_compressor_factory: Box::new(EliasFanoCompressor {}),
        impacts_compressor_factory: Box::new(Identity {}),
    };
    transform.process(dir.path(), &index).expect("compress");

    // Simulate a legacy index directory: remove the manifest entirely.
    std::fs::remove_file(dir.path().join(MANIFEST_FILENAME)).expect("remove manifest");
    assert!(read_manifest(dir.path()).unwrap().is_none());

    // Must still load without error.
    let loaded = load_index(dir.path(), true);
    assert!(loaded.len() > 0);

    // And a manifest should now have been opportunistically stamped.
    let manifest = read_manifest(dir.path())
        .unwrap()
        .expect("load_index should have stamped a manifest for the legacy directory");
    assert_eq!(manifest.format_version, CURRENT_FORMAT_VERSION);
    assert_eq!(manifest.index_kind, IndexKind::Compressed);
}

/// `update_index` migrates a legacy (manifest-less) directory in place.
#[test]
fn test_update_index_migrates_legacy_directory_in_place() {
    let mut data = build_test_index();
    let index = data.indexer.to_index(true);
    let dir = TempDir::new().expect("tmp dir");
    let transform = CompressionTransform {
        max_block_size: 64,
        doc_ids_compressor_factory: Box::new(EliasFanoCompressor {}),
        impacts_compressor_factory: Box::new(Identity {}),
    };
    transform.process(dir.path(), &index).expect("compress");
    std::fs::remove_file(dir.path().join(MANIFEST_FILENAME)).expect("remove manifest");

    let result_path = update_index(dir.path(), None).expect("update_index");
    assert_eq!(result_path, dir.path());

    let manifest = read_manifest(dir.path())
        .unwrap()
        .expect("update_index should have written a manifest");
    assert_eq!(manifest.format_version, CURRENT_FORMAT_VERSION);
    assert_eq!(manifest.index_kind, IndexKind::Compressed);

    // The migrated directory still loads and answers queries.
    let loaded = load_index(dir.path(), true);
    assert!(loaded.len() > 0);
}

/// `update_index` can migrate into a separate destination, leaving the
/// source directory untouched.
#[test]
fn test_update_index_with_dest_leaves_source_untouched() {
    let mut data = build_test_index();
    let index = data.indexer.to_index(true);
    let src_dir = TempDir::new().expect("tmp dir");
    let transform = CompressionTransform {
        max_block_size: 64,
        doc_ids_compressor_factory: Box::new(EliasFanoCompressor {}),
        impacts_compressor_factory: Box::new(Identity {}),
    };
    transform.process(src_dir.path(), &index).expect("compress");
    std::fs::remove_file(src_dir.path().join(MANIFEST_FILENAME)).expect("remove manifest");

    let dest_dir = TempDir::new().expect("tmp dir");
    let dest_path = dest_dir.path().join("migrated");

    let result_path =
        update_index(src_dir.path(), Some(&dest_path)).expect("update_index with dest");
    assert_eq!(result_path, dest_path);

    // Source untouched: still no manifest.
    assert!(read_manifest(src_dir.path()).unwrap().is_none());

    // Destination has a manifest and loads correctly.
    let manifest = read_manifest(&dest_path)
        .unwrap()
        .expect("destination should have a manifest");
    assert_eq!(manifest.format_version, CURRENT_FORMAT_VERSION);
    let loaded = load_index(&dest_path, true);
    assert!(loaded.len() > 0);
}
