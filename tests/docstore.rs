use std::collections::HashMap;

use impact_index::docstore::{
    builder::{BuilderOptions, DocumentStoreBuilder},
    store::{ContentAccess, DocumentStore},
    DocumentData, CHECKPOINT_FILE,
};
use temp_dir::TempDir;

fn init_logger() {
    let _ = env_logger::builder().is_test(true).try_init();
}

fn make_doc(docno: &str, content: &[u8]) -> DocumentData {
    let mut keys = HashMap::new();
    keys.insert("docno".to_string(), docno.to_string());
    DocumentData {
        keys,
        content: content.to_vec(),
    }
}

fn make_doc_multi_keys(docno: &str, url: &str, content: &[u8]) -> DocumentData {
    let mut keys = HashMap::new();
    keys.insert("docno".to_string(), docno.to_string());
    keys.insert("url".to_string(), url.to_string());
    DocumentData {
        keys,
        content: content.to_vec(),
    }
}

#[test]
fn test_roundtrip_basic() {
    init_logger();
    let dir = TempDir::new().unwrap();
    let path = dir.path().join("store");

    let num_docs = 100;
    let mut builder = DocumentStoreBuilder::new(&path, 4096, 3).unwrap();
    for i in 0..num_docs {
        let docno = format!("DOC-{:04}", i);
        let content = format!("Content of document {}", i);
        builder.add(&make_doc(&docno, content.as_bytes())).unwrap();
    }
    builder.build().unwrap();

    let store = DocumentStore::load(&path, ContentAccess::Memory).unwrap();
    assert_eq!(store.num_documents(), num_docs);

    for i in 0..num_docs {
        let docs = store.get_by_number(&[i]).unwrap();
        assert_eq!(docs.len(), 1);
        let expected_docno = format!("DOC-{:04}", i);
        let expected_content = format!("Content of document {}", i);
        assert_eq!(docs[0].keys["docno"], expected_docno);
        assert_eq!(docs[0].content, expected_content.as_bytes());
    }
}

#[test]
fn test_key_lookup() {
    init_logger();
    let dir = TempDir::new().unwrap();
    let path = dir.path().join("store");

    let mut builder = DocumentStoreBuilder::new(&path, 4096, 3).unwrap();
    for i in 0..50u64 {
        let docno = format!("DOC-{:04}", i);
        let content = format!("content-{}", i);
        builder.add(&make_doc(&docno, content.as_bytes())).unwrap();
    }
    builder.build().unwrap();

    let store = DocumentStore::load(&path, ContentAccess::Memory).unwrap();
    let results = store
        .get_by_key("docno", &["DOC-0010", "DOC-0025", "DOC-0049"])
        .unwrap();
    assert_eq!(results.len(), 3);
    assert_eq!(results[0].as_ref().unwrap().keys["docno"], "DOC-0010");
    assert_eq!(results[0].as_ref().unwrap().content, b"content-10");
    assert_eq!(results[1].as_ref().unwrap().keys["docno"], "DOC-0025");
    assert_eq!(results[2].as_ref().unwrap().keys["docno"], "DOC-0049");
}

#[test]
fn test_missing_key_lookup() {
    init_logger();
    let dir = TempDir::new().unwrap();
    let path = dir.path().join("store");

    let mut builder = DocumentStoreBuilder::new(&path, 4096, 3).unwrap();
    builder.add(&make_doc("DOC-0001", b"hello")).unwrap();
    builder.build().unwrap();

    let store = DocumentStore::load(&path, ContentAccess::Memory).unwrap();
    let results = store
        .get_by_key("docno", &["DOC-0001", "NONEXISTENT"])
        .unwrap();
    assert_eq!(results.len(), 2);
    assert!(results[0].is_some());
    assert!(results[1].is_none());
}

#[test]
fn test_batch_retrieval() {
    init_logger();
    let dir = TempDir::new().unwrap();
    let path = dir.path().join("store");

    // Small block size to force multiple blocks
    let mut builder = DocumentStoreBuilder::new(&path, 64, 3).unwrap();
    for i in 0..20u64 {
        let docno = format!("DOC-{:04}", i);
        builder
            .add(&make_doc(&docno, format!("c{}", i).as_bytes()))
            .unwrap();
    }
    builder.build().unwrap();

    let store = DocumentStore::load(&path, ContentAccess::Memory).unwrap();
    let docs = store.get_by_number(&[0, 5, 10, 15, 19]).unwrap();
    assert_eq!(docs.len(), 5);
    assert_eq!(docs[0].keys["docno"], "DOC-0000");
    assert_eq!(docs[1].keys["docno"], "DOC-0005");
    assert_eq!(docs[2].keys["docno"], "DOC-0010");
    assert_eq!(docs[3].keys["docno"], "DOC-0015");
    assert_eq!(docs[4].keys["docno"], "DOC-0019");
}

#[test]
fn test_empty_content() {
    init_logger();
    let dir = TempDir::new().unwrap();
    let path = dir.path().join("store");

    let mut builder = DocumentStoreBuilder::new(&path, 4096, 3).unwrap();
    builder.add(&make_doc("DOC-EMPTY", b"")).unwrap();
    builder.build().unwrap();

    let store = DocumentStore::load(&path, ContentAccess::Memory).unwrap();
    let docs = store.get_by_number(&[0]).unwrap();
    assert_eq!(docs[0].content, b"");
    assert_eq!(docs[0].keys["docno"], "DOC-EMPTY");
}

#[test]
fn test_large_document() {
    init_logger();
    let dir = TempDir::new().unwrap();
    let path = dir.path().join("store");

    let large_content = vec![42u8; 10_000];
    let mut builder = DocumentStoreBuilder::new(&path, 4096, 3).unwrap();
    builder.add(&make_doc("DOC-LARGE", &large_content)).unwrap();
    builder.add(&make_doc("DOC-SMALL", b"tiny")).unwrap();
    builder.build().unwrap();

    let store = DocumentStore::load(&path, ContentAccess::Memory).unwrap();
    let docs = store.get_by_number(&[0]).unwrap();
    assert_eq!(docs[0].content, large_content);
    let docs = store.get_by_number(&[1]).unwrap();
    assert_eq!(docs[0].content, b"tiny");
}

#[test]
fn test_single_document() {
    init_logger();
    let dir = TempDir::new().unwrap();
    let path = dir.path().join("store");

    let mut builder = DocumentStoreBuilder::new(&path, 4096, 3).unwrap();
    builder.add(&make_doc("ONLY-ONE", b"solo")).unwrap();
    builder.build().unwrap();

    let store = DocumentStore::load(&path, ContentAccess::Memory).unwrap();
    assert_eq!(store.num_documents(), 1);
    let docs = store.get_by_number(&[0]).unwrap();
    assert_eq!(docs[0].keys["docno"], "ONLY-ONE");
    assert_eq!(docs[0].content, b"solo");
}

#[test]
fn test_multiple_keys() {
    init_logger();
    let dir = TempDir::new().unwrap();
    let path = dir.path().join("store");

    let mut builder = DocumentStoreBuilder::new(&path, 4096, 3).unwrap();
    builder
        .add(&make_doc_multi_keys("DOC-A", "http://a.com", b"content a"))
        .unwrap();
    builder
        .add(&make_doc_multi_keys("DOC-B", "http://b.com", b"content b"))
        .unwrap();
    builder.build().unwrap();

    let store = DocumentStore::load(&path, ContentAccess::Memory).unwrap();

    // Lookup by docno
    let results = store.get_by_key("docno", &["DOC-B"]).unwrap();
    assert_eq!(results[0].as_ref().unwrap().content, b"content b");

    // Lookup by url
    let results = store.get_by_key("url", &["http://a.com"]).unwrap();
    assert_eq!(results[0].as_ref().unwrap().keys["docno"], "DOC-A");
}

#[test]
fn test_duplicate_key_error() {
    init_logger();
    let dir = TempDir::new().unwrap();
    let path = dir.path().join("store");

    let mut builder = DocumentStoreBuilder::new(&path, 4096, 3).unwrap();
    builder.add(&make_doc("SAME-KEY", b"first")).unwrap();
    builder.add(&make_doc("SAME-KEY", b"second")).unwrap();

    let result = builder.build();
    assert!(result.is_err());
    let err_msg = result.unwrap_err().to_string();
    assert!(
        err_msg.contains("Duplicate"),
        "Expected duplicate error, got: {}",
        err_msg
    );
}

#[test]
fn test_out_of_range() {
    init_logger();
    let dir = TempDir::new().unwrap();
    let path = dir.path().join("store");

    let mut builder = DocumentStoreBuilder::new(&path, 4096, 3).unwrap();
    builder.add(&make_doc("DOC-0", b"data")).unwrap();
    builder.build().unwrap();

    let store = DocumentStore::load(&path, ContentAccess::Memory).unwrap();
    let result = store.get_by_number(&[999]);
    assert!(result.is_err());
}

#[test]
fn test_mmap_access() {
    init_logger();
    let dir = TempDir::new().unwrap();
    let path = dir.path().join("store");

    let mut builder = DocumentStoreBuilder::new(&path, 4096, 3).unwrap();
    for i in 0..10u64 {
        builder
            .add(&make_doc(
                &format!("DOC-{}", i),
                format!("c{}", i).as_bytes(),
            ))
            .unwrap();
    }
    builder.build().unwrap();

    let store = DocumentStore::load(&path, ContentAccess::Mmap).unwrap();
    let docs = store.get_by_number(&[3]).unwrap();
    assert_eq!(docs[0].keys["docno"], "DOC-3");
}

#[test]
fn test_checkpoint_clean_resume() {
    // Write 30 docs, drop the builder mid-run, then resume and finish.
    init_logger();
    let dir = TempDir::new().unwrap();
    let path = dir.path().join("store");

    let opts = BuilderOptions {
        block_size: 64,
        zstd_level: 3,
        checkpoint_frequency: Some(5),
    };

    {
        let mut builder = DocumentStoreBuilder::new_with_options(&path, &opts).unwrap();
        let mut auto_checkpoints = 0;
        for i in 0..30u64 {
            let did_checkpoint = builder
                .add(&make_doc(
                    &format!("DOC-{:04}", i),
                    format!("content-{}", i).as_bytes(),
                ))
                .unwrap();
            if did_checkpoint {
                auto_checkpoints += 1;
            }
        }
        // Every 5 docs => 6 automatic checkpoints over 30 adds.
        assert_eq!(auto_checkpoints, 6);
        // Force one more checkpoint so state after the last add is durable.
        builder.checkpoint().unwrap();
        // Drop without building -> simulates a crash after a clean checkpoint.
    }

    assert!(path.join(CHECKPOINT_FILE).exists());

    let mut builder = DocumentStoreBuilder::new_with_options(&path, &opts).unwrap();
    assert_eq!(builder.num_documents(), 30);
    for i in 30..50u64 {
        builder
            .add(&make_doc(
                &format!("DOC-{:04}", i),
                format!("content-{}", i).as_bytes(),
            ))
            .unwrap();
    }
    builder.build().unwrap();

    // Checkpoint artifacts must be gone after a successful build.
    assert!(!path.join(CHECKPOINT_FILE).exists());

    let store = DocumentStore::load(&path, ContentAccess::Memory).unwrap();
    assert_eq!(store.num_documents(), 50);
    for i in 0..50u64 {
        let docs = store.get_by_number(&[i]).unwrap();
        assert_eq!(docs[0].keys["docno"], format!("DOC-{:04}", i));
        assert_eq!(docs[0].content, format!("content-{}", i).as_bytes());
    }

    // Key lookup still works across the resume boundary.
    let results = store
        .get_by_key("docno", &["DOC-0003", "DOC-0027", "DOC-0042"])
        .unwrap();
    assert_eq!(results.len(), 3);
    for r in &results {
        assert!(r.is_some());
    }
}

#[test]
fn test_checkpoint_crash_after_adds_without_checkpoint() {
    // Adds made after the last checkpoint must be dropped on resume so the
    // on-disk state (content.dat, blocks.dat, key temp files) stays consistent.
    init_logger();
    let dir = TempDir::new().unwrap();
    let path = dir.path().join("store");

    let opts = BuilderOptions {
        block_size: 64,
        zstd_level: 3,
        checkpoint_frequency: Some(10),
    };

    {
        let mut builder = DocumentStoreBuilder::new_with_options(&path, &opts).unwrap();
        // First 10 trigger a checkpoint at doc 10.
        for i in 0..10u64 {
            builder
                .add(&make_doc(
                    &format!("DOC-{:04}", i),
                    format!("c{}", i).as_bytes(),
                ))
                .unwrap();
        }
        // Add 3 more without another checkpoint, then drop: these should be lost.
        for i in 10..13u64 {
            builder
                .add(&make_doc(
                    &format!("DOC-{:04}", i),
                    format!("c{}", i).as_bytes(),
                ))
                .unwrap();
        }
    }

    let mut builder = DocumentStoreBuilder::new_with_options(&path, &opts).unwrap();
    // Only the 10 checkpointed docs survived.
    assert_eq!(builder.num_documents(), 10);

    // Re-add docs 10..13 (under the same key values — no duplicate error proves
    // the temp key files were truncated on resume).
    for i in 10..13u64 {
        builder
            .add(&make_doc(
                &format!("DOC-{:04}", i),
                format!("c{}", i).as_bytes(),
            ))
            .unwrap();
    }
    builder.build().unwrap();

    let store = DocumentStore::load(&path, ContentAccess::Memory).unwrap();
    assert_eq!(store.num_documents(), 13);
    for i in 0..13u64 {
        let docs = store.get_by_number(&[i]).unwrap();
        assert_eq!(docs[0].keys["docno"], format!("DOC-{:04}", i));
        assert_eq!(docs[0].content, format!("c{}", i).as_bytes());
    }
}

#[test]
fn test_checkpoint_manual_mode() {
    // checkpoint_frequency = None: recovery enabled, no automatic checkpoints.
    init_logger();
    let dir = TempDir::new().unwrap();
    let path = dir.path().join("store");

    let opts = BuilderOptions {
        block_size: 64,
        zstd_level: 3,
        checkpoint_frequency: None,
    };

    {
        let mut builder = DocumentStoreBuilder::new_with_options(&path, &opts).unwrap();
        for i in 0..15u64 {
            let did = builder
                .add(&make_doc(
                    &format!("DOC-{:04}", i),
                    format!("c{}", i).as_bytes(),
                ))
                .unwrap();
            // Manual mode: add() never triggers a checkpoint on its own.
            assert!(!did);
        }
        // Without an explicit checkpoint(), there is no checkpoint file yet,
        // so nothing to recover from on a crash here.
        assert!(!path.join(CHECKPOINT_FILE).exists());

        builder.checkpoint().unwrap();
        assert!(path.join(CHECKPOINT_FILE).exists());

        // More adds without another manual checkpoint -> will be lost.
        for i in 15..18u64 {
            builder
                .add(&make_doc(
                    &format!("DOC-{:04}", i),
                    format!("c{}", i).as_bytes(),
                ))
                .unwrap();
        }
        // Drop -> simulates crash.
    }

    let mut builder = DocumentStoreBuilder::new_with_options(&path, &opts).unwrap();
    assert_eq!(builder.num_documents(), 15);
    for i in 15..18u64 {
        builder
            .add(&make_doc(
                &format!("DOC-{:04}", i),
                format!("c{}", i).as_bytes(),
            ))
            .unwrap();
    }
    builder.build().unwrap();

    let store = DocumentStore::load(&path, ContentAccess::Memory).unwrap();
    assert_eq!(store.num_documents(), 18);
}

#[test]
fn test_checkpoint_disabled_starts_fresh() {
    // With checkpoint_frequency = 0, a pre-existing checkpoint file is ignored
    // and the build starts from scratch (truncating output files).
    init_logger();
    let dir = TempDir::new().unwrap();
    let path = dir.path().join("store");

    let opts = BuilderOptions {
        block_size: 64,
        zstd_level: 3,
        checkpoint_frequency: Some(2),
    };
    {
        let mut builder = DocumentStoreBuilder::new_with_options(&path, &opts).unwrap();
        for i in 0..4u64 {
            builder
                .add(&make_doc(&format!("OLD-{}", i), b"old"))
                .unwrap();
        }
        // Drop with an active checkpoint on disk.
    }
    assert!(path.join(CHECKPOINT_FILE).exists());

    // Fresh start via the non-checkpoint constructor.
    let mut builder = DocumentStoreBuilder::new(&path, 4096, 3).unwrap();
    builder.add(&make_doc("NEW-0", b"new")).unwrap();
    builder.build().unwrap();

    let store = DocumentStore::load(&path, ContentAccess::Memory).unwrap();
    assert_eq!(store.num_documents(), 1);
    let docs = store.get_by_number(&[0]).unwrap();
    assert_eq!(docs[0].keys["docno"], "NEW-0");
}

#[test]
fn test_disk_access() {
    init_logger();
    let dir = TempDir::new().unwrap();
    let path = dir.path().join("store");

    let mut builder = DocumentStoreBuilder::new(&path, 4096, 3).unwrap();
    for i in 0..10u64 {
        builder
            .add(&make_doc(
                &format!("DOC-{}", i),
                format!("c{}", i).as_bytes(),
            ))
            .unwrap();
    }
    builder.build().unwrap();

    let store = DocumentStore::load(&path, ContentAccess::Disk).unwrap();
    let docs = store.get_by_number(&[7]).unwrap();
    assert_eq!(docs[0].keys["docno"], "DOC-7");
}
