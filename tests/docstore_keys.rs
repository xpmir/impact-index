//! Document stores, format version 2: document number → key tables,
//! key-only stores, given document numbers (with gaps), in-memory build,
//! and reading version 1 stores.

use std::collections::HashMap;
use std::path::{Path, PathBuf};

use impact_index::docstore::{
    builder::{BuilderOptions, DocumentStoreBuilder},
    keys::KeyStoreBuilder,
    store::{ContentAccess, DocumentStore},
    DocumentData, CHECKPOINT_FILE, CONTENT_FILE, META_FILE,
};
use impact_index::storage::{IndexFiles, MemoryFiles, MemoryOutput};
use temp_dir::TempDir;

const ACCESSES: [ContentAccess; 3] = [
    ContentAccess::Memory,
    ContentAccess::Mmap,
    ContentAccess::Disk,
];

fn keys(pairs: &[(&str, &str)]) -> HashMap<String, String> {
    pairs
        .iter()
        .map(|(k, v)| (k.to_string(), v.to_string()))
        .collect()
}

/// Reads every file of `dir` into memory
fn read_dir(dir: &Path) -> MemoryFiles {
    std::fs::read_dir(dir)
        .unwrap()
        .map(|e| {
            let path = e.unwrap().path();
            let name = path.file_name().unwrap().to_str().unwrap().to_string();
            (name, std::fs::read(&path).unwrap())
        })
        .collect()
}

/// Version 1 store written by impact-index 1.7.1 (tests/data/docstore-v1):
/// 30 documents, `docno` = `DOC-{i:04}` for all, `url` for every third
fn v1_fixture() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/data/docstore-v1")
}

#[test]
fn test_reads_version_1_stores() {
    for access in ACCESSES {
        let store = DocumentStore::load(&v1_fixture(), access).unwrap();
        assert_eq!(store.num_documents(), 30);
        assert!(store.has_content());

        let docs = store.get_by_number(&[0, 7, 29]).unwrap();
        assert_eq!(docs[1].keys["docno"], "DOC-0007");
        assert_eq!(docs[1].content, b"synthetic content number 7");

        let found = store.get_by_key("docno", &["DOC-0012", "nope"]).unwrap();
        assert_eq!(found[0].as_ref().unwrap().internal_id, 12);
        assert!(found[1].is_none());

        // Document number -> key: no tables in version 1, from the blocks
        assert_eq!(
            store.get_keys("docno", &[3, 29, 30]).unwrap(),
            [Some("DOC-0003".into()), Some("DOC-0029".into()), None]
        );
        assert_eq!(
            store.get_keys("url", &[3, 4]).unwrap(),
            [Some("https://example.org/page/3".into()), None]
        );
        assert_eq!(
            store
                .get_numbers("url", &["https://example.org/page/9", "x"])
                .unwrap(),
            [Some(9), None]
        );
    }
    // Also from bytes
    let store = DocumentStore::load_from(&read_dir(&v1_fixture()), ContentAccess::Memory).unwrap();
    assert_eq!(
        store.get_keys("docno", &[5]).unwrap(),
        [Some("DOC-0005".into())]
    );
}

/// Documents at numbers 0, 2, 3, 7, ... (gaps), `docno` for all, `url`
/// for some; values may contain tabs and non-ASCII characters
fn documents() -> Vec<(u64, HashMap<String, String>, Vec<u8>)> {
    (0..200u64)
        .map(|i| {
            let docid = i * 2 - (i % 3).min(i * 2);
            let docno = if i % 7 == 0 {
                format!("doc\t{i}-é")
            } else {
                format!("D{i}")
            };
            let mut k = keys(&[("docno", &docno)]);
            if i % 4 == 0 {
                k.insert("url".into(), format!("https://example.org/{i}"));
            }
            (docid, k, format!("text of {i}").into_bytes())
        })
        .scan(None, |last, (docid, k, c)| {
            // Strictly increasing document numbers
            let docid = last.map_or(docid, |l: u64| docid.max(l + 1));
            *last = Some(docid);
            Some((docid, k, c))
        })
        .collect()
}

fn check_keys(store: &DocumentStore) {
    let docs = documents();
    let last = docs.last().unwrap().0;
    assert_eq!(store.num_documents(), last + 1);
    let mut names = store.key_names().to_vec();
    names.sort();
    assert_eq!(names, ["docno", "url"]);

    let numbers: Vec<u64> = (0..=last + 2).collect();
    let docnos = store.get_keys("docno", &numbers).unwrap();
    let urls = store.get_keys("url", &numbers).unwrap();
    let by_number: HashMap<u64, &HashMap<String, String>> =
        docs.iter().map(|(d, k, _)| (*d, k)).collect();
    for n in numbers {
        let expected = by_number.get(&n);
        assert_eq!(docnos[n as usize].as_ref(), expected.map(|k| &k["docno"]));
        assert_eq!(
            urls[n as usize].as_ref(),
            expected.and_then(|k| k.get("url"))
        );
        assert_eq!(store.contains(n), expected.is_some());
    }
    for (docid, k, _) in &docs {
        assert_eq!(
            store.get_numbers("docno", &[&k["docno"]]).unwrap(),
            [Some(*docid)]
        );
    }
    assert!(store.get_keys("title", &[0]).is_err());
}

fn build_on_disk(dir: &Path, content: bool) {
    let options = BuilderOptions::default();
    let mut builder = if content {
        DocumentStoreBuilder::new_with_options(dir, &options).unwrap()
    } else {
        DocumentStoreBuilder::new_keys_only(dir, &options).unwrap()
    };
    for (docid, keys, content) in documents() {
        builder
            .add_with_docid(docid, &DocumentData { keys, content })
            .unwrap();
    }
    builder.build().unwrap();
}

#[test]
fn test_store_with_content_and_gaps() {
    let dir = TempDir::new().unwrap();
    build_on_disk(dir.path(), true);
    for access in ACCESSES {
        let store = DocumentStore::load(dir.path(), access).unwrap();
        assert!(store.has_content());
        check_keys(&store);
        let (docid, keys, content) = &documents()[10];
        let doc = &store.get_by_number(&[*docid]).unwrap()[0];
        assert_eq!((&doc.keys, &doc.content), (keys, content));
        // A missing document number is an error
        let docids: Vec<u64> = documents().iter().map(|d| d.0).collect();
        let missing = (0..).find(|n| !docids.contains(n)).unwrap();
        assert!(missing < *docids.last().unwrap());
        assert!(store.get_by_number(&[missing]).is_err());
    }
}

#[test]
fn test_keys_only_store() {
    let dir = TempDir::new().unwrap();
    build_on_disk(dir.path(), false);
    assert!(!dir.path().join(CONTENT_FILE).exists());
    for access in ACCESSES {
        let store = DocumentStore::load(dir.path(), access).unwrap();
        assert!(!store.has_content());
        check_keys(&store);
        // Documents come back with their keys and no content
        let (docid, keys, _) = &documents()[4];
        let doc = &store.get_by_number(&[*docid]).unwrap()[0];
        assert_eq!(&doc.keys, keys);
        assert!(doc.content.is_empty());
        let found = store.get_by_key("url", &["https://example.org/4"]).unwrap();
        assert_eq!(found[0].as_ref().unwrap().internal_id, *docid);
    }
}

#[test]
fn test_in_memory_key_store() {
    let mut builder = KeyStoreBuilder::new();
    for (docid, keys, _) in documents() {
        builder.add_with_docid(docid, &keys).unwrap();
    }
    assert!(builder
        .add_with_docid(0, &keys(&[("docno", "late")]))
        .is_err());
    let out = MemoryOutput::new();
    builder.build_to(&out).unwrap();
    let files = out.into_files();
    assert!(!files.exists(CONTENT_FILE));
    let store = DocumentStore::load_from(&files, ContentAccess::Memory).unwrap();
    check_keys(&store);

    // Same key files as the disk builder (the metadata may list key names
    // in another order)
    let dir = TempDir::new().unwrap();
    build_on_disk(dir.path(), false);
    let on_disk = read_dir(dir.path());
    let mut names: Vec<&str> = files.names().filter(|n| *n != META_FILE).collect();
    names.sort();
    assert_eq!(names.len(), 6);
    for name in names {
        assert_eq!(files.get_bytes(name), on_disk.get_bytes(name), "{name}");
    }

    // Sequential numbers
    let mut builder = KeyStoreBuilder::new();
    assert_eq!(builder.add(&keys(&[("docno", "a")])).unwrap(), 0);
    assert_eq!(builder.add(&keys(&[("docno", "b")])).unwrap(), 1);
}

#[test]
fn test_keys_only_checkpoint_resume() {
    let dir = TempDir::new().unwrap();
    let options = BuilderOptions {
        checkpoint_frequency: Some(50),
        ..Default::default()
    };
    let docs = documents();
    {
        let mut builder = DocumentStoreBuilder::new_keys_only(dir.path(), &options).unwrap();
        for (docid, keys, content) in docs[..120].iter().cloned() {
            builder
                .add_with_docid(docid, &DocumentData { keys, content })
                .unwrap();
        }
        builder.checkpoint().unwrap();
    }
    assert!(dir.path().join(CHECKPOINT_FILE).exists());
    // A full store cannot resume a key-only checkpoint
    assert!(DocumentStoreBuilder::new_with_options(dir.path(), &options).is_err());

    let mut builder = DocumentStoreBuilder::new_keys_only(dir.path(), &options).unwrap();
    assert_eq!(builder.num_documents(), docs[119].0 + 1);
    for (docid, keys, content) in docs[120..].iter().cloned() {
        builder
            .add_with_docid(docid, &DocumentData { keys, content })
            .unwrap();
    }
    builder.build().unwrap();
    check_keys(&DocumentStore::load(dir.path(), ContentAccess::Memory).unwrap());
}

#[test]
fn test_rejects_newer_and_corrupt_stores() {
    let dir = TempDir::new().unwrap();
    build_on_disk(dir.path(), false);
    let files = read_dir(dir.path());

    // Newer format version: actionable error
    let mut meta: ciborium::value::Value =
        ciborium::de::from_reader(files.get_bytes(META_FILE).unwrap()).unwrap();
    for (k, v) in meta.as_map_mut().unwrap() {
        if k.as_text() == Some("version") {
            *v = 9.into();
        }
    }
    let mut bytes = Vec::new();
    ciborium::ser::into_writer(&meta, &mut bytes).unwrap();
    let mut newer = files.clone();
    newer.insert(META_FILE, bytes);
    let err = DocumentStore::load_from(&newer, ContentAccess::Memory)
        .err()
        .unwrap();
    assert!(err.to_string().contains("upgrade"), "{err}");

    // Truncated key table
    let mut truncated = files.clone();
    let idx = files.get_bytes("key_docno.idx").unwrap();
    truncated.insert("key_docno.idx", idx[..idx.len() - 8].to_vec());
    assert!(DocumentStore::load_from(&truncated, ContentAccess::Memory).is_err());
}
