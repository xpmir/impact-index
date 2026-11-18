//! Backward-compatibility guardrail: indexes written by an older release
//! (manifest format version 3, crate 1.7.1, commit 54db65c) must keep loading
//! and return identical search results. See `tests/data/index-v3/README.md`
//! for how the fixtures were produced.

use std::collections::HashMap;
use std::path::Path;

use impact_index::base::{load_index, TermIndex};
use impact_index::search::maxscore::{search_maxscore, MaxScoreOptions};
use impact_index::search::wand::search_wand;
use impact_index::search::ScoredDocument;
use serde_json::Value;
use temp_dir::TempDir;

const FIXTURE: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/tests/data/index-v3");

const VARIANTS: [&str; 8] = [
    "forward",
    "ef_identity",
    "ef_quantizer8",
    "pfor_quantbitpacked8",
    "bitpacking_quantbitpacked8",
    "pfor_bitpackedint",
    "ef_bitpackedint",
    "split",
];

/// Copy a directory tree (loading must never touch the checked-in files).
fn copy_tree(from: &Path, to: &Path) {
    std::fs::create_dir_all(to).unwrap();
    for entry in std::fs::read_dir(from).unwrap() {
        let entry = entry.unwrap();
        let target = to.join(entry.file_name());
        if entry.file_type().unwrap().is_dir() {
            copy_tree(&entry.path(), &target);
        } else {
            std::fs::write(&target, std::fs::read(entry.path()).unwrap()).unwrap();
        }
    }
}

fn ranking(results: Vec<ScoredDocument>) -> Vec<(u64, f32)> {
    let mut v: Vec<_> = results.into_iter().map(|s| (s.docid, s.score)).collect();
    // Same canonical order as the generator: score desc, then docid.
    v.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap().then(a.0.cmp(&b.0)));
    v
}

fn expected_ranking(v: &Value) -> Vec<(u64, f32)> {
    v.as_array()
        .unwrap()
        .iter()
        .map(|p| (p[0].as_u64().unwrap(), p[1].as_f64().unwrap() as f32))
        .collect()
}

fn check(variant: &str, algo: &str, q: usize, got: &[(u64, f32)], want: &[(u64, f32)]) {
    let ctx = format!("{variant}/{algo}/query {q}");
    assert_eq!(got.len(), want.len(), "{ctx}: number of results");
    for (rank, (g, w)) in got.iter().zip(want).enumerate() {
        assert_eq!(g.0, w.0, "{ctx}: docid at rank {rank}");
        // The scores come from the same arithmetic on the same stored bytes,
        // so they are expected to be bit-identical; no tolerance.
        assert_eq!(g.1.to_bits(), w.1.to_bits(), "{ctx}: score at rank {rank}");
    }
}

#[test]
fn old_indexes_load_and_search_identically() {
    let root = Path::new(FIXTURE);
    let expected: Value =
        serde_json::from_slice(&std::fs::read(root.join("expected.json")).unwrap()).unwrap();
    let queries: Vec<HashMap<TermIndex, f32>> = expected["queries"]
        .as_array()
        .unwrap()
        .iter()
        .map(|q| {
            q.as_array()
                .unwrap()
                .iter()
                .map(|p| {
                    (
                        p[0].as_u64().unwrap() as usize,
                        p[1].as_f64().unwrap() as f32,
                    )
                })
                .collect()
        })
        .collect();
    assert_eq!(queries.len(), 10);

    let tmp = TempDir::new().unwrap();
    for variant in VARIANTS {
        let dir = tmp.path().join(variant);
        copy_tree(&root.join(variant), &dir);
        // Both in-memory and memory-mapped loading
        for in_memory in [true, false] {
            let index = load_index(&dir, in_memory);
            for (q, query) in queries.iter().enumerate() {
                let w = ranking(search_wand(&*index, query, 10));
                let want_w = expected_ranking(&expected[variant]["wand"][q]);
                check(variant, "wand", q, &w, &want_w);

                let m = ranking(search_maxscore(
                    &*index,
                    query,
                    10,
                    MaxScoreOptions::default(),
                ));
                let want_m = expected_ranking(&expected[variant]["maxscore"][q]);
                check(variant, "maxscore", q, &m, &want_m);
            }
        }
    }
}
