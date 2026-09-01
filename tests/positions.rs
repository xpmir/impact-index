//! Integration tests for token position capture and forward-index storage
//! (positions-plan.md, Phase 1 / part A).

use std::collections::HashMap;

use impact_index::base::{DocId, TermIndex};
use impact_index::bow::BOWIndexBuilder;
use impact_index::builder::{BuilderOptions, SparseBuilderIndex};
use impact_index::index::{SparseIndex, SparseIndexView};
use impact_index::manifest::read_manifest;
use impact_index::vocab::analyzer::TextAnalyzer;
use impact_index::vocab::stemmer::NoStemmer;

fn init_logger() {
    let _ = env_logger::builder().is_test(true).try_init();
}

/// Tiny synthetic corpus with repeated vocabulary and variable doc length,
/// so a small `in_memory_threshold` forces multi-page posting lists for
/// the more frequent terms.
fn corpus() -> Vec<(DocId, String)> {
    let words = ["alpha", "beta", "gamma", "delta", "epsilon"];
    let mut docs = Vec::new();
    for i in 0..20u64 {
        let mut tokens = Vec::new();
        for j in 0..(3 + (i as usize % 4)) {
            let w = words[(i as usize + j) % words.len()];
            tokens.push(w);
            if j % 2 == 0 {
                // Occasional in-doc repeats exercise multi-occurrence tf.
                tokens.push(w);
            }
        }
        docs.push((i, tokens.join(" ")));
    }
    docs
}

/// Builds a positional BOW index over `docs` and, independently, the
/// ground-truth per-term position lists (via `tokenize_and_stem_positional`,
/// which does not touch the vocabulary) aligned to ascending doc id.
fn build_positional_index(
    path: &std::path::Path,
    docs: &[(DocId, String)],
    options: &BuilderOptions,
) -> (
    SparseBuilderIndex<f32>,
    HashMap<String, Vec<Vec<u32>>>,
    HashMap<String, TermIndex>,
) {
    let mut builder = BOWIndexBuilder::<f32>::with_analyzer(
        path,
        options,
        TextAnalyzer::new(Box::new(NoStemmer)),
    );

    let mut ground_truth: HashMap<String, Vec<Vec<u32>>> = HashMap::new();
    for (docid, text) in docs {
        let per_term = builder
            .analyzer_mut()
            .unwrap()
            .tokenize_and_stem_positional(text);
        for (term, positions) in per_term {
            ground_truth.entry(term).or_default().push(positions);
        }
        builder.add_text(*docid, text).unwrap();
    }

    // Vocabulary only settles once every doc has been added.
    let mut term_to_ix: HashMap<String, TermIndex> = HashMap::new();
    for term in ground_truth.keys() {
        let ix = builder
            .analyzer_mut()
            .unwrap()
            .vocab()
            .get(term)
            .expect("term should be in the vocabulary after add_text");
        term_to_ix.insert(term.clone(), ix);
    }

    let (index, _doc_meta) = builder.build(true).expect("build failed");
    (index, ground_truth, term_to_ix)
}

/// Checks that both positional read paths (`positions_iterator` and
/// per-posting `block_iterator` + `positions()`) match the ground truth
/// exactly, posting-for-posting.
fn assert_positions_match(
    index: &SparseBuilderIndex<f32>,
    ground_truth: &HashMap<String, Vec<Vec<u32>>>,
    term_to_ix: &HashMap<String, TermIndex>,
) {
    assert!(
        SparseIndex::has_positions(index),
        "index should report positions"
    );

    for (term, expected) in ground_truth {
        let term_ix = term_to_ix[term];

        let via_positions_iterator: Vec<Vec<u32>> =
            SparseIndexView::positions_iterator(index, term_ix)
                .expect("positions_iterator should be Some for a positional index")
                .collect();
        assert_eq!(
            &via_positions_iterator, expected,
            "positions_iterator mismatch for term {:?}",
            term
        );

        let mut via_block_iterator = Vec::new();
        let mut iter = index.block_iterator(term_ix);
        while iter.next_min_doc_id(0).is_some() {
            via_block_iterator.push(
                iter.positions()
                    .expect("positions() should be Some for a positional index")
                    .to_vec(),
            );
        }
        assert_eq!(
            &via_block_iterator, expected,
            "block_iterator positions() mismatch for term {:?}",
            term
        );
    }
}

#[test]
fn test_analyze_doc_positional_stopword_gaps() {
    let mut analyzer = TextAnalyzer::with_stop_words(Box::new(NoStemmer), &["the"]);
    let positional = analyzer.analyze_doc_positional("the quick brown fox the quick");

    let mut by_term: HashMap<TermIndex, Vec<u32>> = positional.into_iter().collect();
    let quick_ix = analyzer.vocab().get("quick").unwrap();
    let brown_ix = analyzer.vocab().get("brown").unwrap();
    let fox_ix = analyzer.vocab().get("fox").unwrap();

    // Positions are pre-stopword-filtering token indices: "the" at 0 and 4
    // is dropped, leaving gaps in the surviving positions.
    assert_eq!(by_term.remove(&quick_ix).unwrap(), vec![1, 5]);
    assert_eq!(by_term.remove(&brown_ix).unwrap(), vec![2]);
    assert_eq!(by_term.remove(&fox_ix).unwrap(), vec![3]);
    assert!(
        by_term.is_empty(),
        "no other terms expected ('the' is a stopword): {:?}",
        by_term
    );

    // tf == positions.len() for the multi-occurrence term, and the
    // aggregate `analyze_doc` path still reports the same tf.
    let mut analyzer2 = TextAnalyzer::with_stop_words(Box::new(NoStemmer), &["the"]);
    let (terms, values) = analyzer2.analyze_doc("the quick brown fox the quick");
    let quick_ix2 = analyzer2.vocab().get("quick").unwrap();
    let pos_quick = terms.iter().position(|&t| t == quick_ix2).unwrap();
    assert_eq!(values[pos_quick], 2.0);
}

#[test]
fn test_forward_index_positions_round_trip() {
    init_logger();

    let dir = temp_dir::TempDir::new().unwrap();
    let docs = corpus();

    let (index, ground_truth, term_to_ix) = build_positional_index(
        dir.path(),
        &docs,
        &BuilderOptions {
            positions: true,
            in_memory_threshold: 4, // force multi-page terms
            ..Default::default()
        },
    );

    assert_positions_match(&index, &ground_truth, &term_to_ix);

    // A non-positional index built over the same corpus reports no
    // positions at all.
    let dir2 = temp_dir::TempDir::new().unwrap();
    let mut builder2 = BOWIndexBuilder::<f32>::with_analyzer(
        dir2.path(),
        &BuilderOptions {
            in_memory_threshold: 4,
            ..Default::default()
        },
        TextAnalyzer::new(Box::new(NoStemmer)),
    );
    for (docid, text) in &docs {
        builder2.add_text(*docid, text).unwrap();
    }
    let (index2, _doc_meta2) = builder2.build(true).unwrap();

    assert!(!SparseIndex::has_positions(&index2));
    assert!(SparseIndexView::positions_iterator(&index2, 0).is_none());
    let mut iter2 = index2.block_iterator(0);
    assert!(iter2.next_min_doc_id(0).is_some());
    assert!(iter2.positions().is_none());
}

#[test]
fn test_checkpointed_positions_round_trip() {
    init_logger();

    let dir = temp_dir::TempDir::new().unwrap();
    let docs = corpus();

    // Small checkpoint frequency + small in-memory threshold exercises the
    // checkpoint write/resume path (4-tuple CBOR tail) alongside multi-page
    // posting lists.
    let (index, ground_truth, term_to_ix) = build_positional_index(
        dir.path(),
        &docs,
        &BuilderOptions {
            positions: true,
            checkpoint_frequency: 5,
            in_memory_threshold: 4,
            ..Default::default()
        },
    );

    assert_positions_match(&index, &ground_truth, &term_to_ix);
}

#[test]
fn test_manifest_features_positions() {
    let dir = temp_dir::TempDir::new().unwrap();
    let path = dir.path();

    let mut builder = BOWIndexBuilder::<f32>::with_analyzer(
        path,
        &BuilderOptions {
            positions: true,
            ..Default::default()
        },
        TextAnalyzer::new(Box::new(NoStemmer)),
    );
    builder.add_text(0, "alpha beta gamma").unwrap();
    builder.build(true).expect("build failed");

    let manifest = read_manifest(path)
        .expect("manifest read should not error")
        .expect("manifest should exist");
    assert_eq!(manifest.features, vec!["positions".to_string()]);
}
