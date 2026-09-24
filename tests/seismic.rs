#![cfg(feature = "seismic")]

use std::collections::{HashMap, HashSet};

use helpers::index::TestIndex;
use impact_index::base::{load_index, DocId, ImpactValue, TermIndex};
use impact_index::builder::{BuilderOptions, Indexer};
use impact_index::index::SparseIndex;
use impact_index::search::maxscore::{search_maxscore, MaxScoreOptions};
use impact_index::search::ScoredDocument;
use impact_index::seismic::{SeismicConfig, SeismicSearchParams, SeismicSearcher};
use ndarray::Array1;
use rand::{rngs::StdRng, Rng, SeedableRng};
use temp_dir::TempDir;

fn options() -> BuilderOptions {
    BuilderOptions {
        checkpoint_frequency: 0,
        in_memory_threshold: 10,
        checkpoint_flush_ratio: 0.5,
        positions: false,
    }
}

/// Keeps every posting, uses full summaries and never skips a block: the
/// search is then exhaustive, so it must match exact MaxScore up to the
/// f16 rounding of stored values.
fn exhaustive_config() -> SeismicConfig {
    SeismicConfig {
        n_postings: 1_000_000,
        max_fraction: 1_000.,
        summary_energy: 1.0,
        ..SeismicConfig::default()
    }
}

fn exhaustive_params(query_len: usize) -> SeismicSearchParams {
    SeismicSearchParams {
        query_cut: query_len,
        heap_factor: 0.0,
        n_knn: 0,
    }
}

fn check_scores(expected: &[ScoredDocument], observed: &[ScoredDocument]) {
    assert_eq!(expected.len(), observed.len(), "result count differs");
    for (e, o) in expected.iter().zip(observed) {
        let tol = 1e-2 * e.score.abs().max(1.);
        assert!(
            (e.score - o.score).abs() <= tol,
            "score mismatch: expected {} (doc {}), got {} (doc {})",
            e.score,
            e.docid,
            o.score,
            o.docid
        );
    }
}

fn random_query(rng: &mut StdRng, vocabulary_size: usize) -> Vec<(TermIndex, ImpactValue)> {
    let n = rng.gen_range(1..=6);
    let mut terms = HashSet::new();
    while terms.len() < n {
        terms.insert(rng.gen_range(0..vocabulary_size));
    }
    terms
        .into_iter()
        .map(|t| (t, rng.gen_range(0.1f32..2.0)))
        .collect()
}

fn compare_with_maxscore(index: &dyn SparseIndex, searcher: &SeismicSearcher, vocab: usize) {
    let mut rng = StdRng::seed_from_u64(7);
    for _ in 0..20 {
        let query = random_query(&mut rng, vocab);
        let hquery: HashMap<TermIndex, ImpactValue> = query.iter().copied().collect();
        let expected = search_maxscore(index, &hquery, 10, MaxScoreOptions::default());
        let observed = searcher.search(&query, 10, &exhaustive_params(query.len()));
        check_scores(&expected, &observed);
    }
}

#[test]
fn test_seismic_exhaustive_matches_maxscore() {
    let mut data = TestIndex::new(
        100,
        2_000,
        5.,
        10,
        Some(1),
        options(),
        &HashSet::<DocId>::new(),
    );
    let index = data.indexer.to_index(true);
    let out = data.dir.path().join("seismic");
    index
        .convert_to_seismic(&out, &exhaustive_config())
        .unwrap();

    let searcher = SeismicSearcher::load(&out).unwrap();
    assert_eq!(
        searcher.num_documents() as DocId,
        SparseIndex::max_doc_id(&index) + 1
    );
    compare_with_maxscore(&index, &searcher, data.vocabulary_size);
}

#[test]
fn test_seismic_empty_documents_keep_docids() {
    // Documents 1 and 3 have no terms: Seismic positional ids must still
    // coincide with impact-index doc ids.
    let dir = TempDir::new().unwrap();
    let mut indexer = Indexer::new(dir.path(), &options());
    for (docid, terms, values) in [
        (0, vec![0usize, 2], vec![1.0f32, 0.5]),
        (2, vec![1, 2], vec![2.0, 1.0]),
        (4, vec![2], vec![3.0]),
    ] {
        indexer
            .add(docid, &Array1::from(terms), &Array1::from(values))
            .unwrap();
    }
    indexer.build().unwrap();
    let index = indexer.to_index(true);

    let out = dir.path().join("seismic");
    index
        .convert_to_seismic(&out, &exhaustive_config())
        .unwrap();
    let searcher = SeismicSearcher::load(&out).unwrap();
    assert_eq!(searcher.num_documents(), 5);

    let results = searcher.search(&[(2, 1.0)], 10, &exhaustive_params(1));
    let docids: Vec<DocId> = results.iter().map(|r| r.docid).collect();
    assert_eq!(docids, vec![4, 2, 0]);

    // Out-of-vocabulary and non-positive terms are ignored
    let results = searcher.search(&[(1, 1.0), (99, 5.0), (0, -1.0)], 10, &exhaustive_params(3));
    assert_eq!(results.iter().map(|r| r.docid).collect::<Vec<_>>(), vec![2]);
}

#[test]
fn test_seismic_u32_components() {
    // Vocabulary larger than u16 can hold
    let vocab = 70_000;
    let mut data = TestIndex::new(
        vocab,
        300,
        20.,
        40,
        Some(3),
        options(),
        &HashSet::<DocId>::new(),
    );
    let index = data.indexer.to_index(true);
    let out = data.dir.path().join("seismic");
    index
        .convert_to_seismic(&out, &exhaustive_config())
        .unwrap();

    let manifest = std::fs::read_to_string(out.join("manifest.json")).unwrap();
    assert!(manifest.contains("components=u32"), "{}", manifest);
    let searcher = SeismicSearcher::load(&out).unwrap();

    // Query terms that do occur, to get non-empty results
    let mut rng = StdRng::seed_from_u64(11);
    let occurring: Vec<TermIndex> = data.all_terms.keys().copied().collect();
    for _ in 0..20 {
        let query: Vec<(TermIndex, ImpactValue)> = (0..3)
            .map(|_| {
                (
                    occurring[rng.gen_range(0..occurring.len())],
                    rng.gen_range(0.1f32..2.0),
                )
            })
            .collect::<HashMap<_, _>>()
            .into_iter()
            .collect();
        let hquery: HashMap<TermIndex, ImpactValue> = query.iter().copied().collect();
        let expected = search_maxscore(&index, &hquery, 10, MaxScoreOptions::default());
        let observed = searcher.search(&query, 10, &exhaustive_params(query.len()));
        check_scores(&expected, &observed);
    }
}

#[test]
#[should_panic(expected = "is a Seismic index")]
fn test_load_index_rejects_seismic_directory() {
    let mut data = TestIndex::new(
        50,
        100,
        5.,
        10,
        Some(5),
        options(),
        &HashSet::<DocId>::new(),
    );
    let out = data.dir.path().join("seismic");
    data.indexer
        .to_index(true)
        .convert_to_seismic(&out, &SeismicConfig::default())
        .unwrap();
    let _ = load_index(&out, true);
}

#[test]
fn test_seismic_rejects_other_format() {
    let mut data = TestIndex::new(
        50,
        100,
        5.,
        10,
        Some(6),
        options(),
        &HashSet::<DocId>::new(),
    );
    let out = data.dir.path().join("seismic");
    data.indexer
        .to_index(true)
        .convert_to_seismic(&out, &SeismicConfig::default())
        .unwrap();

    let path = out.join("manifest.json");
    let manifest = std::fs::read_to_string(&path).unwrap();
    std::fs::write(&path, manifest.replace("seismic@", "seismic-old@")).unwrap();
    let err = SeismicSearcher::load(&out).err().expect("should fail");
    assert!(err.to_string().contains("rebuild"), "{}", err);
}
