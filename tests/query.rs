//! Integration tests for the structured query language (`src/query.rs`,
//! `src/search/ops.rs`) -- positions-plan.md Phase 3.
//!
//! Test corpus pattern mirrors `tests/positions.rs`/`tests/bm25.rs`: a
//! `BOWIndexBuilder` with positions on, tiny thresholds, wrapped in
//! `ScoredIndex` with BM25 the way `tests/bm25.rs::test_compressed_index_standalone`
//! does.

use std::collections::HashMap;
use std::sync::Arc;

use impact_index::base::{load_index, DocId, ImpactValue, TermIndex};
use impact_index::bow::BOWIndexBuilder;
use impact_index::builder::{BuilderOptions, SparseBuilderIndex};
use impact_index::compress::docid::BitPackingCompressor;
use impact_index::compress::impact::Identity;
use impact_index::compress::CompressionTransform;
use impact_index::docmeta::DocMetadata;
use impact_index::index::SparseIndex;
use impact_index::query::{evaluate, parse_matchop, search_maxscore_query, search_wand_query};
use impact_index::query::{QueryError, QueryNode};
use impact_index::scoring::bm25::BM25Scoring;
use impact_index::scoring::ScoredIndex;
use impact_index::search::maxscore::{search_maxscore, MaxScoreOptions};
use impact_index::search::wand::search_wand;
use impact_index::search::{ScoredDocument, TopScoredDocuments};
use impact_index::transforms::reorder::{BpOptions, ReorderTransform};
use impact_index::transforms::IndexTransform;
use impact_index::vocab::analyzer::TextAnalyzer;
use impact_index::vocab::stemmer::NoStemmer;

fn init_logger() {
    let _ = env_logger::builder().is_test(true).try_init();
}

// =======================================================================
// Small, hand-verifiable corpus (test 1: operator semantics on a RAW index)
// =======================================================================

/// Docs chosen so every operator's expected (docid, value) pairs can be
/// hand-computed -- see the per-test comments below for the derivation.
fn small_corpus() -> Vec<(DocId, &'static str)> {
    vec![
        (0, "new york city guide"),
        (1, "new the york again"), // stopword gap: "the" breaks new/york adjacency
        (2, "new york new york today"), // two phrase occurrences
        (3, "old york city guide"), // "york" but no "new"
        (4, "alpha beta gamma"),   // band(alpha,beta) matches, value 1+1=2
        (5, "alpha gamma"),        // band(alpha,beta): missing beta
        (6, "beta gamma"),         // band(alpha,beta): missing alpha
        (7, "alpha beta extra"),   // band(alpha,beta) matches, value 1+1=2
        (8, "delta epsilon"),      // syn(delta,epsilon) value 1+1=2
        (9, "delta delta epsilon"), // syn value 2+1=3
        (10, "epsilon only"),      // syn matches on epsilon alone, value 1
        (11, "cat sat on the mat"), // stopword gap -> cat@0, mat@4 (span 4)
        (12, "cat mat"),           // cat@0, mat@1 (span 1)
        (13, "mat cat"),           // order-independence: mat@0, cat@1 (span 1)
        (14, "rock jazz rock jazz rock"), // interleaved: window count exceeds both tfs
    ]
}

/// Builds a positional BOW forward index over [`small_corpus`] (with "the"
/// as a stopword, so gapped positions exist) and a word -> TermIndex
/// lookup resolved via the analyzer's vocabulary after indexing.
fn build_small() -> (SparseBuilderIndex<f32>, HashMap<&'static str, TermIndex>) {
    let dir = temp_dir::TempDir::new().unwrap();
    let mut builder = BOWIndexBuilder::<f32>::with_analyzer(
        dir.path(),
        &BuilderOptions {
            positions: true,
            in_memory_threshold: 4,
            ..Default::default()
        },
        TextAnalyzer::with_stop_words(Box::new(NoStemmer), &["the"]),
    );
    for (docid, text) in small_corpus() {
        builder.add_text(docid, text).unwrap();
    }

    let words = [
        "new", "york", "city", "guide", "again", "old", "alpha", "beta", "gamma", "extra", "delta",
        "epsilon", "only", "cat", "sat", "on", "mat", "rock", "jazz",
    ];
    let mut term_to_ix = HashMap::new();
    for w in words {
        if let Some(ix) = builder.analyzer_mut().unwrap().vocab().get(w) {
            term_to_ix.insert(w, ix);
        }
    }

    let (index, _doc_meta) = builder.build(true).expect("build failed");
    (index, term_to_ix)
}

/// Evaluates `node` (expected to be a bare non-`Combine` root, so
/// `evaluate` returns exactly one `(1.0, cursor)` entry) and drains it into
/// sorted `(docid, value)` pairs.
fn eval_pairs(index: &dyn SparseIndex, node: &QueryNode) -> Vec<(DocId, ImpactValue)> {
    let mut cursors = evaluate(index, node).expect("evaluate should succeed");
    assert_eq!(cursors.len(), 1, "expected a single top-level entry");
    let (weight, mut cursor) = cursors.remove(0);
    assert_eq!(weight, 1.0);
    let mut out = Vec::new();
    while let Some(imp) = cursor.next() {
        out.push((imp.docid, imp.value));
    }
    out.sort_by_key(|&(d, _)| d);
    out
}

#[test]
fn test_phrase_semantics() {
    init_logger();
    let (index, terms) = build_small();

    let node = QueryNode::Phrase {
        terms: vec![terms["new"], terms["york"]],
    };
    let got = eval_pairs(&index, &node);
    // doc0: new@0, york@1 -> adjacent, 1 match.
    // doc1: new@0, york@2 -> stopword gap, no match (must NOT match).
    // doc2: new@[0,3], york@[1,4] -> two adjacent occurrences.
    // doc3: no "new" at all -> AndCursor never aligns, excluded.
    assert_eq!(got, vec![(0, 1.0), (2, 2.0)]);
}

#[test]
fn test_window_semantics_within_and_outside_width() {
    init_logger();
    let (index, terms) = build_small();
    let cat = terms["cat"];
    let mat = terms["mat"];

    // width=3: doc11's span (4) is NOT < 3 -> excluded (outside width).
    // doc12/doc13 have span 1 < 3 -> both match, regardless of which term
    // comes first in the text (order-independence).
    let narrow = QueryNode::Window {
        terms: vec![cat, mat],
        width: 3,
    };
    assert_eq!(eval_pairs(&index, &narrow), vec![(12, 1.0), (13, 1.0)]);

    // width=5: doc11's span (4) now IS < 5 -> included too.
    let wide = QueryNode::Window {
        terms: vec![cat, mat],
        width: 5,
    };
    assert_eq!(
        eval_pairs(&index, &wide),
        vec![(11, 1.0), (12, 1.0), (13, 1.0)]
    );
}

/// Regression: the minimal-window sweep counts one window per pointer
/// advance, so interleaved occurrences ("rock jazz rock jazz rock",
/// width 3) produce a count (4) EXCEEDING both children's tfs (3 and 2).
/// The cursor's `max_value()` bound must still dominate the real value --
/// with a `min`-of-children bound (the original bug) it would report 2.0
/// and WAND/MaxScore could prune the document.
#[test]
fn test_window_count_exceeds_child_tf_bound_stays_safe() {
    init_logger();
    let (index, terms) = build_small();

    let node = QueryNode::Window {
        terms: vec![terms["rock"], terms["jazz"]],
        width: 3,
    };

    // rock@[0,2,4], jazz@[1,3]: sweep visits frontiers (0,1) (2,1) (2,3)
    // (4,3), all with span 1 < 3 -> count 4.
    assert_eq!(eval_pairs(&index, &node), vec![(14, 4.0)]);

    let mut cursors = evaluate(&index, &node).expect("evaluate should succeed");
    let (_, mut cursor) = cursors.remove(0);
    let bound = cursor.max_value();
    while let Some(imp) = cursor.next() {
        assert!(
            imp.value <= bound,
            "window value {} exceeds max_value bound {} (pruning-safety violation)",
            imp.value,
            bound
        );
    }
}

#[test]
fn test_band_semantics() {
    init_logger();
    let (index, terms) = build_small();

    let node = QueryNode::Band {
        children: vec![
            QueryNode::Term {
                term: terms["alpha"],
                weight: 1.0,
            },
            QueryNode::Term {
                term: terms["beta"],
                weight: 1.0,
            },
        ],
    };
    // doc4, doc7 have both alpha and beta (value = tf_alpha + tf_beta = 2);
    // doc5 (no beta) and doc6 (no alpha) are excluded.
    assert_eq!(eval_pairs(&index, &node), vec![(4, 2.0), (7, 2.0)]);
}

#[test]
fn test_syn_semantics() {
    init_logger();
    let (index, terms) = build_small();

    let node = QueryNode::Syn {
        terms: vec![terms["delta"], terms["epsilon"]],
    };
    // doc8: tf_delta=1, tf_epsilon=1 -> 2.
    // doc9: tf_delta=2, tf_epsilon=1 -> 3.
    // doc10: only epsilon (tf=1) -> matches with value 1 (OR semantics).
    assert_eq!(
        eval_pairs(&index, &node),
        vec![(8, 2.0), (9, 3.0), (10, 1.0)]
    );
}

// =======================================================================
// Larger corpus (tests 2, 4, 5, 6): compressed BM25 ScoredIndex
// =======================================================================

const BIG_WORDS: [&str; 12] = [
    "alpha", "beta", "gamma", "delta", "epsilon", "zeta", "eta", "theta", "new", "york", "city",
    "old",
];

fn big_corpus() -> Vec<(DocId, String)> {
    let mut docs = Vec::new();
    let mut seed: u64 = 0xC0FFEE_u64;
    let mut next = || {
        seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1);
        (seed >> 33) as usize
    };
    for i in 0..40u64 {
        let n = 4 + next() % 8;
        let tokens: Vec<&str> = (0..n)
            .map(|_| BIG_WORDS[next() % BIG_WORDS.len()])
            .collect();
        docs.push((i, tokens.join(" ")));
    }
    docs
}

/// Builds a (positional or not) forward index over [`big_corpus`], plus a
/// word -> TermIndex lookup.
fn build_big(positions: bool) -> (SparseBuilderIndex<f32>, HashMap<&'static str, TermIndex>) {
    let dir = temp_dir::TempDir::new().unwrap();
    let mut builder = BOWIndexBuilder::<f32>::with_analyzer(
        dir.path(),
        &BuilderOptions {
            positions,
            in_memory_threshold: 8,
            ..Default::default()
        },
        TextAnalyzer::new(Box::new(NoStemmer)),
    );
    for (docid, text) in big_corpus() {
        builder.add_text(docid, &text).unwrap();
    }
    let mut term_to_ix = HashMap::new();
    for w in BIG_WORDS {
        if let Some(ix) = builder.analyzer_mut().unwrap().vocab().get(w) {
            term_to_ix.insert(w, ix);
        }
    }
    let (index, _doc_meta) = builder.build(true).expect("build failed");
    (index, term_to_ix)
}

/// Compresses `forward` (small `max_block_size`, so multiple blocks per
/// term) into a fresh directory and wraps it in a BM25 `ScoredIndex`.
fn compress_and_score(forward: &SparseBuilderIndex<f32>) -> ScoredIndex {
    let dir = temp_dir::TempDir::new().unwrap();
    let compressed_path = dir.path().join("compressed");
    let transform = CompressionTransform {
        max_block_size: 8,
        doc_ids_compressor_factory: Box::new(BitPackingCompressor {}),
        impacts_compressor_factory: Box::new(Identity {}),
        positions_codec: None,
    };
    transform.process(&compressed_path, forward).unwrap();
    forward.save_auxiliary(&compressed_path).unwrap();

    let loaded = load_index(&compressed_path, true);
    let doc_meta = Arc::new(DocMetadata::load(&compressed_path).unwrap());
    ScoredIndex::new(Arc::new(loaded), doc_meta, Box::new(BM25Scoring::new()))
}

/// Exhaustive (no-pruning) reference: `evaluate()`s the query, then sums
/// `weight * value` per doc over a full DAAT traversal of every cursor,
/// with no early termination whatsoever -- top_k applied only at the end.
fn exhaustive_query(
    index: &dyn SparseIndex,
    query: &QueryNode,
    top_k: usize,
) -> Vec<ScoredDocument> {
    let cursors = match evaluate(index, query) {
        Ok(c) => c,
        Err(QueryError::EmptyQuery) => return Vec::new(),
        Err(e) => panic!("evaluate failed: {:?}", e),
    };

    let mut scores: HashMap<DocId, f64> = HashMap::new();
    for (weight, mut cursor) in cursors {
        while let Some(imp) = cursor.next() {
            *scores.entry(imp.docid).or_insert(0.0) += weight as f64 * imp.value as f64;
        }
    }

    let mut top = TopScoredDocuments::new(top_k);
    for (docid, score) in scores {
        top.add(docid, score as f32);
    }
    top.into_sorted_vec()
}

fn sorted_pairs(results: &[ScoredDocument]) -> Vec<(DocId, ImpactValue)> {
    let mut pairs: Vec<(DocId, ImpactValue)> = results.iter().map(|d| (d.docid, d.score)).collect();
    pairs.sort_by_key(|&(docid, _)| docid);
    pairs
}

/// Asserts `got` (a pruned top-`top_k` result) is correct against `full`
/// (the *complete*, unbounded exhaustive ranking for the same query --
/// [`exhaustive_query`] with a large enough `top_k` that nothing is
/// dropped): the pruning-safety identity check.
///
/// This does NOT require `got` to match `full`'s prefix docid-for-docid:
/// with a small vocabulary and repeated (tf-vector, doc-length) tuples, the
/// corpus produces genuine, exact BM25 score ties, so which particular
/// doc(s) sitting exactly *at* the top_k cutoff score make the cut is a
/// legitimate ordering ambiguity (`TopScoredDocuments`'s heap and WAND/
/// MaxScore's traversal order don't need to agree on tie-breaking) -- not a
/// pruning bug. What must hold, and is checked here, is the actual
/// pruning-safety property: every doc *strictly* above the cutoff score
/// must be present with the exhaustive-computed score, and every returned
/// doc's score must be at or above the cutoff.
fn assert_topk_correct(label: &str, got: &[ScoredDocument], full: &[ScoredDocument], top_k: usize) {
    let rel_tol = |s: f32| 1e-5 * s.abs().max(1.0);
    let expected_len = top_k.min(full.len());
    assert_eq!(
        got.len(),
        expected_len,
        "{label}: result count mismatch (got {}, expected {})",
        got.len(),
        expected_len
    );
    if got.is_empty() {
        return;
    }

    // Every returned doc's score must match the exhaustive score for that
    // docid (the actual "did pruning compute the right score" check).
    let full_by_id: HashMap<DocId, f32> = full.iter().map(|d| (d.docid, d.score)).collect();
    for doc in got {
        let expected_score = *full_by_id.get(&doc.docid).unwrap_or_else(|| {
            panic!(
                "{label}: doc {} is not present in the exhaustive ranking at all",
                doc.docid
            )
        });
        let tol = rel_tol(expected_score);
        assert!(
            (doc.score - expected_score).abs() <= tol,
            "{label}: doc {} score mismatch: got {} expected {} (tol {})",
            doc.docid,
            doc.score,
            expected_score,
            tol
        );
    }

    // The cutoff = the top_k-th best score in the full ranking. Everyone
    // strictly above it must be present (missing = pruning dropped a real
    // winner); everyone returned must be at or above it (extra = pruning
    // returned a non-winner).
    let cutoff = full[expected_len - 1].score;
    let tol = rel_tol(cutoff);

    let got_ids: std::collections::HashSet<DocId> = got.iter().map(|d| d.docid).collect();
    for doc in full.iter().take(expected_len) {
        if doc.score > cutoff + tol {
            assert!(
                got_ids.contains(&doc.docid),
                "{label}: doc {} (score {}) is strictly above the top_k cutoff {} but missing \
                 from the result -- pruning dropped a real winner",
                doc.docid,
                doc.score,
                cutoff
            );
        }
    }
    for doc in got {
        assert!(
            doc.score >= cutoff - tol,
            "{label}: doc {} (score {}) is below the top_k cutoff {} -- pruning returned a \
             non-winner",
            doc.docid,
            doc.score,
            cutoff
        );
    }
}

/// ~10 structured queries mixing every operator, built from [`BIG_WORDS`]
/// term indices.
fn mixed_queries(t: &HashMap<&'static str, TermIndex>) -> Vec<QueryNode> {
    let term = |w: &str, weight: f32| QueryNode::Term { term: t[w], weight };

    vec![
        // 1. Plain weighted combine.
        QueryNode::Combine {
            children: vec![(1.0, term("alpha", 1.0)), (2.0, term("beta", 1.0))],
        },
        // 2. Band of two terms.
        QueryNode::Band {
            children: vec![term("alpha", 1.0), term("gamma", 1.0)],
        },
        // 3. Bare synonym.
        QueryNode::Syn {
            terms: vec![t["delta"], t["epsilon"]],
        },
        // 4. Combine over a nested band and a term.
        QueryNode::Combine {
            children: vec![
                (
                    1.0,
                    QueryNode::Band {
                        children: vec![term("alpha", 1.0), term("beta", 1.0)],
                    },
                ),
                (1.0, term("gamma", 1.0)),
            ],
        },
        // 5. Bare phrase.
        QueryNode::Phrase {
            terms: vec![t["new"], t["york"]],
        },
        // 6. Bare window.
        QueryNode::Window {
            terms: vec![t["new"], t["york"]],
            width: 4,
        },
        // 7. Combine of synonym and term.
        QueryNode::Combine {
            children: vec![
                (
                    1.5,
                    QueryNode::Syn {
                        terms: vec![t["zeta"], t["eta"]],
                    },
                ),
                (0.5, term("theta", 1.0)),
            ],
        },
        // 8. Band of a synonym and a term.
        QueryNode::Band {
            children: vec![
                QueryNode::Syn {
                    terms: vec![t["alpha"], t["beta"]],
                },
                term("gamma", 1.0),
            ],
        },
        // 9. Combine of a phrase and a term.
        QueryNode::Combine {
            children: vec![
                (
                    1.0,
                    QueryNode::Phrase {
                        terms: vec![t["city"], t["old"]],
                    },
                ),
                (1.0, term("new", 1.0)),
            ],
        },
        // 10. Three-term window.
        QueryNode::Window {
            terms: vec![t["alpha"], t["beta"], t["gamma"]],
            width: 6,
        },
    ]
}

#[test]
fn test_pruning_safety_identity() {
    init_logger();
    let (forward, terms) = build_big(true);
    let scored = compress_and_score(&forward);

    for (qi, query) in mixed_queries(&terms).into_iter().enumerate() {
        // The complete, unbounded ranking (nothing pruned) -- computed once
        // per query, reused as the reference for every top_k below.
        let full = exhaustive_query(&scored, &query, usize::MAX);

        for &top_k in &[1usize, 5, 20] {
            let wand = search_wand_query(&scored, &query, top_k).unwrap();
            let maxscore =
                search_maxscore_query(&scored, &query, top_k, MaxScoreOptions::default()).unwrap();

            assert_topk_correct(
                &format!("query {qi} top_k {top_k} WAND"),
                &wand,
                &full,
                top_k,
            );
            assert_topk_correct(
                &format!("query {qi} top_k {top_k} MaxScore"),
                &maxscore,
                &full,
                top_k,
            );
        }
    }
}

#[test]
fn test_positions_not_available_error() {
    init_logger();
    let (forward, terms) = build_big(false);
    assert!(!SparseIndex::has_positions(&forward));
    let scored = compress_and_score(&forward);
    assert!(!SparseIndex::has_positions(&scored));

    let phrase = QueryNode::Phrase {
        terms: vec![terms["new"], terms["york"]],
    };

    let err = match search_wand_query(&scored, &phrase, 5) {
        Err(e) => e,
        Ok(_) => panic!("should error without positions"),
    };
    assert_eq!(err, QueryError::PositionsNotAvailable);
    assert!(
        err.to_string().contains("rebuild with positions=true"),
        "error message should be actionable: {}",
        err
    );

    let err2 = match search_maxscore_query(&scored, &phrase, 5, MaxScoreOptions::default()) {
        Err(e) => e,
        Ok(_) => panic!("should error without positions"),
    };
    assert_eq!(err2, QueryError::PositionsNotAvailable);
}

#[test]
fn test_flat_query_equivalence() {
    init_logger();
    let (forward, terms) = build_big(true);
    let scored = compress_and_score(&forward);

    let query_node = QueryNode::Combine {
        children: vec![
            (
                1.0,
                QueryNode::Term {
                    term: terms["alpha"],
                    weight: 1.0,
                },
            ),
            (
                2.0,
                QueryNode::Term {
                    term: terms["beta"],
                    weight: 1.0,
                },
            ),
            (
                0.5,
                QueryNode::Term {
                    term: terms["gamma"],
                    weight: 1.0,
                },
            ),
        ],
    };
    let flat_query: HashMap<TermIndex, ImpactValue> = [
        (terms["alpha"], 1.0),
        (terms["beta"], 2.0),
        (terms["gamma"], 0.5),
    ]
    .into();

    for &top_k in &[1usize, 5, 20] {
        let wand_structured = search_wand_query(&scored, &query_node, top_k).unwrap();
        let wand_flat = search_wand(&scored, &flat_query, top_k);
        assert_results_close(&wand_structured, &wand_flat, 1e-6);

        let ms_structured =
            search_maxscore_query(&scored, &query_node, top_k, MaxScoreOptions::default()).unwrap();
        let ms_flat = search_maxscore(&scored, &flat_query, top_k, MaxScoreOptions::default());
        assert_results_close(&ms_structured, &ms_flat, 1e-6);
    }
}

fn assert_results_close(a: &[ScoredDocument], b: &[ScoredDocument], rel_tol: f32) {
    assert_eq!(a.len(), b.len(), "result count mismatch");
    for (x, y) in a.iter().zip(b.iter()) {
        assert_eq!(x.docid, y.docid, "docid mismatch");
        let tol = rel_tol * y.score.abs().max(1.0);
        assert!(
            (x.score - y.score).abs() <= tol,
            "score mismatch for doc {}: {} vs {} (tol {})",
            x.docid,
            x.score,
            y.score,
            tol
        );
    }
}

#[test]
fn test_reordered_index_structured_query() {
    init_logger();
    // Positions block reordering (Part B guard); use a NON-positional big
    // corpus instead, as the spec calls for.
    let (forward, terms) = build_big(false);

    let baseline_dir = temp_dir::TempDir::new().unwrap();
    let baseline_path = baseline_dir.path().join("baseline");
    let sink = || CompressionTransform {
        max_block_size: 8,
        doc_ids_compressor_factory: Box::new(BitPackingCompressor {}),
        impacts_compressor_factory: Box::new(Identity {}),
        positions_codec: None,
    };
    sink().process(&baseline_path, &forward).unwrap();
    forward.save_auxiliary(&baseline_path).unwrap();

    let reordered_dir = temp_dir::TempDir::new().unwrap();
    let reordered_path = reordered_dir.path().join("reordered");
    let transform = ReorderTransform {
        sink: Box::new(sink()),
        options: BpOptions::default(),
    };
    transform.process(&reordered_path, &forward).unwrap();

    let baseline_index = load_index(&baseline_path, true);
    let baseline_meta = Arc::new(DocMetadata::load(&baseline_path).unwrap());
    let baseline_scored = ScoredIndex::new(
        Arc::new(baseline_index),
        baseline_meta,
        Box::new(BM25Scoring::new()),
    );

    let reordered_index = load_index(&reordered_path, true);
    assert!(
        SparseIndex::reorder_map(&*reordered_index).is_some(),
        "reordered index should expose a reorder map"
    );
    let reordered_meta = Arc::new(DocMetadata::load(&reordered_path).unwrap());
    let reordered_scored = ScoredIndex::new(
        Arc::new(reordered_index),
        reordered_meta,
        Box::new(BM25Scoring::new()),
    );

    let query = QueryNode::Combine {
        children: vec![
            (
                1.0,
                QueryNode::Band {
                    children: vec![
                        QueryNode::Term {
                            term: terms["alpha"],
                            weight: 1.0,
                        },
                        QueryNode::Term {
                            term: terms["beta"],
                            weight: 1.0,
                        },
                    ],
                },
            ),
            (
                1.0,
                QueryNode::Term {
                    term: terms["gamma"],
                    weight: 1.0,
                },
            ),
        ],
    };

    for &top_k in &[1usize, 5, 20] {
        let base_wand = search_wand_query(&baseline_scored, &query, top_k).unwrap();
        let reord_wand = search_wand_query(&reordered_scored, &query, top_k).unwrap();
        assert_eq!(
            sorted_pairs(&base_wand)
                .iter()
                .map(|p| p.0)
                .collect::<Vec<_>>(),
            sorted_pairs(&reord_wand)
                .iter()
                .map(|p| p.0)
                .collect::<Vec<_>>(),
            "reordered WAND docid set (top_k {top_k}) should match baseline (original ids)"
        );

        let base_ms =
            search_maxscore_query(&baseline_scored, &query, top_k, MaxScoreOptions::default())
                .unwrap();
        let reord_ms =
            search_maxscore_query(&reordered_scored, &query, top_k, MaxScoreOptions::default())
                .unwrap();
        assert_eq!(
            sorted_pairs(&base_ms)
                .iter()
                .map(|p| p.0)
                .collect::<Vec<_>>(),
            sorted_pairs(&reord_ms)
                .iter()
                .map(|p| p.0)
                .collect::<Vec<_>>(),
            "reordered MaxScore docid set (top_k {top_k}) should match baseline (original ids)"
        );
    }
}

// =======================================================================
// Parser
// =======================================================================

fn resolver_map() -> HashMap<&'static str, TermIndex> {
    [("a", 0), ("b", 1), ("c", 2), ("new", 3), ("york", 4)]
        .into_iter()
        .collect()
}

fn resolve<'a>(
    map: &'a HashMap<&'static str, TermIndex>,
) -> impl Fn(&str) -> Option<TermIndex> + 'a {
    move |tok: &str| map.get(tok).copied()
}

#[test]
fn test_parse_combine_with_weights_and_nested_phrase() {
    let map = resolver_map();
    let parsed = parse_matchop("#combine:0=2:1=1(a #1(new york))", &resolve(&map)).unwrap();

    let expected = QueryNode::Combine {
        children: vec![
            (
                2.0,
                QueryNode::Term {
                    term: map["a"],
                    weight: 1.0,
                },
            ),
            (
                1.0,
                QueryNode::Phrase {
                    terms: vec![map["new"], map["york"]],
                },
            ),
        ],
    };
    assert_eq!(parsed, expected);
}

#[test]
fn test_parse_unknown_term_dropped_under_combine() {
    let map = resolver_map();
    let parsed = parse_matchop("#combine(a unknown b)", &resolve(&map)).unwrap();
    let expected = QueryNode::Combine {
        children: vec![
            (
                1.0,
                QueryNode::Term {
                    term: map["a"],
                    weight: 1.0,
                },
            ),
            (
                1.0,
                QueryNode::Term {
                    term: map["b"],
                    weight: 1.0,
                },
            ),
        ],
    };
    assert_eq!(parsed, expected);
}

#[test]
fn test_parse_unknown_term_top_level_dropped() {
    let map = resolver_map();
    // Top level: "a unknown" -> only "a" survives, collapses to the bare term.
    let parsed = parse_matchop("a unknown", &resolve(&map)).unwrap();
    assert_eq!(
        parsed,
        QueryNode::Term {
            term: map["a"],
            weight: 1.0
        }
    );
}

#[test]
fn test_parse_nested_band_syn() {
    let map = resolver_map();
    let parsed = parse_matchop("#band(#syn(a b) c)", &resolve(&map)).unwrap();
    let expected = QueryNode::Band {
        children: vec![
            QueryNode::Syn {
                terms: vec![map["a"], map["b"]],
            },
            QueryNode::Term {
                term: map["c"],
                weight: 1.0,
            },
        ],
    };
    assert_eq!(parsed, expected);
}

#[test]
fn test_parse_band_drops_whole_node_on_unresolved_term() {
    let map = resolver_map();
    // "#band(a unknown)" can never match (unknown resolves to nothing), so
    // the whole band collapses away; the surrounding combine is then empty.
    let parsed = parse_matchop("#band(a unknown)", &resolve(&map)).unwrap();
    assert_eq!(parsed, QueryNode::Combine { children: vec![] });
    assert_eq!(parsed.validate(), Err(QueryError::EmptyQuery));
}

#[test]
fn test_parse_phrase_drops_whole_node_on_unresolved_term() {
    let map = resolver_map();
    let parsed = parse_matchop("#1(new unknown)", &resolve(&map)).unwrap();
    assert_eq!(parsed, QueryNode::Combine { children: vec![] });
}

#[test]
fn test_parse_malformed_input_errors() {
    let map = resolver_map();
    // Unbalanced parens.
    assert!(matches!(
        parse_matchop("#combine(a b", &resolve(&map)),
        Err(QueryError::Parse(_))
    ));
    // Unknown operator.
    assert!(matches!(
        parse_matchop("#bogus(a b)", &resolve(&map)),
        Err(QueryError::Parse(_))
    ));
    // Stray closing paren.
    assert!(matches!(
        parse_matchop("a b)", &resolve(&map)),
        Err(QueryError::Parse(_))
    ));
    // Malformed combine weight spec.
    assert!(matches!(
        parse_matchop("#combine:oops(a b)", &resolve(&map)),
        Err(QueryError::Parse(_))
    ));
    // Window width < 2.
    assert!(matches!(
        parse_matchop("#uw1(a b)", &resolve(&map)),
        Err(QueryError::Parse(_))
    ));
}

#[test]
fn test_parse_window_width() {
    let map = resolver_map();
    let parsed = parse_matchop("#uw8(new york)", &resolve(&map)).unwrap();
    assert_eq!(
        parsed,
        QueryNode::Window {
            terms: vec![map["new"], map["york"]],
            width: 8,
        }
    );
}
