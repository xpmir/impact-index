//! Criterion benchmark for structured/positional queries (`src/query.rs`,
//! `src/search/ops.rs`), exercising WAND and MaxScore end-to-end over
//! `Combine`/`Syn`/`Band`/`Phrase`/`Window` query trees built directly (no
//! parser), so Phase 4 optimizations (composite block bounds, two-phase
//! verification) can be measured against a stable baseline.
//!
//! Style mirrors `benches/sparse.rs`: seeded RNG (`StdRng::seed_from_u64`),
//! `harness = false`, one shared corpus. Unlike `sparse.rs` (which rebuilds
//! its raw index once per `criterion_group` target), building a 50k-doc
//! positional + compressed index is expensive enough that all 14
//! benchmarks here share a SINGLE build, done once at the top of the lone
//! `criterion_group` target ([`run_benchmarks`]).

use std::collections::BTreeMap;
use std::sync::Arc;

use criterion::{criterion_group, criterion_main, Criterion};

use impact_index::base::{load_index, Len, TermIndex};
use impact_index::builder::{BuilderOptions, Indexer};
use impact_index::compress::docid::BitPackingCompressor;
use impact_index::compress::impact::Identity;
use impact_index::compress::CompressionTransform;
use impact_index::docmeta::DocMetadata;
use impact_index::index::SparseIndex;
use impact_index::query::{search_maxscore_query, search_wand_query, QueryNode};
use impact_index::scoring::bm25::BM25Scoring;
use impact_index::scoring::ScoredIndex;
use impact_index::search::maxscore::MaxScoreOptions;
use impact_index::transforms::IndexTransform;

use rand::{rngs::StdRng, Rng, SeedableRng};
use temp_dir::TempDir;

const NUM_DOCS: u64 = 50_000;
const VOCAB: usize = 1_000;

/// Builds the shared positional corpus (Zipf-ish term distribution via
/// `(r*r*VOCAB) as usize` on `r ~ U[0,1)`, 32..96 tokens/doc), indexes it
/// with [`Indexer::add_with_positions`], compresses it (`BitPacking` doc
/// ids, `Identity` -- i.e. lossless -- impacts, see the comment at the
/// `CompressionTransform` construction below for why), and wraps it in a
/// BM25 [`ScoredIndex`].
///
/// Returns the [`TempDir`] alongside the index purely to keep the backing
/// directory alive for the caller's scope (parity with `sparse.rs`); the
/// compressed index is loaded fully in-memory (`load_index(.., true)`) so
/// nothing actually borrows from disk afterwards.
fn build_scored_index() -> (TempDir, ScoredIndex) {
    let dir = TempDir::new().expect("could not create temporary directory");
    let mut rng = StdRng::seed_from_u64(0xBEEF);

    let mut indexer: Indexer<f32> = Indexer::new(
        dir.path(),
        &BuilderOptions {
            in_memory_threshold: 128,
            checkpoint_frequency: 0,
            checkpoint_flush_ratio: 0.5,
            positions: true,
        },
    );

    let mut lengths: Vec<u32> = Vec::with_capacity(NUM_DOCS as usize);

    for doc_id in 0..NUM_DOCS {
        let num_tokens: u32 = rng.gen_range(32..96);

        // Per-term sorted position lists for this document (BTreeMap keeps
        // term iteration order deterministic regardless of hash state).
        let mut term_positions: BTreeMap<TermIndex, Vec<u32>> = BTreeMap::new();
        for p in 0..num_tokens {
            let r: f32 = rng.gen();
            let term = ((r * r * VOCAB as f32) as usize).min(VOCAB - 1);
            term_positions.entry(term).or_default().push(p);
        }

        let terms: Vec<TermIndex> = term_positions.keys().copied().collect();
        let values: Vec<f32> = term_positions.values().map(|v| v.len() as f32).collect();
        let positions: Vec<Vec<u32>> = term_positions.into_values().collect();

        indexer
            .add_with_positions(doc_id, &terms, &values, &positions)
            .expect("error while adding terms to the index");
        lengths.push(num_tokens);
    }

    indexer.build().expect("error while building the index");
    let raw_index = indexer.to_index(true);

    let compressed_path = dir.path().join("compressed");
    let transform = CompressionTransform {
        max_block_size: 128,
        doc_ids_compressor_factory: Box::new(BitPackingCompressor {}),
        // Positions decode recovers each posting's tf (run length) by
        // casting the decompressed impact value back to `u32`
        // (`CompressedIndexIterator::ensure_positions_loaded`). A lossy
        // impact compressor (`GlobalQuantizerFactory`) reconstructs
        // *approximate* values -- an original tf of `1.0` can come back as
        // `0.99998`, which truncates to `0` and silently corrupts every
        // position run boundary after it (the `debug_assert_eq!` that
        // would catch this is compiled out in release builds, which is
        // what `cargo bench` uses). Every positional test in this repo
        // (`tests/positions.rs`, `tests/positions_compressed.rs`,
        // `tests/query.rs`) uses the lossless `Identity` compressor for
        // exactly this reason -- follow suit rather than the spec's
        // `GlobalQuantizerFactory { nbits: 0 }` (which isn't used anywhere
        // in this codebase and would itself panic: `Quantizer::new`
        // computes `2 << (nbits - 1)`, underflowing at `nbits: 0`).
        impacts_compressor_factory: Box::new(Identity {}),
        positions_codec: None,
    };
    transform
        .process(&compressed_path, &raw_index)
        .expect("could not build compressed index");

    let compressed_index = load_index(&compressed_path, true);
    let doc_meta = Arc::new(DocMetadata::from_lengths(lengths));
    let scored = ScoredIndex::new(
        Arc::new(compressed_index),
        doc_meta,
        Box::new(BM25Scoring::new()),
    );

    (dir, scored)
}

/// Builds the 7 structured/positional queries from the spec, given the
/// term ids selected by [`select_terms_and_queries`]. `lo2` is selected
/// (and printed) per the df-rank spec but, as specified, unused by any
/// query below.
fn build_queries(
    hi1: TermIndex,
    hi2: TermIndex,
    mid1: TermIndex,
    mid2: TermIndex,
    mid3: TermIndex,
    lo1: TermIndex,
) -> Vec<(&'static str, QueryNode)> {
    let term = |t: TermIndex| QueryNode::Term {
        term: t,
        weight: 1.0,
    };

    vec![
        (
            "flat4",
            QueryNode::Combine {
                children: vec![
                    (1.0, term(hi1)),
                    (1.0, term(mid1)),
                    (1.0, term(mid2)),
                    (1.0, term(lo1)),
                ],
            },
        ),
        (
            "phrase_mid",
            QueryNode::Phrase {
                terms: vec![mid1, mid2],
            },
        ),
        (
            "phrase_hi",
            QueryNode::Phrase {
                terms: vec![hi1, hi2],
            },
        ),
        (
            "window8",
            QueryNode::Window {
                terms: vec![mid1, mid3],
                width: 8,
            },
        ),
        (
            "band3",
            QueryNode::Band {
                children: vec![term(hi1), term(mid1), term(lo1)],
            },
        ),
        (
            "syn3",
            QueryNode::Syn {
                terms: vec![mid1, mid2, mid3],
            },
        ),
        (
            "combine_mixed",
            QueryNode::Combine {
                children: vec![
                    (1.0, term(hi1)),
                    (1.0, term(mid3)),
                    (
                        2.0,
                        QueryNode::Phrase {
                            terms: vec![mid1, mid2],
                        },
                    ),
                ],
            },
        ),
    ]
}

/// Selects `hi1,hi2,mid1,mid2,mid3,lo1,lo2` by document-frequency rank
/// (`hi` ~ rank 10, `mid` ~ rank `VOCAB/4`, `lo` ~ rank `VOCAB/2`, ranked
/// over terms sorted by `block_iterator(t).length()` descending, ties
/// broken by term id ascending for determinism) and builds the 7 queries.
///
/// If any query's WAND top-10 comes back empty, every rank is shifted by
/// +1 in lockstep (terms are shared across queries, e.g. `mid1`/`mid2`
/// feed `phrase_mid`, `flat4`, `syn3`, and `combine_mixed`, so shifting
/// them independently per-query would desync the shared corpus story) and
/// the whole batch is retried -- still fully deterministic. Returns the
/// term ids/dfs (for the caller to print) and the resulting queries.
fn select_terms_and_queries(
    scored: &ScoredIndex,
) -> (
    Vec<(&'static str, TermIndex, u64)>,
    Vec<(&'static str, QueryNode)>,
) {
    let vocab = VOCAB.min(scored.len());
    let mut term_dfs: Vec<(TermIndex, u64)> = (0..vocab)
        .map(|t| (t, scored.block_iterator(t).length() as u64))
        .collect();
    term_dfs.sort_by(|a, b| b.1.cmp(&a.1).then(a.0.cmp(&b.0)));

    const BASE_RANKS: [usize; 7] = [
        9,
        10,
        VOCAB / 4 - 1,
        VOCAB / 4,
        VOCAB / 4 + 1,
        VOCAB / 2 - 1,
        VOCAB / 2,
    ];
    const MAX_SHIFT: usize = 200;

    for shift in 0..MAX_SHIFT {
        let ranks: Vec<usize> = BASE_RANKS.iter().map(|&r| r + shift).collect();
        if *ranks.last().unwrap() >= term_dfs.len() {
            break;
        }
        let picked: Vec<(TermIndex, u64)> = ranks.iter().map(|&r| term_dfs[r]).collect();
        let [hi1, hi2, mid1, mid2, mid3, lo1, _lo2] = [
            picked[0].0,
            picked[1].0,
            picked[2].0,
            picked[3].0,
            picked[4].0,
            picked[5].0,
            picked[6].0,
        ];

        let queries = build_queries(hi1, hi2, mid1, mid2, mid3, lo1);
        let all_non_empty = queries.iter().all(|(_, q)| {
            !search_wand_query(scored, q, 10)
                .expect("query evaluation should not error")
                .is_empty()
        });

        if all_non_empty {
            let names = ["hi1", "hi2", "mid1", "mid2", "mid3", "lo1", "lo2"];
            let chosen: Vec<(&'static str, TermIndex, u64)> = names
                .iter()
                .zip(picked.iter())
                .map(|(&name, &(t, df))| (name, t, df))
                .collect();
            return (chosen, queries);
        }
    }

    panic!(
        "could not find a rank shift (< {}) giving non-empty results for every query",
        MAX_SHIFT
    );
}

fn run_benchmarks(c: &mut Criterion) {
    let (_dir, scored) = build_scored_index();

    let (chosen_terms, queries) = select_terms_and_queries(&scored);

    eprint!("structured.rs: chosen terms (name=term_id df=doc_frequency):");
    for (name, term, df) in &chosen_terms {
        eprint!(" {}={} (df={})", name, term, df);
    }
    eprintln!();

    // Defensive, named-panic re-check (spec requirement) -- guaranteed to
    // pass given `select_terms_and_queries` only returns non-empty
    // WAND results, but also exercises MaxScore's own top-10 emptiness.
    for (name, query) in &queries {
        assert!(
            !search_wand_query(&scored, query, 10).unwrap().is_empty(),
            "query '{}' returned no results for WAND",
            name
        );
        assert!(
            !search_maxscore_query(&scored, query, 10, MaxScoreOptions::default())
                .unwrap()
                .is_empty(),
            "query '{}' returned no results for MaxScore",
            name
        );
    }

    for (name, query) in &queries {
        let wand_name = format!("wand_{}", name);
        c.bench_function(&wand_name, |b| {
            b.iter(|| search_wand_query(&scored, query, 10).unwrap())
        });

        let maxscore_name = format!("maxscore_{}", name);
        c.bench_function(&maxscore_name, |b| {
            b.iter(|| {
                search_maxscore_query(&scored, query, 10, MaxScoreOptions::default()).unwrap()
            })
        });
    }
}

criterion_group! {
    name = benches;
    config = Criterion::default();
    targets = run_benchmarks
}
criterion_main!(benches);
