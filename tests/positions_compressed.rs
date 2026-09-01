//! Integration tests for positions through the compressed index
//! (positions-plan.md, Phase 2 / part B): the pluggable position codecs,
//! the v5 compressed binary format + v4 migration, lazy decode, and
//! forwarding through `ScoredIndex`/reorder guards.
//!
//! Complements `tests/positions.rs` (forward-index positions, Part A).

use std::collections::HashMap;
use std::path::Path;
use std::sync::Arc;

use impact_index::base::{load_index, DocId, ImpactValue, Len, TermIndex};
use impact_index::bow::BOWIndexBuilder;
use impact_index::builder::{BuilderOptions, SparseBuilderIndex};
use impact_index::compress::docid::BitPackingCompressor;
use impact_index::compress::impact::Identity;
use impact_index::compress::positions::{BitPackingPositions, PositionsCompressor, VBytePositions};
use impact_index::compress::{CompressedIndexInformation, CompressionTransform};
use impact_index::docmeta::DocMetadata;
use impact_index::index::{SparseIndex, SparseIndexView};
use impact_index::manifest::{self, write_manifest_raw};
use impact_index::scoring::bm25::BM25Scoring;
use impact_index::scoring::ScoredIndex;
use impact_index::search::maxscore::{search_maxscore, MaxScoreOptions};
use impact_index::search::wand::search_wand;
use impact_index::search::ScoredDocument;
use impact_index::transforms::reorder::{BpOptions, ReorderTransform};
use impact_index::transforms::IndexTransform;
use impact_index::vocab::analyzer::TextAnalyzer;
use impact_index::vocab::stemmer::NoStemmer;

fn init_logger() {
    let _ = env_logger::builder().is_test(true).try_init();
}

/// Corpus with repeated vocabulary (to force multi-page/multi-block
/// posting lists under a small `max_block_size`), plus one document with
/// a 150-occurrence run of a single term -- long enough that its position
/// deltas alone span a full 128-delta bitpacked chunk plus a partial tail.
fn corpus() -> Vec<(DocId, String)> {
    let words = ["alpha", "beta", "gamma", "delta", "epsilon"];
    let mut docs = Vec::new();
    for i in 0..20u64 {
        let mut tokens = Vec::new();
        for j in 0..(3 + (i as usize % 4)) {
            let w = words[(i as usize + j) % words.len()];
            tokens.push(w);
            if j % 2 == 0 {
                tokens.push(w);
            }
        }
        docs.push((i, tokens.join(" ")));
    }
    docs.push((20, vec!["omega"; 150].join(" ")));
    docs
}

/// Builds a positional BOW forward index over [`corpus`].
fn build_forward_index(path: &Path, in_memory_threshold: usize) -> SparseBuilderIndex<f32> {
    std::fs::create_dir_all(path).expect("create forward index dir");
    let mut builder = BOWIndexBuilder::<f32>::with_analyzer(
        path,
        &BuilderOptions {
            positions: true,
            in_memory_threshold,
            ..Default::default()
        },
        TextAnalyzer::new(Box::new(NoStemmer)),
    );
    for (docid, text) in corpus() {
        builder.add_text(docid, &text).unwrap();
    }
    let (index, _doc_meta) = builder.build(true).expect("build failed");
    index
}

/// Compresses `index` into `out_dir` (tiny `max_block_size` by default in
/// callers, to force many blocks) and copies auxiliary files (docmeta)
/// alongside it, as `Index.compress()` does in the Python bindings.
fn compress_index(
    out_dir: &Path,
    index: &SparseBuilderIndex<f32>,
    max_block_size: usize,
    positions_codec: Option<Box<dyn PositionsCompressor>>,
) {
    let transform = CompressionTransform {
        max_block_size,
        doc_ids_compressor_factory: Box::new(BitPackingCompressor {}),
        impacts_compressor_factory: Box::new(Identity {}),
        positions_codec,
    };
    transform.process(out_dir, index).expect("compress");
    index.save_auxiliary(out_dir).expect("save auxiliary data");
}

/// Verifies `loaded` (a compressed index built from `forward`) reproduces
/// `forward`'s positions exactly:
/// - `positions_iterator` output equals the forward index's, for every term;
/// - a `block_iterator` advanced with skips (every 3rd known docid, so it
///   exercises `next_min_doc_id` actually jumping blocks) returns the
///   correct `positions()` at each stop.
fn assert_positions_round_trip(forward: &SparseBuilderIndex<f32>, loaded: &dyn SparseIndex) {
    assert!(
        SparseIndex::has_positions(loaded),
        "compressed index should report positions"
    );

    for term_ix in 0..forward.len() {
        let forward_docids: Vec<DocId> = SparseIndexView::iterator(forward, term_ix)
            .map(|p| p.docid)
            .collect();
        let expected_positions: Vec<Vec<u32>> =
            SparseIndexView::positions_iterator(forward, term_ix)
                .expect("forward index should have positions")
                .collect();
        assert_eq!(forward_docids.len(), expected_positions.len());

        let actual_positions: Vec<Vec<u32>> = SparseIndex::positions_iterator(loaded, term_ix)
            .expect("compressed index should have positions")
            .collect();
        assert_eq!(
            actual_positions, expected_positions,
            "positions_iterator mismatch for term {}",
            term_ix
        );

        let mut iter = loaded.block_iterator(term_ix);
        for (i, (&docid, positions)) in forward_docids
            .iter()
            .zip(expected_positions.iter())
            .enumerate()
        {
            if i % 3 != 0 {
                continue;
            }
            assert!(
                iter.next_min_doc_id(docid).is_some(),
                "term {} docid {} should be reachable",
                term_ix,
                docid
            );
            let current = iter.current();
            assert_eq!(current.docid, docid);
            let got = iter
                .positions()
                .unwrap_or_else(|| {
                    panic!(
                        "positions() should be Some for term {} docid {}",
                        term_ix, docid
                    )
                })
                .to_vec();
            assert_eq!(
                &got, positions,
                "block_iterator positions() mismatch for term {} docid {}",
                term_ix, docid
            );
        }
    }
}

/// (docid, score) pairs, sorted by docid, for order-independent comparison.
fn sorted_pairs(results: &[ScoredDocument]) -> Vec<(DocId, ImpactValue)> {
    let mut pairs: Vec<(DocId, ImpactValue)> = results.iter().map(|d| (d.docid, d.score)).collect();
    pairs.sort_by_key(|&(docid, _)| docid);
    pairs
}

// ---------------------------------------------------------------------
// 1. Codec unit round-trips
// ---------------------------------------------------------------------

#[test]
fn test_positions_codec_roundtrips() {
    // Deltas including 0 (a legitimate first-position value) and large
    // values (> 2^20), at sizes that exercise: a single delta, exactly one
    // full bitpacked chunk (128), one full chunk + 1 (129), and multiple
    // chunks + a partial tail (300).
    fn make_deltas(len: usize) -> Vec<u32> {
        (0..len)
            .map(|i| match i % 5 {
                0 => 0,
                1 => (1u32 << 21) + i as u32,
                _ => ((i as u32) * 37) % 5000,
            })
            .collect()
    }

    for len in [1usize, 128, 129, 300] {
        let deltas = make_deltas(len);

        let mut vbyte_buf = Vec::new();
        VBytePositions.write(&mut vbyte_buf, &deltas);
        let mut vbyte_decoded = Vec::new();
        VBytePositions.decode_into(&vbyte_buf, deltas.len(), &mut vbyte_decoded);
        assert_eq!(
            vbyte_decoded, deltas,
            "VBytePositions roundtrip, len={}",
            len
        );

        let mut bp_buf = Vec::new();
        BitPackingPositions.write(&mut bp_buf, &deltas);
        let mut bp_decoded = Vec::new();
        BitPackingPositions.decode_into(&bp_buf, deltas.len(), &mut bp_decoded);
        assert_eq!(
            bp_decoded, deltas,
            "BitPackingPositions roundtrip, len={}",
            len
        );
    }
}

// ---------------------------------------------------------------------
// 2. Compressed round-trip, default codec (BitPackingPositions)
// ---------------------------------------------------------------------

#[test]
fn test_compressed_positions_round_trip_default_codec() {
    init_logger();
    let dir = temp_dir::TempDir::new().unwrap();

    // Small max_block_size forces many blocks per term.
    let forward = build_forward_index(&dir.path().join("forward"), 4);
    assert!(SparseIndex::has_positions(&forward));

    let compressed_path = dir.path().join("compressed");
    compress_index(&compressed_path, &forward, 4, None);

    let loaded = load_index(&compressed_path, true);
    assert_positions_round_trip(&forward, &*loaded);
}

// ---------------------------------------------------------------------
// 3. Compressed round-trip, VByte codec
// ---------------------------------------------------------------------

#[test]
fn test_compressed_positions_round_trip_vbyte_codec() {
    init_logger();
    let dir = temp_dir::TempDir::new().unwrap();

    let forward = build_forward_index(&dir.path().join("forward"), 4);

    let compressed_path = dir.path().join("compressed");
    compress_index(
        &compressed_path,
        &forward,
        4,
        Some(Box::new(VBytePositions)),
    );

    let loaded = load_index(&compressed_path, true);
    assert_positions_round_trip(&forward, &*loaded);
}

// ---------------------------------------------------------------------
// 4. Lazy guarantee (cheap proxy): positions never change search behavior
// ---------------------------------------------------------------------

#[test]
fn test_positions_do_not_change_search_results() {
    init_logger();
    let dir = temp_dir::TempDir::new().unwrap();

    let forward_pos = build_forward_index(&dir.path().join("forward_pos"), 128);

    let nonpos_dir = dir.path().join("forward_nonpos");
    std::fs::create_dir_all(&nonpos_dir).expect("create forward_nonpos dir");
    let mut builder_nonpos = BOWIndexBuilder::<f32>::with_analyzer(
        &nonpos_dir,
        &BuilderOptions {
            positions: false,
            ..Default::default()
        },
        TextAnalyzer::new(Box::new(NoStemmer)),
    );
    for (docid, text) in corpus() {
        builder_nonpos.add_text(docid, &text).unwrap();
    }
    let (forward_nonpos, _doc_meta) = builder_nonpos.build(true).unwrap();

    assert_eq!(
        forward_pos.len(),
        forward_nonpos.len(),
        "same corpus/analyzer should produce the same vocabulary"
    );

    let compressed_pos_path = dir.path().join("compressed_pos");
    compress_index(&compressed_pos_path, &forward_pos, 128, None);
    let compressed_nonpos_path = dir.path().join("compressed_nonpos");
    compress_index(&compressed_nonpos_path, &forward_nonpos, 128, None);

    let index_pos = load_index(&compressed_pos_path, true);
    let index_nonpos = load_index(&compressed_nonpos_path, true);

    assert!(SparseIndex::has_positions(&*index_pos));
    assert!(!SparseIndex::has_positions(&*index_nonpos));

    let query: HashMap<TermIndex, ImpactValue> = (0..forward_pos.len().min(6) as TermIndex)
        .map(|t| (t, 1.0))
        .collect();

    let wand_pos = sorted_pairs(&search_wand(&*index_pos, &query, 10));
    let wand_nonpos = sorted_pairs(&search_wand(&*index_nonpos, &query, 10));
    assert_eq!(
        wand_pos, wand_nonpos,
        "search_wand results must be identical regardless of positions"
    );

    let ms_pos = sorted_pairs(&search_maxscore(
        &*index_pos,
        &query,
        10,
        MaxScoreOptions::default(),
    ));
    let ms_nonpos = sorted_pairs(&search_maxscore(
        &*index_nonpos,
        &query,
        10,
        MaxScoreOptions::default(),
    ));
    assert_eq!(
        ms_pos, ms_nonpos,
        "search_maxscore results must be identical regardless of positions"
    );
}

// ---------------------------------------------------------------------
// 5. Migration: v4 (pre-positions) binary -> v5, via manifest::update_index
// ---------------------------------------------------------------------

#[test]
fn test_migration_v4_binary_to_v5_via_manifest_update() {
    init_logger();
    let dir = temp_dir::TempDir::new().unwrap();

    // A NON-positional compressed index, built with the current (v5) code.
    let forward_dir = dir.path().join("forward");
    std::fs::create_dir_all(&forward_dir).expect("create forward dir");
    let mut builder = BOWIndexBuilder::<f32>::with_analyzer(
        &forward_dir,
        &BuilderOptions::default(),
        TextAnalyzer::new(Box::new(NoStemmer)),
    );
    for (docid, text) in corpus() {
        builder.add_text(docid, &text).unwrap();
    }
    let (forward, _doc_meta) = builder.build(true).unwrap();

    let compressed_path = dir.path().join("compressed");
    compress_index(&compressed_path, &forward, 128, None);

    // Rewrite index.bin in the legacy (v4) layout -- valid here since a
    // non-positional index's v5 and v4 metadata carry the same
    // information, just laid out differently.
    let bin_path = compressed_path.join("index.bin");
    let info = {
        let mut reader = std::io::BufReader::new(std::fs::File::open(&bin_path).unwrap());
        CompressedIndexInformation::read_binary(&mut reader).unwrap()
    };
    {
        let mut writer = std::io::BufWriter::new(std::fs::File::create(&bin_path).unwrap());
        info.write_binary_v4(&mut writer).unwrap();
    }

    // Stamp the manifest as a pre-positions build would have (format v2).
    let mut stale_manifest = manifest::read_manifest(&compressed_path).unwrap().unwrap();
    stale_manifest.format_version = 2;
    write_manifest_raw(&compressed_path, &stale_manifest).unwrap();

    // `load_index` must refuse with an actionable error pointing at the
    // migration entry point, not silently misread the v4 bytes as v5.
    let path_for_panic = compressed_path.clone();
    let prev_hook = std::panic::take_hook();
    std::panic::set_hook(Box::new(|_| {})); // silence the default backtrace print
    let result = std::panic::catch_unwind(move || {
        load_index(&path_for_panic, true);
    });
    std::panic::set_hook(prev_hook);
    let payload = result.expect_err("load_index should refuse a stale v4 binary");
    let msg = payload
        .downcast_ref::<String>()
        .cloned()
        .or_else(|| payload.downcast_ref::<&str>().map(|s| s.to_string()))
        .unwrap_or_default();
    assert!(
        msg.contains("Index.update"),
        "panic message should point at the migration entry point: {}",
        msg
    );

    // Migrate, then loading and searching must work.
    manifest::update_index(&compressed_path, None).expect("migration should succeed");

    let loaded = load_index(&compressed_path, true);
    assert!(!SparseIndex::has_positions(&*loaded));

    let query: HashMap<TermIndex, ImpactValue> = (0..loaded.len().min(3) as TermIndex)
        .map(|t| (t, 1.0))
        .collect();
    let results = search_wand(&*loaded, &query, 5);
    assert!(!results.is_empty(), "search should work after migration");
}

// ---------------------------------------------------------------------
// 6. ScoredIndex (BM25) forwards has_positions()/positions()
// ---------------------------------------------------------------------

#[test]
fn test_scored_index_forwards_positions() {
    init_logger();
    let dir = temp_dir::TempDir::new().unwrap();

    let forward = build_forward_index(&dir.path().join("forward"), 4);
    let compressed_path = dir.path().join("compressed");
    compress_index(&compressed_path, &forward, 4, None);

    let loaded = load_index(&compressed_path, true);
    let doc_meta = Arc::new(DocMetadata::load(&compressed_path).expect("docmeta should be copied"));
    let scored = ScoredIndex::new(Arc::new(loaded), doc_meta, Box::new(BM25Scoring::new()));

    assert!(
        SparseIndex::has_positions(&scored),
        "ScoredIndex should forward has_positions"
    );

    for term_ix in 0..forward.len() {
        let expected: Vec<Vec<u32>> = SparseIndexView::positions_iterator(&forward, term_ix)
            .expect("forward index should have positions")
            .collect();

        let mut iter = scored.block_iterator(term_ix);
        let mut actual = Vec::new();
        while iter.next_min_doc_id(0).is_some() {
            actual.push(
                iter.positions()
                    .expect("positions() should be Some through ScoredIndex")
                    .to_vec(),
            );
        }
        assert_eq!(actual, expected, "term {}", term_ix);
    }
}

// ---------------------------------------------------------------------
// 7. Reorder guard: reordering a positional index errors actionably
// ---------------------------------------------------------------------

#[test]
fn test_reorder_refuses_positional_index() {
    init_logger();
    let dir = temp_dir::TempDir::new().unwrap();
    let forward = build_forward_index(&dir.path().join("forward"), 4);

    let sink = Box::new(CompressionTransform {
        max_block_size: 4,
        doc_ids_compressor_factory: Box::new(BitPackingCompressor {}),
        impacts_compressor_factory: Box::new(Identity {}),
        positions_codec: None,
    });
    let transform = ReorderTransform {
        sink,
        options: BpOptions::default(),
    };

    let out_path = dir.path().join("reordered");
    let err = transform
        .process(&out_path, &forward)
        .expect_err("reordering a positional index should error");
    assert!(
        err.to_string()
            .contains("reordering a positional index is not yet supported"),
        "error message should be actionable: {}",
        err
    );
}
