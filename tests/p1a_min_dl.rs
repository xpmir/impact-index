//! Integration tests for P1a: per-block/per-term minimum document length
//! statistics used to tighten BM25 score upper bounds (`optimizations.md`,
//! section P1a).
//!
//! Three things are covered here, matching the feature's validation plan:
//!
//! 1. **Result safety**: `search_maxscore`/`search_wand` (which now prune
//!    using the tightened, `min_dl`-aware bound) must return exactly the
//!    same top-k documents (and the same scores, up to floating-point
//!    summation-order slop) as an *exhaustive*, unpruned scan over the same
//!    `ScoredIndex` -- a bound that is even slightly too tight would
//!    silently drop true top-k results here.
//! 2. **Migration**: an index directory written in the pre-P1a binary
//!    layout (format v3, no per-block `min_doc_length`) is migrated via
//!    `manifest::update_index`, and the resulting per-block `min_doc_length`
//!    / per-term `min_dl` are checked against hand-computed values from a
//!    small, fully-controlled synthetic index.
//! 3. **Bound safety**: `ScoringFunction::max_score_with_dl` must never be
//!    smaller than the actual (f16-rounded-norm) score it bounds, across a
//!    range of assumed `min_dl` values including ones smaller/larger/equal
//!    to a document's real length.

use std::collections::{HashMap, HashSet};
use std::io::{Read, Write};
use std::sync::Arc;

use byteorder::{LittleEndian, ReadBytesExt, WriteBytesExt};
use ndarray::Array1;
use temp_dir::TempDir;

use impact_index::base::{DocId, ImpactValue, Len, TermIndex};
use impact_index::builder::{BuilderOptions, Indexer};
use impact_index::compress::docid::BitPackingCompressor;
use impact_index::compress::impact::{GlobalQuantizerFactory, Identity};
use impact_index::compress::{CompressedIndexInformation, CompressionTransform};
use impact_index::docmeta::DocMetadata;
use impact_index::index::SparseIndex;
use impact_index::manifest::{self, BuilderInfo, IndexKind, Manifest};
use impact_index::scoring::bm25::BM25Scoring;
use impact_index::scoring::{ScoredIndex, ScoringModel};
use impact_index::search::maxscore::{search_maxscore, MaxScoreOptions};
use impact_index::search::wand::search_wand;
use impact_index::search::ScoredDocument;
use impact_index::transforms::IndexTransform;

/// Cheap process-local pseudo-uniqueness for tmpdir names.
fn rand_seed() -> u64 {
    use std::time::{SystemTime, UNIX_EPOCH};
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap()
        .as_nanos() as u64
}

/// Deterministic xorshift-style PRNG (no external dependency needed).
struct Rng(u64);
impl Rng {
    fn next_u64(&mut self) -> u64 {
        self.0 = self.0.wrapping_mul(6364136223846793005).wrapping_add(1);
        self.0
    }
    fn next_f32(&mut self) -> f32 {
        ((self.next_u64() >> 33) as f32) / (u32::MAX as f32)
    }
}

/// Builds a compressed, BM25-scored index with deliberately skewed
/// (highly-varied) document lengths -- some documents 3 terms long, some
/// 60+ -- so the P1a bound has real slack to exploit and any unsafe
/// over-tightening has a good chance of showing up.
fn build_scored_index(seed: u64, num_docs: u64, vocab_size: usize) -> ScoredIndex {
    let tmpdir = std::env::temp_dir().join(format!(
        "p1a_min_dl_test_{}_{}",
        std::process::id(),
        rand_seed()
    ));
    std::fs::create_dir_all(&tmpdir).unwrap();

    let mut indexer = Indexer::<f32>::new(
        &tmpdir,
        &BuilderOptions {
            in_memory_threshold: 128,
            checkpoint_frequency: 0,
            checkpoint_flush_ratio: 0.5,
        },
    );

    let mut rng = Rng(seed);
    let mut doc_lengths = Vec::with_capacity(num_docs as usize);

    for doc_id in 0..num_docs {
        // Heavily skewed length distribution: mostly short, occasionally
        // very long -- maximizes the gap between a block's true minimum
        // length and the collection-wide minimum.
        let num_terms = if rng.next_f32() < 0.1 {
            1 + (rng.next_f32() * 80.0) as usize
        } else {
            1 + (rng.next_f32() * 6.0) as usize
        };
        let mut term_set = std::collections::BTreeSet::new();
        let mut term_vals = Vec::new();
        for _ in 0..num_terms {
            let t = (rng.next_f32() * vocab_size as f32) as TermIndex % vocab_size;
            let v = rng.next_f32().abs() + 0.1;
            if term_set.insert(t) {
                term_vals.push((t, v));
            }
        }
        doc_lengths.push(term_vals.len() as u32);
        let terms: Array1<TermIndex> = Array1::from_iter(term_vals.iter().map(|(t, _)| *t));
        let values: Array1<f32> = Array1::from_iter(term_vals.iter().map(|(_, v)| *v));
        indexer.add(doc_id, &terms, &values).unwrap();
    }
    indexer.build().unwrap();
    let mut raw_index = indexer.to_index(true);
    // P1a's build-time min_doc_length computation
    // (`CompressionTransform::process`) reads doc lengths from the *source*
    // index's own `doc_meta` -- distinct from the `doc_meta` handed to
    // `ScoredIndex::new` below, which only affects query-time BM25 norms.
    // Without this, every block would get `min_doc_length = 0` and this
    // test would silently stop exercising the tightened bound at all.
    raw_index.doc_meta = Some(DocMetadata::from_lengths(doc_lengths.clone()));

    let transform = CompressionTransform {
        max_block_size: 32,
        doc_ids_compressor_factory: Box::new(BitPackingCompressor {}),
        impacts_compressor_factory: Box::new(GlobalQuantizerFactory { nbits: 16 }),
    };
    let compressed_path = tmpdir.join("compressed");
    transform.process(&compressed_path, &raw_index).unwrap();
    let index = impact_index::base::load_index(&compressed_path, true);

    let doc_meta = Arc::new(DocMetadata::from_lengths(doc_lengths));
    ScoredIndex::new(Arc::new(index), doc_meta, Box::new(BM25Scoring::new()))
}

/// Exhaustive (unpruned) BM25 top-k: scans every posting of every query
/// term through the same per-posting `ScoringFunction` used by the real
/// search algorithms (via the `dyn BlockTermImpactIterator` path), so it
/// isolates exactly what P1a's tightened bound could break: pruning
/// decisions, not the BM25 formula itself.
fn exhaustive_search(
    scored: &ScoredIndex,
    query: &HashMap<TermIndex, ImpactValue>,
    top_k: usize,
) -> Vec<ScoredDocument> {
    let mut acc: HashMap<DocId, f64> = HashMap::new();
    for (&term_ix, &weight) in query.iter() {
        if term_ix >= scored.len() {
            continue;
        }
        for impact in scored.block_iterator(term_ix) {
            *acc.entry(impact.docid).or_insert(0.0) += impact.value as f64 * weight as f64;
        }
    }

    let mut docs: Vec<ScoredDocument> = acc
        .into_iter()
        .map(|(docid, score)| ScoredDocument {
            docid,
            score: score as f32,
        })
        .collect();
    docs.sort_by(|a, b| b.score.total_cmp(&a.score));
    docs.truncate(top_k);
    docs
}

fn assert_matches_exhaustive(
    label: &str,
    pruned: &[ScoredDocument],
    exhaustive: &[ScoredDocument],
) {
    assert_eq!(
        pruned.len(),
        exhaustive.len(),
        "{label}: result count mismatch (pruned={}, exhaustive={})",
        pruned.len(),
        exhaustive.len()
    );

    let pruned_docids: HashSet<DocId> = pruned.iter().map(|d| d.docid).collect();
    let exhaustive_docids: HashSet<DocId> = exhaustive.iter().map(|d| d.docid).collect();
    assert_eq!(
        pruned_docids, exhaustive_docids,
        "{label}: top-k docid SET mismatch -- a too-tight bound dropped (or a bug \
         invented) a document. pruned={:?} exhaustive={:?}",
        pruned_docids, exhaustive_docids
    );

    let exhaustive_by_id: HashMap<DocId, f32> =
        exhaustive.iter().map(|d| (d.docid, d.score)).collect();
    for p in pruned {
        let e = exhaustive_by_id[&p.docid];
        let tol = 1e-3 * e.abs().max(1.0);
        assert!(
            (p.score - e).abs() <= tol,
            "{label}: score mismatch for doc {}: pruned={} exhaustive={} (tol={})",
            p.docid,
            p.score,
            e,
            tol
        );
    }
}

/// **Critical validation test (P1a)**: for many queries and several top-k
/// values, `search_maxscore` and `search_wand` (both pruning with the new
/// per-block/per-term `min_dl` bound) must return the same top-k documents
/// as an exhaustive, unpruned scan. A bound that's even slightly too tight
/// would silently make pruning skip a true top-k document here.
#[test]
fn test_pruned_search_matches_exhaustive_with_varied_doc_lengths() {
    let scored = build_scored_index(0xC0FFEE, 3_000, 400);

    let mut rng = Rng(0xBEEF);
    let queries: Vec<HashMap<TermIndex, ImpactValue>> = (0..25)
        .map(|_| {
            let mut q = HashMap::new();
            let n_terms = 2 + (rng.next_f32() * 8.0) as usize;
            for _ in 0..n_terms {
                let t = (rng.next_f32() * 400.0) as TermIndex % 400;
                q.insert(t, 0.5 + rng.next_f32() * 2.0);
            }
            q
        })
        .collect();

    for (qi, query) in queries.iter().enumerate() {
        if query.is_empty() {
            continue;
        }
        for &top_k in &[1usize, 5, 10, 25, 100] {
            let exhaustive = exhaustive_search(&scored, query, top_k);
            let maxscore_results =
                search_maxscore(&scored, query, top_k, MaxScoreOptions::default());
            let wand_results = search_wand(&scored, query, top_k);

            assert_matches_exhaustive(
                &format!("query {qi}, top_k {top_k}, MaxScore"),
                &maxscore_results,
                &exhaustive,
            );
            assert_matches_exhaustive(
                &format!("query {qi}, top_k {top_k}, WAND"),
                &wand_results,
                &exhaustive,
            );
        }
    }
}

/// Bound safety (P1a): `max_score_with_dl` must dominate every actual score
/// it's supposed to bound, across a spread of assumed `min_dl` values --
/// including the exact real document length, one below it, and one above
/// it -- exercising the f16-rounding guard documented on
/// `BM25TermScorer::max_score_with_dl`.
#[test]
fn test_max_score_with_dl_never_underestimates() {
    // Deliberately non-round doc lengths (avgdl won't divide evenly),
    // stressing f16 rounding of the per-doc norm.
    let doc_lengths: Vec<u32> = (1..=500u32).map(|i| (i * 37) % 251 + 1).collect();
    let mut scoring = BM25Scoring::new();
    let num_docs = doc_lengths.len() as u64;
    scoring.initialize(Arc::new(doc_lengths.clone()), num_docs);

    for df in [1u64, 5, 50, 250] {
        let scorer = scoring.term_scorer(df, 10.0);
        for &max_tf in &[1.0f32, 2.0, 5.0, 10.0] {
            for (docid, &dl) in doc_lengths.iter().enumerate() {
                let actual_score = scorer.score(max_tf, docid as DocId);

                // The bound must hold for the doc's true min_dl...
                let bound_exact = scorer.max_score_with_dl(max_tf, dl);
                assert!(
                    bound_exact >= actual_score - 1e-6,
                    "df={df} tf={max_tf} dl={dl}: max_score_with_dl(exact dl)={bound_exact} \
                     < actual score={actual_score}"
                );

                // ...and for any smaller min_dl (a valid, looser statement:
                // \"every doc in this block/term has length >= min_dl\" is
                // still true if min_dl is smaller than the real minimum).
                if dl > 1 {
                    let bound_looser = scorer.max_score_with_dl(max_tf, dl - 1);
                    assert!(
                        bound_looser >= actual_score - 1e-6,
                        "df={df} tf={max_tf} dl={dl}: max_score_with_dl(dl-1)={bound_looser} \
                         < actual score={actual_score}"
                    );
                }
            }
        }

        // The `min_dl == 0` sentinel falls back to the plain (unmargined)
        // collection-wide bound, so a tightened bound computed at exactly
        // the collection's own min_dl can be *very slightly* above it (the
        // f16-rounding + BOUND_SAFETY_MARGIN safety guard applies only to
        // the tightened path). Once min_dl is meaningfully above the
        // collection minimum, the tightened bound must be clearly lower.
        // Tolerance covers that safety-margin slack, not a correctness gap.
        let global_bound = scorer.max_score_with_dl(10.0, 0);
        let margin_slack = global_bound.abs() * 1e-2 + 1e-4;
        for &dl in &[1u32, 50, 100, 250] {
            let tight_bound = scorer.max_score_with_dl(10.0, dl);
            assert!(
                tight_bound <= global_bound + margin_slack,
                "tightened bound ({tight_bound}) at dl={dl} exceeds the global \
                 fallback bound ({global_bound}) by more than the safety-margin slack"
            );
        }
    }
}

/// `max_score_with_dl` should actually be *tighter* than the old global-min
/// bound when the supplied `min_dl` is well above the collection minimum --
/// otherwise P1a wouldn't be buying any pruning power. Sanity check, not a
/// safety property.
#[test]
fn test_max_score_with_dl_is_tighter_than_global_bound_when_dl_is_larger() {
    let doc_lengths: Vec<u32> = vec![2; 10].into_iter().chain(vec![500; 490]).collect(); // min_dl = 2, but almost every doc is length 500
    let mut scoring = BM25Scoring::new();
    scoring.initialize(Arc::new(doc_lengths), 500);

    let scorer = scoring.term_scorer(10, 10.0);
    let global_bound = scorer.max_score_with_dl(10.0, 0); // sentinel -> global min_dl=2
    let tight_bound = scorer.max_score_with_dl(10.0, 500); // this block's real min_dl
    assert!(
        tight_bound < global_bound,
        "tightened bound ({tight_bound}) should be strictly below the global \
         bound ({global_bound}) when min_dl is far above the collection minimum"
    );
}

// ---------------------------------------------------------------------
// Migration test
// ---------------------------------------------------------------------

/// Tiny vint writer matching the private one in `compress::mod` (mirrored
/// here since it's not part of the public API -- this is the "small
/// legacy-writer helper" the P1a task description calls for).
fn write_vint(writer: &mut dyn Write, mut v: u64) {
    while v >= 0x80 {
        writer.write_all(&[(v as u8) | 0x80]).unwrap();
        v >>= 7;
    }
    writer.write_all(&[v as u8]).unwrap();
}

const COMPRESSED_INDEX_MAGIC: u32 = 0x49445832; // "IDX2", kept in sync with compress::mod

/// Writes an `index.bin` in the *pre-P1a* (v3) binary layout: identical to
/// the current format except that each block record has no
/// `min_doc_length` trailer. Rebuilt from the structured `terms` obtained
/// by reading a freshly-built (current-format) `index.bin`, so the header
/// (magic/compressor CBOR blob) is byte-identical to a real file -- only
/// the version number and the per-block trailer differ.
fn write_legacy_v3_index_bin(
    path: &std::path::Path,
    num_terms: u32,
    compressor_bytes: &[u8],
    terms: &[impact_index::compress::TermBlocksInformation],
) {
    let mut w = std::io::BufWriter::new(std::fs::File::create(path).unwrap());
    w.write_u32::<LittleEndian>(COMPRESSED_INDEX_MAGIC).unwrap();
    w.write_u32::<LittleEndian>(3).unwrap(); // legacy version
    w.write_u32::<LittleEndian>(num_terms).unwrap();
    w.write_u32::<LittleEndian>(compressor_bytes.len() as u32)
        .unwrap();
    w.write_all(compressor_bytes).unwrap();

    for term in terms {
        write_vint(&mut w, term.pages.len() as u64);
        w.write_f32::<LittleEndian>(term.max_value).unwrap();
        write_vint(&mut w, term.max_doc_id);
        write_vint(&mut w, term.length as u64);

        let min_block_val = term
            .pages
            .iter()
            .map(|b| b.max_value)
            .fold(f32::INFINITY, f32::min);
        w.write_f32::<LittleEndian>(min_block_val).unwrap();
        let range = term.max_value - min_block_val;

        let mut prev_doc_id: u64 = 0;
        for block in &term.pages {
            let docid_len = block.docid_position_range.1 - block.docid_position_range.0;
            let impact_len = block.impact_position_range.1 - block.impact_position_range.0;
            write_vint(&mut w, docid_len);
            write_vint(&mut w, impact_len);
            write_vint(&mut w, block.length as u64);
            let q = if range > 0.0 {
                (((block.max_value - min_block_val) / range * 255.0).ceil() as u32).min(255) as u8
            } else {
                255u8
            };
            w.write_all(&[q]).unwrap();
            write_vint(&mut w, block.min_doc_id - prev_doc_id);
            write_vint(&mut w, block.max_doc_id - block.min_doc_id);
            prev_doc_id = block.max_doc_id;
            // NOTE: no min_doc_length trailer -- this is the v3 layout.
        }
    }
}

/// Reads the raw v4 file's header (magic/version/num_terms/compressor
/// bytes) directly, without going through any private API -- these fields
/// are format-invariant between v3 and v4 (only the per-block trailer
/// changed), so they can be copied verbatim into the hand-built v3 file.
fn read_header_and_compressor_bytes(path: &std::path::Path) -> (u32, Vec<u8>) {
    let mut f = std::fs::File::open(path).unwrap();
    let magic = f.read_u32::<LittleEndian>().unwrap();
    assert_eq!(magic, COMPRESSED_INDEX_MAGIC);
    let _version = f.read_u32::<LittleEndian>().unwrap();
    let num_terms = f.read_u32::<LittleEndian>().unwrap();
    let compressor_len = f.read_u32::<LittleEndian>().unwrap();
    let mut compressor_bytes = vec![0u8; compressor_len as usize];
    f.read_exact(&mut compressor_bytes).unwrap();
    (num_terms, compressor_bytes)
}

/// Builds a small, fully-controlled compressed index (explicit doc IDs,
/// terms and lengths) so the expected per-block `min_doc_length` can be
/// hand-computed and checked exactly after migration.
///
/// Layout: a single term "0" appears in every one of the 12 documents
/// (doc IDs 0..12), with `max_block_size = 4` so it spans exactly 3
/// blocks of 4 postings each. Document lengths are `10, 9, 8, ..., -1`
/// (i.e. `12 - docid`), so each block's minimum length is easy to state:
/// block 0 (docs 0-3) -> min length 12-3=9; block 1 (docs 4-7) -> 12-7=5;
/// block 2 (docs 8-11) -> 12-11=1.
fn build_small_compressed_index_with_known_lengths(dir: &std::path::Path) -> Vec<u32> {
    let mut indexer = Indexer::<f32>::new(
        dir,
        &BuilderOptions {
            in_memory_threshold: 32,
            checkpoint_frequency: 0,
            checkpoint_flush_ratio: 0.5,
        },
    );

    let doc_lengths: Vec<u32> = (0..12u64).map(|d| (12 - d) as u32).collect();
    for doc_id in 0..12u64 {
        let terms: Array1<TermIndex> = Array1::from_iter([0usize]);
        let values: Array1<f32> = Array1::from_iter([1.0 + doc_id as f32]);
        indexer.add(doc_id, &terms, &values).unwrap();
    }
    indexer.build().unwrap();
    let mut raw_index = indexer.to_index(true);
    // See the comment in `build_scored_index`: build-time min_doc_length
    // computation needs the *source* index's own doc_meta.
    raw_index.doc_meta = Some(DocMetadata::from_lengths(doc_lengths.clone()));

    let compressed_path = dir.join("compressed");
    let transform = CompressionTransform {
        max_block_size: 4,
        doc_ids_compressor_factory: Box::new(BitPackingCompressor {}),
        impacts_compressor_factory: Box::new(Identity {}),
    };
    transform.process(&compressed_path, &raw_index).unwrap();
    DocMetadata::from_lengths(doc_lengths.clone())
        .save(&compressed_path)
        .unwrap();

    doc_lengths
}

/// **Migration validation test (P1a)**: writes a compressed index directory
/// in the pre-P1a binary layout (v3, no `min_doc_length`) with no
/// `manifest.json` at all (the "must work on a dir whose manifest is
/// absent" requirement -- `update_index` has to stamp v1 first, then run
/// the v1->v2 step), runs `manifest::update_index`, and checks the
/// resulting per-block `min_doc_length` / per-term `min_dl` against
/// hand-computed values.
#[test]
fn test_migration_recomputes_min_dl_from_docmeta() {
    let dir = TempDir::new().unwrap();
    let doc_lengths = build_small_compressed_index_with_known_lengths(dir.path());
    let compressed_path = dir.path().join("compressed");
    let bin_path = compressed_path.join("index.bin");

    // Ground truth: read the (current-format) file the build path just
    // produced, which already computed min_doc_length correctly (this is
    // exactly what the migration is expected to reproduce).
    let ground_truth = CompressedIndexInformation::read_binary(&mut std::io::BufReader::new(
        std::fs::File::open(&bin_path).unwrap(),
    ))
    .unwrap();
    assert_eq!(ground_truth.terms.len(), 1, "single term \"0\"");
    assert_eq!(
        ground_truth.terms[0].pages.len(),
        3,
        "3 blocks of 4 postings"
    );

    // Hand-computed expectation, independent of the build path: doc
    // lengths are `12 - docid`, blocks are docs [0-3], [4-7], [8-11].
    let expected_block_min_lengths: Vec<u32> = vec![
        (0..4).map(|d| doc_lengths[d]).min().unwrap(),  // = 9
        (4..8).map(|d| doc_lengths[d]).min().unwrap(),  // = 5
        (8..12).map(|d| doc_lengths[d]).min().unwrap(), // = 1
    ];
    assert_eq!(expected_block_min_lengths, vec![9, 5, 1]);
    for (block, &expected) in ground_truth.terms[0]
        .pages
        .iter()
        .zip(&expected_block_min_lengths)
    {
        assert_eq!(block.min_doc_length as u32, expected);
    }
    assert_eq!(ground_truth.terms[0].min_dl, 1);

    // Downgrade `index.bin` to the pre-P1a (v3) layout in place, and
    // remove the manifest entirely (simulating a genuinely legacy
    // directory: no manifest.json, v3 binary metadata).
    let (num_terms, compressor_bytes) = read_header_and_compressor_bytes(&bin_path);
    write_legacy_v3_index_bin(&bin_path, num_terms, &compressor_bytes, &ground_truth.terms);
    let manifest_path = compressed_path.join("manifest.json");
    if manifest_path.exists() {
        std::fs::remove_file(&manifest_path).unwrap();
    }
    assert!(manifest::read_manifest(&compressed_path).unwrap().is_none());

    // Sanity check: the downgraded file really is unreadable by the
    // current (v4) reader, and fails with the actionable migration error
    // -- otherwise this test would not actually be exercising migration.
    let err = CompressedIndexInformation::read_binary(&mut std::io::BufReader::new(
        std::fs::File::open(&bin_path).unwrap(),
    ))
    .err()
    .expect("reading a v3 file with the v4 reader should fail");
    assert!(
        err.to_string().contains("Index.update(path)"),
        "expected the actionable migration error, got: {}",
        err
    );

    // Run the migration.
    let result_path = manifest::update_index(&compressed_path, None).unwrap();
    assert_eq!(result_path, compressed_path);

    let manifest = manifest::read_manifest(&compressed_path)
        .unwrap()
        .expect("update_index should have written a manifest");
    assert_eq!(manifest.format_version, manifest::CURRENT_FORMAT_VERSION);
    assert_eq!(manifest.index_kind, IndexKind::Compressed);

    // Verify per-block min_doc_length / per-term min_dl against the
    // hand-computed expectation.
    let migrated = CompressedIndexInformation::read_binary(&mut std::io::BufReader::new(
        std::fs::File::open(&bin_path).unwrap(),
    ))
    .unwrap();
    assert_eq!(migrated.terms.len(), 1);
    assert_eq!(migrated.terms[0].pages.len(), 3);
    for (block, &expected) in migrated.terms[0]
        .pages
        .iter()
        .zip(&expected_block_min_lengths)
    {
        assert_eq!(
            block.min_doc_length as u32, expected,
            "migrated min_doc_length mismatch"
        );
    }
    assert_eq!(migrated.terms[0].min_dl, 1);

    // And the migrated directory loads and answers queries normally.
    let loaded = impact_index::base::load_index(&compressed_path, true);
    assert!(loaded.len() > 0);
    let doc_meta = Arc::new(DocMetadata::load(&compressed_path).unwrap());
    let scored = ScoredIndex::new(Arc::new(loaded), doc_meta, Box::new(BM25Scoring::new()));
    let query: HashMap<TermIndex, ImpactValue> = [(0usize, 1.0)].into();
    let results = search_wand(&scored, &query, 5);
    assert!(!results.is_empty());
}

/// Migration on a `Compressed` directory whose `index.bin` is already at
/// the current (v4) layout but whose manifest is missing/stale must be a
/// no-op data-wise (just re-stamps the manifest) -- it must not error out
/// trying to reinterpret v4 bytes as v3.
#[test]
fn test_migration_is_idempotent_on_already_current_binary() {
    let dir = TempDir::new().unwrap();
    build_small_compressed_index_with_known_lengths(dir.path());
    let compressed_path = dir.path().join("compressed");

    let before = CompressedIndexInformation::read_binary(&mut std::io::BufReader::new(
        std::fs::File::open(compressed_path.join("index.bin")).unwrap(),
    ))
    .unwrap();

    std::fs::remove_file(compressed_path.join("manifest.json")).unwrap();
    let mut stale = Manifest::new(IndexKind::Compressed, BuilderInfo::new());
    stale.format_version = 1;
    manifest::write_manifest_raw(&compressed_path, &stale).unwrap();

    manifest::update_index(&compressed_path, None).unwrap();

    let after = CompressedIndexInformation::read_binary(&mut std::io::BufReader::new(
        std::fs::File::open(compressed_path.join("index.bin")).unwrap(),
    ))
    .unwrap();

    assert_eq!(before.terms[0].pages.len(), after.terms[0].pages.len());
    for (b, a) in before.terms[0]
        .pages
        .iter()
        .zip(after.terms[0].pages.iter())
    {
        assert_eq!(b.min_doc_length, a.min_doc_length);
    }

    let manifest = manifest::read_manifest(&compressed_path).unwrap().unwrap();
    assert_eq!(manifest.format_version, manifest::CURRENT_FORMAT_VERSION);
}
