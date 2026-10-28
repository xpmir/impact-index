//! Incremental indexing (segments + tombstones + comparable quantisation).
//!
//! The main check: a randomised add / modify / delete sequence, applied to
//! a growing list of immutable segments, returns the same top-k as one index
//! built from scratch over the live documents -- exactly with a `Fixed` or
//! `Reference` quantisation range, within one quantisation step per term
//! with `PerIndex`.

use std::collections::{BTreeMap, HashMap};
use std::path::Path;

use ndarray::Array1;
use temp_dir::TempDir;

use impact_index::base::{load_index, DocId, ImpactValue, TermIndex};
use impact_index::builder::{BuilderOptions, Indexer};
use impact_index::compress::docid::PForCompressor;
use impact_index::compress::impact::{
    QuantizationRange, QuantizerCodec, QuantizerParams, RangedQuantizerFactory,
};
use impact_index::compress::{CompressionReport, CompressionTransform};
use impact_index::index::SparseIndex;
use impact_index::search::filter::{DocBitSet, Exclude, NoFilter, Only};
use impact_index::search::maxscore::{search_maxscore, search_maxscore_filtered, MaxScoreOptions};
use impact_index::search::segments::{
    Segment, SegmentAlgorithm, SegmentScoredDocument, SegmentedSearch,
};
use impact_index::search::wand::{search_wand, search_wand_filtered};
use impact_index::search::ScoredDocument;
use impact_index::transforms::reorder::{BpOptions, ReorderTransform};
use impact_index::transforms::IndexTransform;

const VOCABULARY: usize = 200;
const NBITS: u32 = 8;

/// Deterministic PRNG (same as the other integration tests).
struct Rng(u64);
impl Rng {
    fn next_u64(&mut self) -> u64 {
        self.0 = self
            .0
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        self.0 >> 11
    }
    fn next_f32(&mut self) -> f32 {
        (self.next_u64() >> 29) as f32 / (1u64 << 24) as f32
    }
    fn below(&mut self, bound: usize) -> usize {
        (self.next_u64() % bound as u64) as usize
    }
}

type Doc = Vec<(TermIndex, ImpactValue)>;

/// A random sparse vector; weights in `(0.01, max_value)`.
fn random_doc(rng: &mut Rng, max_value: f32) -> Doc {
    let n = 3 + rng.below(15);
    let mut terms = BTreeMap::new();
    for _ in 0..n {
        // Skewed term distribution, so some posting lists span many blocks
        let t = (rng.next_f32().powi(2) * VOCABULARY as f32) as TermIndex;
        terms.insert(t, 0.01 + rng.next_f32() * (max_value - 0.01));
    }
    terms.into_iter().collect()
}

fn random_query(rng: &mut Rng) -> HashMap<TermIndex, ImpactValue> {
    (0..2 + rng.below(6))
        .map(|_| (rng.below(VOCABULARY), 0.5 + rng.next_f32()))
        .collect()
}

/// Builds and compresses an index over `docs` (sorted by docid) in `dir`.
fn build(
    dir: &Path,
    docs: &BTreeMap<DocId, Doc>,
    range: QuantizationRange,
) -> (Box<dyn SparseIndex>, CompressionReport) {
    let raw_dir = dir.join("raw");
    std::fs::create_dir_all(&raw_dir).unwrap();
    let mut indexer = Indexer::<f32>::new(
        &raw_dir,
        &BuilderOptions {
            checkpoint_frequency: 0,
            in_memory_threshold: 1000,
            checkpoint_flush_ratio: 0.5,
            positions: false,
        },
    );
    for (&docid, doc) in docs {
        let terms: Array1<TermIndex> = doc.iter().map(|(t, _)| *t).collect();
        let values: Array1<f32> = doc.iter().map(|(_, v)| *v).collect();
        indexer.add(docid, &terms, &values).unwrap();
    }
    indexer.build().unwrap();
    let raw = indexer.to_index(true);

    let transform = CompressionTransform {
        max_block_size: 32,
        doc_ids_compressor_factory: Box::new(PForCompressor {}),
        impacts_compressor_factory: Box::new(RangedQuantizerFactory {
            nbits: NBITS,
            range,
            codec: QuantizerCodec::BitPacked,
        }),
        positions_codec: None,
    };
    let out = dir.join("compressed");
    let report = transform.compress(&out, &raw).unwrap();
    (load_index(&out, true), report)
}

/// Exhaustive top-k (decoded values, no pruning).
fn exhaustive(
    index: &dyn SparseIndex,
    query: &HashMap<TermIndex, ImpactValue>,
    top_k: usize,
    deleted: Option<&DocBitSet>,
) -> Vec<(DocId, ImpactValue)> {
    let mut scores: HashMap<DocId, f64> = HashMap::new();
    for (&t, &w) in query {
        if t >= index.len() {
            continue;
        }
        let mut it = index.block_iterator(t);
        while it.next_min_doc_id(0).is_some() {
            let ti = it.current();
            *scores.entry(ti.docid).or_default() += (ti.value * w) as f64;
        }
    }
    let mut v: Vec<(DocId, ImpactValue)> = scores
        .into_iter()
        .filter(|(d, _)| deleted.is_none_or(|del| !del.contains(*d)))
        .map(|(d, s)| (d, s as f32))
        .collect();
    v.sort_by(|a, b| b.1.total_cmp(&a.1));
    v.truncate(top_k);
    v
}

/// Same top-k: identical score lists (within `tol`), and identical
/// documents above the k-th score (documents tied at the k-th score may
/// legitimately differ).
fn assert_same_top_k(got: &[(DocId, ImpactValue)], expected: &[(DocId, ImpactValue)], tol: f32) {
    assert_eq!(got.len(), expected.len(), "result count");
    for (g, e) in got.iter().zip(expected) {
        assert!(
            (g.1 - e.1).abs() <= tol,
            "scores differ: {got:?} vs {expected:?}"
        );
    }
    if tol == 0. {
        if let Some(&(_, kth)) = expected.last() {
            let above = |r: &[(DocId, ImpactValue)]| {
                let mut v: Vec<DocId> = r.iter().filter(|x| x.1 > kth).map(|x| x.0).collect();
                v.sort();
                v
            };
            assert_eq!(
                above(got),
                above(expected),
                "documents above the k-th score"
            );
        }
    }
}

#[derive(Clone, Copy, Debug)]
enum Mode {
    PerIndex,
    Fixed,
    Reference,
}

/// Runs the add / modify / delete simulation and compares segmented search
/// with a from-scratch index after every round.
fn run_simulation(mode: Mode, seed: u64) {
    let tmp = TempDir::new().unwrap();
    let mut rng = Rng(seed);
    // Fixed range [0, 2]; with Reference, later documents may exceed the
    // first segment's range (clamped identically in both builds).
    let max_value = 2.;

    let mut live: BTreeMap<DocId, Doc> = BTreeMap::new();
    // Segment holding the live version of each document
    let mut location: HashMap<DocId, usize> = HashMap::new();
    let mut segments: Vec<Box<dyn SparseIndex>> = Vec::new();
    let mut tombstones: Vec<DocBitSet> = Vec::new();
    let mut reference: Option<QuantizerParams> = None;
    let mut next_docid: DocId = 0;

    for round in 0..6 {
        // Buffer of new / modified documents for the next segment
        let mut buffer: BTreeMap<DocId, Doc> = BTreeMap::new();
        let n_ops = if round == 0 { 300 } else { 40 + rng.below(80) };
        for _ in 0..n_ops {
            let op = rng.below(10);
            let existing: Vec<DocId> = live.keys().copied().collect();
            if op < 6 || existing.is_empty() {
                // Add (sparse docid space: gaps between ids)
                next_docid += 1 + rng.below(3) as u64;
                let doc = random_doc(&mut rng, max_value);
                live.insert(next_docid, doc.clone());
                buffer.insert(next_docid, doc);
            } else {
                let docid = existing[rng.below(existing.len())];
                if let Some(seg) = location.remove(&docid) {
                    tombstones[seg].insert(docid);
                }
                if op < 8 {
                    // Modify: new version goes to the buffer
                    let doc = random_doc(&mut rng, max_value * 1.2);
                    live.insert(docid, doc.clone());
                    buffer.insert(docid, doc);
                } else {
                    live.remove(&docid);
                    buffer.remove(&docid);
                }
            }
        }

        // Flush the buffer into a new segment
        if !buffer.is_empty() {
            let range = match mode {
                Mode::PerIndex => QuantizationRange::PerIndex,
                Mode::Fixed => QuantizationRange::Fixed {
                    min: 0.,
                    max: max_value,
                },
                Mode::Reference => reference
                    .map(QuantizationRange::Reference)
                    .unwrap_or(QuantizationRange::PerIndex),
            };
            let dir = tmp.path().join(format!("segment-{round}"));
            let (index, _) = build(&dir, &buffer, range);
            if reference.is_none() {
                reference = index.quantizer();
                assert!(reference.is_some());
            }
            let seg = segments.len();
            for &docid in buffer.keys() {
                location.insert(docid, seg);
            }
            segments.push(index);
            tombstones.push(DocBitSet::new());
        }

        // From-scratch index over the live documents
        let scratch_range = match mode {
            Mode::PerIndex => QuantizationRange::PerIndex,
            Mode::Fixed => QuantizationRange::Fixed {
                min: 0.,
                max: max_value,
            },
            Mode::Reference => QuantizationRange::Reference(reference.unwrap()),
        };
        let (scratch, _) = build(
            &tmp.path().join(format!("scratch-{round}")),
            &live,
            scratch_range,
        );

        // One quantisation step (of the widest range) per query term
        let step = (max_value * 1.2) / (1u32 << NBITS) as f32;

        for algorithm in [
            SegmentAlgorithm::MaxScore(MaxScoreOptions::default()),
            SegmentAlgorithm::Wand,
        ] {
            for share_threshold in [true, false] {
                let mut search = SegmentedSearch::new(
                    segments
                        .iter()
                        .zip(&tombstones)
                        .map(|(index, deleted)| Segment {
                            index: index.as_ref(),
                            filter: Exclude(deleted),
                        })
                        .collect(),
                );
                search.algorithm = algorithm;
                search.share_threshold = share_threshold;

                for _ in 0..20 {
                    let query = random_query(&mut rng);
                    let weight_sum: f32 = query.values().sum();
                    for top_k in [1, 10, 50] {
                        let expected: Vec<(DocId, ImpactValue)> = match algorithm {
                            SegmentAlgorithm::MaxScore(o) => {
                                search_maxscore(scratch.as_ref(), &query, top_k, o)
                            }
                            SegmentAlgorithm::Wand => search_wand(scratch.as_ref(), &query, top_k),
                        }
                        .iter()
                        .map(|r| (r.docid, r.score))
                        .collect();
                        let results = search.search(&query, top_k);
                        // Pruning is exact: same scores as exhaustive scoring
                        assert_same_top_k(
                            &expected,
                            &exhaustive(scratch.as_ref(), &query, top_k, None),
                            1e-5,
                        );
                        // Every hit is the live version of a document
                        for r in &results {
                            assert_eq!(location.get(&r.docid), Some(&r.segment), "{r:?}");
                        }
                        let got: Vec<(DocId, ImpactValue)> =
                            results.iter().map(|r| (r.docid, r.score)).collect();
                        let tol = match mode {
                            Mode::PerIndex => weight_sum * step * 1.01,
                            _ => 0.,
                        };
                        assert_same_top_k(&got, &expected, tol);
                    }
                }
            }
        }
    }
}

#[test]
fn segments_match_from_scratch_fixed() {
    run_simulation(Mode::Fixed, 1);
}

#[test]
fn segments_match_from_scratch_reference() {
    run_simulation(Mode::Reference, 2);
}

#[test]
fn segments_match_from_scratch_per_index() {
    run_simulation(Mode::PerIndex, 3);
}

/// Random collection used by the filter tests.
fn random_collection(rng: &mut Rng, n: u64) -> BTreeMap<DocId, Doc> {
    (0..n).map(|d| (d, random_doc(rng, 2.))).collect()
}

/// Filtered top-k == unfiltered full ranking with rejected docs removed,
/// truncated to k.
fn check_filtered(index: &dyn SparseIndex, n_docs: usize, rng: &mut Rng) {
    let deleted: DocBitSet = (0..n_docs as u64).filter(|d| d % 3 == 0).collect();
    let allowed: DocBitSet = (0..n_docs as u64).filter(|d| d % 5 != 1).collect();
    let pairs = |r: Vec<ScoredDocument>| -> Vec<(DocId, ImpactValue)> {
        r.iter().map(|r| (r.docid, r.score)).collect()
    };

    for _ in 0..30 {
        let query = random_query(rng);
        let all_ms = pairs(search_maxscore(
            index,
            &query,
            n_docs,
            MaxScoreOptions::default(),
        ));
        let all_wand = pairs(search_wand(index, &query, n_docs));
        for top_k in [1, 10, 40] {
            let expect = |all: &[(DocId, ImpactValue)], keep: &dyn Fn(DocId) -> bool| {
                all.iter()
                    .copied()
                    .filter(|r| keep(r.0))
                    .take(top_k)
                    .collect::<Vec<_>>()
            };
            let not_deleted = |d| !deleted.contains(d);
            let is_allowed = |d| allowed.contains(d);

            let o = MaxScoreOptions::default();
            assert_same_top_k(
                &pairs(search_maxscore_filtered(
                    index,
                    &query,
                    top_k,
                    o,
                    Exclude(&deleted),
                )),
                &expect(&all_ms, &not_deleted),
                0.,
            );
            assert_same_top_k(
                &pairs(search_maxscore_filtered(
                    index,
                    &query,
                    top_k,
                    o,
                    Only(&allowed),
                )),
                &expect(&all_ms, &is_allowed),
                0.,
            );
            assert_same_top_k(
                &pairs(search_wand_filtered(
                    index,
                    &query,
                    top_k,
                    Exclude(&deleted),
                )),
                &expect(&all_wand, &not_deleted),
                0.,
            );
            assert_same_top_k(
                &pairs(search_wand_filtered(index, &query, top_k, Only(&allowed))),
                &expect(&all_wand, &is_allowed),
                0.,
            );
            // NoFilter is the unfiltered search
            assert_eq!(
                pairs(search_maxscore_filtered(index, &query, top_k, o, NoFilter)),
                pairs(search_maxscore(index, &query, top_k, o)),
            );
        }
    }
}

#[test]
fn filtered_search_matches_post_filtering() {
    let tmp = TempDir::new().unwrap();
    let mut rng = Rng(10);
    let docs = random_collection(&mut rng, 400);
    let (index, _) = build(tmp.path(), &docs, QuantizationRange::PerIndex);
    check_filtered(index.as_ref(), docs.len(), &mut rng);
}

/// Filters see original doc ids on a reordered index.
#[test]
fn filtered_search_on_reordered_index() {
    let tmp = TempDir::new().unwrap();
    let mut rng = Rng(11);
    let docs = random_collection(&mut rng, 400);

    let raw_dir = tmp.path().join("raw");
    std::fs::create_dir_all(&raw_dir).unwrap();
    let mut indexer = Indexer::<f32>::new(
        &raw_dir,
        &BuilderOptions {
            checkpoint_frequency: 0,
            in_memory_threshold: 1000,
            checkpoint_flush_ratio: 0.5,
            positions: false,
        },
    );
    for (&docid, doc) in &docs {
        let terms: Array1<TermIndex> = doc.iter().map(|(t, _)| *t).collect();
        let values: Array1<f32> = doc.iter().map(|(_, v)| *v).collect();
        indexer.add(docid, &terms, &values).unwrap();
    }
    indexer.build().unwrap();
    let raw = indexer.to_index(true);
    let out = tmp.path().join("reordered");
    ReorderTransform {
        sink: Box::new(CompressionTransform {
            max_block_size: 32,
            doc_ids_compressor_factory: Box::new(PForCompressor {}),
            impacts_compressor_factory: Box::new(RangedQuantizerFactory {
                nbits: NBITS,
                range: QuantizationRange::PerIndex,
                codec: QuantizerCodec::BitPacked,
            }),
            positions_codec: None,
        }),
        options: BpOptions {
            leaf_size: 16,
            ..Default::default()
        },
    }
    .process(&out, &raw)
    .unwrap();
    let index = load_index(&out, true);
    let map = index.reorder_map().expect("reordered index");
    assert!(map.iter().enumerate().any(|(i, &d)| i as u64 != d));
    check_filtered(index.as_ref(), docs.len(), &mut rng);
}

/// Values outside a fixed range are clamped and counted; the reference
/// parameters round-trip through `quantizer()`.
#[test]
fn fixed_range_clamps_and_reports() {
    let tmp = TempDir::new().unwrap();
    let docs: BTreeMap<DocId, Doc> = [
        (0, vec![(0, 0.5), (1, 1.5)]),
        (1, vec![(0, 2.5), (2, 1.0)]),
        (2, vec![(1, 3.0)]),
    ]
    .into_iter()
    .collect();
    let range = QuantizationRange::Fixed { min: 0., max: 2. };
    let (index, report) = build(&tmp.path().join("a"), &docs, range);
    assert_eq!(report.clamped_impacts, 2);
    let params = index.quantizer().unwrap();
    assert_eq!(params, QuantizerParams::new(NBITS, 0., 2.));

    // Clamped values decode to the top level
    let top = search_maxscore(
        index.as_ref(),
        &HashMap::from([(0, 1.)]),
        1,
        Default::default(),
    );
    assert!((top[0].score - (2. - params.step / 2.)).abs() < 1e-6);

    // A per-index build clamps nothing
    let (_, report) = build(&tmp.path().join("b"), &docs, QuantizationRange::PerIndex);
    assert_eq!(report.clamped_impacts, 0);

    // Reference: same parameters, whatever the content
    let (other, _) = build(
        &tmp.path().join("c"),
        &docs,
        QuantizationRange::Reference(params),
    );
    assert_eq!(other.quantizer(), Some(params));
}

/// The segmented result type carries the segment index.
#[test]
fn segment_ids_in_results() {
    let tmp = TempDir::new().unwrap();
    let a: BTreeMap<DocId, Doc> = [(0, vec![(0, 1.0)]), (1, vec![(0, 0.5)])].into();
    let b: BTreeMap<DocId, Doc> = [(0, vec![(0, 1.5)])].into();
    let range = QuantizationRange::Fixed { min: 0., max: 2. };
    let (ia, _) = build(&tmp.path().join("a"), &a, range);
    let (ib, _) = build(&tmp.path().join("b"), &b, range);
    let search = SegmentedSearch::new(vec![
        Segment {
            index: ia.as_ref(),
            filter: None::<Exclude>,
        },
        Segment {
            index: ib.as_ref(),
            filter: None::<Exclude>,
        },
    ]);
    let r: Vec<(usize, DocId)> = search
        .search(&HashMap::from([(0, 1.)]), 10)
        .iter()
        .map(|SegmentScoredDocument { segment, docid, .. }| (*segment, *docid))
        .collect();
    assert_eq!(r, vec![(1, 0), (0, 0), (0, 1)]);
}
