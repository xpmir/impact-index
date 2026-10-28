//! The randomised incremental-indexing check of `tests/incremental.rs`,
//! inside WASM: add / modify / delete documents over a growing list of
//! segments built in memory, and compare the segmented search (with
//! tombstones) to an index built from scratch over the live documents.
//!
//! ```sh
//! CARGO_TARGET_WASM32_UNKNOWN_UNKNOWN_RUNNER=wasm-bindgen-test-runner \
//!     cargo test -p impact-index-wasm --target wasm32-unknown-unknown
//! ```

use std::collections::{BTreeMap, HashMap};

use impact_index::base::{load_index_from_files, DocId, ImpactValue, TermIndex};
use impact_index::builder::MemoryIndexer;
use impact_index::compress::docid::EliasFanoCompressor;
use impact_index::compress::impact::{
    QuantizationRange, QuantizerCodec, QuantizerParams, RangedQuantizerFactory,
};
use impact_index::compress::CompressionTransform;
use impact_index::index::SparseIndex;
use impact_index::search::filter::{DocBitSet, Exclude};
use impact_index::search::maxscore::{search_maxscore, MaxScoreOptions};
use impact_index::search::segments::{Segment, SegmentAlgorithm, SegmentedSearch};
use impact_index::search::wand::search_wand;
use impact_index::storage::MemoryOutput;
use wasm_bindgen_test::wasm_bindgen_test;

const VOCABULARY: usize = 200;
const NBITS: u32 = 8;

/// Deterministic PRNG (same as the native tests)
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

fn random_doc(rng: &mut Rng, max_value: f32) -> Doc {
    let n = 3 + rng.below(15);
    let terms: BTreeMap<TermIndex, ImpactValue> = (0..n)
        .map(|_| {
            let t = (rng.next_f32().powi(2) * VOCABULARY as f32) as TermIndex;
            (t, 0.01 + rng.next_f32() * (max_value - 0.01))
        })
        .collect();
    terms.into_iter().collect()
}

fn random_query(rng: &mut Rng) -> HashMap<TermIndex, ImpactValue> {
    (0..2 + rng.below(6))
        .map(|_| (rng.below(VOCABULARY), 0.5 + rng.next_f32()))
        .collect()
}

/// Builds a compressed index over `docs` in memory, and loads it back
fn build(docs: &BTreeMap<DocId, Doc>, range: QuantizationRange) -> Box<dyn SparseIndex> {
    let mut indexer = MemoryIndexer::new();
    for (&docid, doc) in docs {
        let (terms, values): (Vec<_>, Vec<_>) = doc.iter().copied().unzip();
        indexer.add(docid, &terms, &values).unwrap();
    }
    let out = MemoryOutput::new();
    CompressionTransform {
        max_block_size: 32,
        doc_ids_compressor_factory: Box::new(EliasFanoCompressor {}),
        impacts_compressor_factory: Box::new(RangedQuantizerFactory {
            nbits: NBITS,
            range,
            codec: QuantizerCodec::BitPacked,
        }),
        positions_codec: None,
    }
    .compress_to(&out, &indexer)
    .unwrap();
    load_index_from_files(&out.into_files(), true).unwrap()
}

/// Same scores (within `tol`); with `tol == 0`, also the same documents
/// above the k-th score (ties at the k-th score may differ)
fn assert_same_top_k(got: &[(DocId, ImpactValue)], expected: &[(DocId, ImpactValue)], tol: f32) {
    assert_eq!(got.len(), expected.len(), "result count");
    for (g, e) in got.iter().zip(expected) {
        assert!(
            (g.1 - e.1).abs() <= tol,
            "scores differ: {got:?} vs {expected:?}"
        );
    }
    if let (0., Some(&(_, kth))) = (tol, expected.last()) {
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

#[derive(Clone, Copy)]
enum Mode {
    PerIndex,
    Fixed,
    Reference,
}

fn run_simulation(mode: Mode, seed: u64) {
    let mut rng = Rng(seed);
    let max_value = 2.;

    let mut live: BTreeMap<DocId, Doc> = BTreeMap::new();
    let mut location: HashMap<DocId, usize> = HashMap::new();
    let mut segments: Vec<Box<dyn SparseIndex>> = Vec::new();
    let mut tombstones: Vec<DocBitSet> = Vec::new();
    let mut reference: Option<QuantizerParams> = None;
    let mut next_docid: DocId = 0;
    let fixed = QuantizationRange::Fixed {
        min: 0.,
        max: max_value,
    };

    for round in 0..5 {
        let mut buffer: BTreeMap<DocId, Doc> = BTreeMap::new();
        let n_ops = if round == 0 { 300 } else { 40 + rng.below(80) };
        for _ in 0..n_ops {
            let op = rng.below(10);
            let existing: Vec<DocId> = live.keys().copied().collect();
            if op < 6 || existing.is_empty() {
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
                    let doc = random_doc(&mut rng, max_value * 1.2);
                    live.insert(docid, doc.clone());
                    buffer.insert(docid, doc);
                } else {
                    live.remove(&docid);
                    buffer.remove(&docid);
                }
            }
        }

        if !buffer.is_empty() {
            let range = match mode {
                Mode::PerIndex => QuantizationRange::PerIndex,
                Mode::Fixed => fixed,
                Mode::Reference => reference
                    .map(QuantizationRange::Reference)
                    .unwrap_or(QuantizationRange::PerIndex),
            };
            let index = build(&buffer, range);
            if reference.is_none() {
                reference = index.quantizer();
            }
            let seg = segments.len();
            for &docid in buffer.keys() {
                location.insert(docid, seg);
            }
            segments.push(index);
            tombstones.push(DocBitSet::new());
        }

        let scratch = build(
            &live,
            match mode {
                Mode::PerIndex => QuantizationRange::PerIndex,
                Mode::Fixed => fixed,
                Mode::Reference => QuantizationRange::Reference(reference.unwrap()),
            },
        );
        let step = (max_value * 1.2) / (1u32 << NBITS) as f32;

        for algorithm in [
            SegmentAlgorithm::MaxScore(MaxScoreOptions::default()),
            SegmentAlgorithm::Wand,
        ] {
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
                    for r in &results {
                        assert_eq!(location.get(&r.docid), Some(&r.segment));
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

#[wasm_bindgen_test]
fn segments_match_from_scratch_fixed() {
    run_simulation(Mode::Fixed, 1);
}

#[wasm_bindgen_test]
fn segments_match_from_scratch_reference() {
    run_simulation(Mode::Reference, 2);
}

#[wasm_bindgen_test]
fn segments_match_from_scratch_per_index() {
    run_simulation(Mode::PerIndex, 3);
}
