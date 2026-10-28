//! Builds synthetic indexes and the native search results over them, for
//! checking the WASM build against native (`wasm/tests/check.mjs`).
//!
//! ```sh
//! cargo run --release --example wasm_fixture -- <out-dir> [num-docs] [num-queries]
//! ```
//!
//! Writes into `<out-dir>`:
//! - `compressed/`: Elias-Fano docids, 8-bit quantised impacts (fixed range)
//! - `split/`: the same postings, split by impact quantiles (PFOR docids)
//! - `seg0/`, `seg1/`: two segments (halves of the collection)
//! - `fixture.json`: queries, deleted documents per segment, the native
//!   top-k for each index and for the segmented search, and (up to 50k
//!   documents) the documents, so that the WASM build can rebuild
//!   `compressed/` and compare

use std::collections::{BTreeMap, HashMap};
use std::path::Path;

use ndarray::Array1;
use serde_json::{json, Value};

use impact_index::base::{load_index, DocId, ImpactValue, TermIndex};
use impact_index::builder::{BuilderOptions, Indexer};
use impact_index::compress::docid::{EliasFanoCompressor, PForCompressor};
use impact_index::compress::impact::{QuantizationRange, QuantizerCodec, RangedQuantizerFactory};
use impact_index::compress::{CompressionTransform, DocIdCompressorFactory};
use impact_index::search::filter::{DocBitSet, Exclude};
use impact_index::search::maxscore::{search_maxscore, MaxScoreOptions};
use impact_index::search::segments::{Segment, SegmentedSearch};
use impact_index::transforms::split::SplitIndexTransform;
use impact_index::transforms::IndexTransform;

const VOCABULARY: usize = 30_000;
const TOP_K: usize = 100;

/// Deterministic PRNG
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
    /// Skewed term distribution (frequent low term ids), as in learned
    /// sparse vocabularies
    fn term(&mut self) -> TermIndex {
        (self.next_f32().powi(3) * VOCABULARY as f32) as TermIndex
    }
}

type Doc = Vec<(TermIndex, ImpactValue)>;

fn random_doc(rng: &mut Rng) -> Doc {
    let n = 30 + rng.below(70);
    let terms: BTreeMap<TermIndex, ImpactValue> = (0..n)
        .map(|_| (rng.term(), 0.01 + 2.99 * rng.next_f32().powi(2)))
        .collect();
    terms.into_iter().collect()
}

fn compression(docids: Box<dyn DocIdCompressorFactory>) -> CompressionTransform {
    CompressionTransform {
        max_block_size: 128,
        doc_ids_compressor_factory: docids,
        impacts_compressor_factory: Box::new(RangedQuantizerFactory {
            nbits: 8,
            range: QuantizationRange::Fixed { min: 0., max: 3. },
            codec: QuantizerCodec::BitPacked,
        }),
        positions_codec: None,
    }
}

/// Forward index over `docs`, renumbered from 0
fn forward(dir: &Path, docs: &[Doc]) -> Indexer<f32> {
    std::fs::create_dir_all(dir).unwrap();
    let mut indexer = Indexer::<f32>::new(
        dir,
        &BuilderOptions {
            checkpoint_frequency: 0,
            in_memory_threshold: 128,
            checkpoint_flush_ratio: 0.5,
            positions: false,
        },
    );
    for (docid, doc) in docs.iter().enumerate() {
        let terms: Array1<TermIndex> = doc.iter().map(|(t, _)| *t).collect();
        let values: Array1<f32> = doc.iter().map(|(_, v)| *v).collect();
        indexer.add(docid as DocId, &terms, &values).unwrap();
    }
    indexer.build().unwrap();
    indexer
}

fn results(docs: &[impact_index::search::ScoredDocument]) -> Value {
    docs.iter().map(|d| json!([d.docid, d.score])).collect()
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let out = Path::new(
        args.get(1)
            .expect("usage: wasm_fixture <out-dir> [docs] [queries]"),
    );
    let num_docs: usize = args.get(2).map_or(20_000, |s| s.parse().unwrap());
    let num_queries: usize = args.get(3).map_or(50, |s| s.parse().unwrap());

    let mut rng = Rng(2026);
    let docs: Vec<Doc> = (0..num_docs).map(|_| random_doc(&mut rng)).collect();
    let tmp = out.join("forward");

    let mut indexer = forward(&tmp.join("all"), &docs);
    let raw = indexer.to_index(true);
    compression(Box::new(EliasFanoCompressor {}))
        .process(&out.join("compressed"), &raw)
        .unwrap();
    SplitIndexTransform {
        sink: Box::new(compression(Box::new(PForCompressor {}))),
        quantiles: vec![0.9],
    }
    .process(&out.join("split"), &raw)
    .unwrap();

    let half = num_docs / 2;
    for (name, part) in [("seg0", &docs[..half]), ("seg1", &docs[half..])] {
        let mut indexer = forward(&tmp.join(name), part);
        compression(Box::new(PForCompressor {}))
            .process(&out.join(name), &indexer.to_index(true))
            .unwrap();
    }
    std::fs::remove_dir_all(&tmp).unwrap();

    let compressed = load_index(&out.join("compressed"), true);
    let split = load_index(&out.join("split"), true);
    let seg0 = load_index(&out.join("seg0"), true);
    let seg1 = load_index(&out.join("seg1"), true);

    // Every 7th document of segment 0 and every 5th of segment 1 deleted
    let deleted0: DocBitSet = (0..half as DocId).step_by(7).collect();
    let deleted1: DocBitSet = (0..(num_docs - half) as DocId).step_by(5).collect();

    let mut queries = Vec::new();
    let (mut r_compressed, mut r_split, mut r_segmented) = (vec![], vec![], vec![]);
    for _ in 0..num_queries {
        let query: HashMap<TermIndex, ImpactValue> = (0..5 + rng.below(20))
            .map(|_| (rng.term(), 0.1 + 2. * rng.next_f32()))
            .collect();
        let options = MaxScoreOptions::default();
        r_compressed.push(results(&search_maxscore(
            &*compressed,
            &query,
            TOP_K,
            options.clone(),
        )));
        r_split.push(results(&search_maxscore(
            &*split,
            &query,
            TOP_K,
            options.clone(),
        )));
        let segmented = SegmentedSearch::new(vec![
            Segment {
                index: &*seg0,
                filter: Exclude(&deleted0),
            },
            Segment {
                index: &*seg1,
                filter: Exclude(&deleted1),
            },
        ])
        .search(&query, TOP_K);
        r_segmented.push(
            segmented
                .iter()
                .map(|d| json!([d.segment, d.docid, d.score]))
                .collect::<Value>(),
        );
        let (terms, weights): (Vec<_>, Vec<_>) = query.into_iter().unzip();
        queries.push(json!({"terms": terms, "weights": weights}));
    }

    let docs_json: Value = if num_docs <= 50_000 {
        docs.iter()
            .map(|doc| {
                let (terms, values): (Vec<_>, Vec<_>) = doc.iter().copied().unzip();
                json!([terms, values])
            })
            .collect()
    } else {
        Value::Null
    };

    let fixture = json!({
        "docs": docs_json,
        "top_k": TOP_K,
        "num_docs": num_docs,
        "queries": queries,
        "deleted": [deleted0.iter().collect::<Vec<_>>(), deleted1.iter().collect::<Vec<_>>()],
        "quantizer": compressed.quantizer().map(|q| json!({"nbits": q.nbits, "min": q.min, "max": q.max})),
        "expected": {
            "compressed": r_compressed,
            "split": r_split,
            "segmented": r_segmented,
        },
    });
    std::fs::write(
        out.join("fixture.json"),
        serde_json::to_string(&fixture).unwrap(),
    )
    .unwrap();
}
