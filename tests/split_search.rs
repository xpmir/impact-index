//! Search correctness on split indices with realistic (SPLADE-like) shapes:
//! lognormal impacts, Zipf-like term frequencies, long queries.
//!
//! Split search must return the same top-k (up to quantization error) as an
//! exhaustive evaluation over the raw postings.

use std::collections::{BTreeMap, HashMap};

use impact_index::{
    base::{load_index, DocId, ImpactValue, Len, TermIndex},
    builder::{BuilderOptions, Indexer},
    compress::{
        docid::{BitPackingCompressor, EliasFanoCompressor},
        impact::{GlobalQuantizerFactory, Identity, QuantizedBitPackedFactory},
        CompressionTransform, ImpactCompressorFactory,
    },
    index::SparseIndex,
    search::{maxscore::search_maxscore, maxscore::MaxScoreOptions, wand::search_wand},
    search::{ScoredDocument, TopScoredDocuments},
    transforms::{split::SplitIndexTransform, IndexTransform},
};
use ndarray::Array1;
use rand::{rngs::StdRng, Rng, SeedableRng};
use rand_distr::{Distribution, LogNormal, Zipf};
use rstest::rstest;
use temp_dir::TempDir;

const VOCABULARY: usize = 2_000;

struct Collection {
    dir: TempDir,
    /// docid -> (term -> impact)
    docs: Vec<BTreeMap<TermIndex, ImpactValue>>,
    queries: Vec<HashMap<TermIndex, ImpactValue>>,
}

fn build_collection(num_docs: usize, seed: u64) -> Collection {
    let mut rng = StdRng::seed_from_u64(seed);
    let zipf = Zipf::new(VOCABULARY as u64, 1.1).unwrap();
    let impacts = LogNormal::new(-0.5, 0.8).unwrap();

    let dir = TempDir::new().unwrap();
    let mut indexer = Indexer::<f32>::new(
        dir.path(),
        &BuilderOptions {
            in_memory_threshold: 1000,
            checkpoint_frequency: 0,
            checkpoint_flush_ratio: 0.5,
            positions: false,
        },
    );

    let mut docs = Vec::with_capacity(num_docs);
    for docid in 0..num_docs {
        let mut terms = BTreeMap::new();
        // Some documents have no postings at all
        if rng.gen::<f32>() > 0.02 {
            let n = rng.gen_range(5..80);
            for _ in 0..n {
                let t = (zipf.sample(&mut rng) as usize) - 1;
                let v: f32 = (impacts.sample(&mut rng) as f32).max(1e-3);
                terms.insert(t, v);
            }
        }
        if !terms.is_empty() {
            let t: Array1<TermIndex> = terms.keys().copied().collect();
            let v: Array1<ImpactValue> = terms.values().copied().collect();
            indexer.add(docid as DocId, &t, &v).unwrap();
        }
        docs.push(terms);
    }
    indexer.build().unwrap();

    let mut queries = Vec::new();
    for _ in 0..30 {
        let mut q = HashMap::new();
        let n = rng.gen_range(20..60);
        for _ in 0..n {
            let t = (zipf.sample(&mut rng) as usize) - 1;
            q.insert(t, (impacts.sample(&mut rng) as f32).max(1e-3));
        }
        queries.push(q);
    }

    // Keep the directory alive (indexer writes into it)
    let c = Collection { dir, docs, queries };
    drop(indexer);
    c
}

fn exhaustive(c: &Collection, q: &HashMap<TermIndex, ImpactValue>) -> Vec<f32> {
    c.docs
        .iter()
        .map(|d| {
            d.iter()
                .map(|(t, v)| q.get(t).map_or(0., |w| w * v))
                .sum::<f32>()
        })
        .collect()
}

/// Checks that `observed` is a valid top-k w.r.t. exact scores, with
/// tolerance `eps` (relative to the query's total weight).
fn check_topk(observed: &[ScoredDocument], scores: &[f32], top_k: usize, eps: f32, label: &str) {
    let mut top = TopScoredDocuments::new(top_k);
    for (d, &s) in scores.iter().enumerate() {
        if s > 0. {
            top.add(d as DocId, s);
        }
    }
    let expected = top.into_sorted_vec();
    assert_eq!(
        observed.len(),
        expected.len(),
        "{label}: result size differs"
    );
    for (i, (o, e)) in observed.iter().zip(expected.iter()).enumerate() {
        let exact = scores[o.docid as usize];
        assert!(
            (o.score - exact).abs() <= eps,
            "{label}: rank {i}, doc {} has score {} but exact score is {}",
            o.docid,
            o.score,
            exact
        );
        assert!(
            (o.score - e.score).abs() <= 2. * eps,
            "{label}: rank {i}: observed doc {} ({}), expected doc {} ({})",
            o.docid,
            o.score,
            e.docid,
            e.score
        );
    }
}

fn impacts_factory(name: &str) -> Box<dyn ImpactCompressorFactory> {
    match name {
        "identity" => Box::new(Identity {}),
        "qbp16" => Box::new(QuantizedBitPackedFactory { nbits: 16 }),
        "global16" => Box::new(GlobalQuantizerFactory { nbits: 16 }),
        _ => unreachable!(),
    }
}

#[rstest]
fn test_split_search_matches_exhaustive(
    #[values(vec![0.9], vec![0.5], vec![0.8, 0.95])] quantiles: Vec<f64>,
    #[values("identity", "qbp16", "global16")] impacts: &str,
    #[values(64, 128)] block_size: usize,
    #[values(true, false)] in_memory: bool,
) {
    let c = build_collection(3_000, 42);
    let raw_index = impact_index::builder::load_forward_index::<f32>(c.dir.path(), true);

    let doc_ids: Box<dyn impact_index::compress::DocIdCompressorFactory> = if block_size == 64 {
        Box::new(EliasFanoCompressor {})
    } else {
        Box::new(BitPackingCompressor {})
    };
    let transform = SplitIndexTransform {
        sink: Box::new(CompressionTransform {
            max_block_size: block_size,
            doc_ids_compressor_factory: doc_ids,
            impacts_compressor_factory: impacts_factory(impacts),
            positions_codec: None,
        }),
        quantiles: quantiles.clone(),
    };
    let out = TempDir::new().unwrap();
    let split_path = out.path().join("split");
    transform.process(&split_path, &raw_index).unwrap();
    let split = load_index(&split_path, in_memory);

    for (qi, q) in c.queries.iter().enumerate() {
        let scores = exhaustive(&c, q);
        let total_w: f32 = q.values().sum();
        let eps = if impacts == "identity" {
            1e-3 * total_w.max(1.)
        } else {
            1e-2 * total_w.max(1.)
        };
        for top_k in [10, 100] {
            let label = format!("q{qi} k{top_k} {quantiles:?} {impacts} bs{block_size}");
            let r = search_maxscore(&*split, q, top_k, MaxScoreOptions::default());
            check_topk(&r, &scores, top_k, eps, &format!("maxscore {label}"));
            let r = search_wand(&*split, q, top_k);
            check_topk(&r, &scores, top_k, eps, &format!("wand {label}"));
        }
    }
}

/// Every posting of a split index, iterated through `block_iterator`, must
/// match the source (docids and values, within quantization error).
#[test]
fn test_split_iterator_roundtrip_quantized() {
    let c = build_collection(2_000, 7);
    let raw_index = impact_index::builder::load_forward_index::<f32>(c.dir.path(), true);
    let transform = SplitIndexTransform {
        sink: Box::new(CompressionTransform {
            max_block_size: 128,
            doc_ids_compressor_factory: Box::new(BitPackingCompressor {}),
            impacts_compressor_factory: Box::new(QuantizedBitPackedFactory { nbits: 16 }),
            positions_codec: None,
        }),
        quantiles: vec![0.9],
    };
    let out = TempDir::new().unwrap();
    transform.process(out.path(), &raw_index).unwrap();
    let split = load_index(out.path(), true);

    for term_ix in 0..raw_index.len() {
        let mut expected = raw_index.block_iterator(term_ix);
        let mut observed = split.block_iterator(term_ix);
        let max_value = observed.max_value();
        let mut true_max: f32 = 0.;
        while let Some(a) = expected.next() {
            true_max = true_max.max(a.value);
            let b = observed
                .next()
                .unwrap_or_else(|| panic!("term {term_ix}: missing doc {}", a.docid));
            assert_eq!(a.docid, b.docid, "term {term_ix}");
            assert!(
                (a.value - b.value).abs() < 1e-2,
                "term {term_ix}, doc {}: {} vs {}",
                a.docid,
                a.value,
                b.value
            );
        }
        assert!(observed.next().is_none(), "term {term_ix}: extra postings");
        assert!(
            max_value + 1e-2 >= true_max,
            "term {term_ix}: max value {max_value} < {true_max}"
        );
    }
}

/// Sanity check of the test harness: the raw (non-split) index.
#[test]
fn test_raw_search_matches_exhaustive() {
    let c = build_collection(3_000, 42);
    let raw_index = impact_index::builder::load_forward_index::<f32>(c.dir.path(), true);
    for (qi, q) in c.queries.iter().enumerate() {
        let scores = exhaustive(&c, q);
        let eps = 1e-3 * q.values().sum::<f32>().max(1.);
        for top_k in [10, 100] {
            let r = search_maxscore(&raw_index, q, top_k, MaxScoreOptions::default());
            check_topk(
                &r,
                &scores,
                top_k,
                eps,
                &format!("raw maxscore q{qi} k{top_k}"),
            );
            let r = search_wand(&raw_index, q, top_k);
            check_topk(&r, &scores, top_k, eps, &format!("raw wand q{qi} k{top_k}"));
        }
    }
}
