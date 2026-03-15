use std::collections::HashMap;

use criterion::{criterion_group, criterion_main, Criterion};

use helpers::documents::{create_document, document_vectors};
use impact_index::{
    base::{load_index, SearchFn},
    builder::{BuilderOptions, Indexer},
    compress::{docid::BitPackingCompressor, impact::GlobalQuantizerFactory, CompressionTransform},
    search::{
        maxscore::{search_maxscore, MaxScoreOptions},
        wand::search_wand,
    },
    transforms::IndexTransform,
};
use log::info;
use rand::thread_rng;
use temp_dir::TempDir;

fn build_raw_index(dir: &std::path::Path) -> Box<dyn impact_index::index::SparseIndex> {
    let mut rng = thread_rng();
    const FLOPS: f32 = 1.;
    const NUM_DOCS: u64 = 100_000;
    const VOCABULARY_SIZE: usize = 1_000;

    let lambda_words: f32 = f32::sqrt(FLOPS * (VOCABULARY_SIZE as f32));

    info!(
        "Generating an index: FLOPS={}, # docs={}, # tokens={}, # lambda_tokens={}",
        FLOPS, NUM_DOCS, VOCABULARY_SIZE, lambda_words
    );

    let mut indexer = Indexer::new(
        dir,
        &BuilderOptions {
            in_memory_threshold: 128,
            checkpoint_frequency: 0,
            checkpoint_flush_ratio: 0.5,
        },
    );

    for doc_id in 0..NUM_DOCS {
        let document = create_document(lambda_words, 100, VOCABULARY_SIZE, &mut rng);
        let (terms, values) = document_vectors(&document);
        indexer
            .add(doc_id, &terms, &values)
            .expect("Error while adding terms to the index");
    }

    indexer.build().expect("Error while building the index");
    Box::new(indexer.to_index(true))
}

fn benchmark(c: &mut Criterion, name: &str, search_fn: SearchFn) {
    let dir = TempDir::new().expect("Could not create temporary directory");
    let index = build_raw_index(dir.path());

    let query = HashMap::from([(0, 1.2), (1, 2.3), (2, 3.2), (3, 1.2), (4, 0.7), (5, 2.3)]);

    c.bench_function(name, |b| b.iter(|| search_fn(&*index, &query, 1000)));
}

fn benchmark_compressed(c: &mut Criterion, name: &str, search_fn: SearchFn) {
    let dir = TempDir::new().expect("Could not create temporary directory");
    let raw_index = build_raw_index(dir.path());

    // Build compressed index with BitPacking + Quantizer
    let transform = CompressionTransform {
        max_block_size: 128,
        doc_ids_compressor_factory: Box::new(BitPackingCompressor {}),
        impacts_compressor_factory: Box::new(GlobalQuantizerFactory { nbits: 16 }),
    };
    let compressed_path = dir.path().join("compressed");
    transform
        .process(&compressed_path, raw_index.as_ref())
        .expect("Could not build compressed index");
    let compressed_index = load_index(&compressed_path, true);

    let query = HashMap::from([(0, 1.2), (1, 2.3), (2, 3.2), (3, 1.2), (4, 0.7), (5, 2.3)]);

    c.bench_function(name, |b| {
        b.iter(|| search_fn(&*compressed_index, &query, 1000))
    });
}

fn benchmark_maxscore(c: &mut Criterion) {
    benchmark(c, "raw_maxscore", |index, query, top_k| {
        search_maxscore(index, query, top_k, MaxScoreOptions::default())
    })
}

fn benchmark_wand(c: &mut Criterion) {
    benchmark(c, "raw_wand", search_wand)
}

fn benchmark_compressed_maxscore(c: &mut Criterion) {
    benchmark_compressed(c, "compressed_maxscore", |index, query, top_k| {
        search_maxscore(index, query, top_k, MaxScoreOptions::default())
    })
}

fn benchmark_compressed_wand(c: &mut Criterion) {
    benchmark_compressed(c, "compressed_wand", search_wand)
}

criterion_group! {
    name = benches;
    config = Criterion::default().significance_level(0.1).sample_size(500);
    targets = benchmark_maxscore, benchmark_wand, benchmark_compressed_maxscore, benchmark_compressed_wand
}
criterion_main!(benches);
