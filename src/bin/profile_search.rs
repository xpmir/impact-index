//! Profiling binary: builds index once, then runs search in a tight loop.
//!
//! Usage:
//!   CARGO_PROFILE_RELEASE_DEBUG=true cargo build --release --bin profile_search
//!   ./target/release/profile_search &
//!   sample <pid> 10 -f /tmp/profile.txt

use std::collections::HashMap;

use impact_index::{
    base::{load_index, ImpactValue, TermIndex},
    builder::{BuilderOptions, Indexer},
    compress::{docid::BitPackingCompressor, impact::GlobalQuantizerFactory, CompressionTransform},
    search::maxscore::{search_maxscore, MaxScoreOptions},
    transforms::IndexTransform,
};
use ndarray::Array1;

fn main() {
    let tmpdir = std::env::temp_dir().join(format!("profile_search_{}", std::process::id()));
    std::fs::create_dir_all(&tmpdir).unwrap();

    // Build raw index with random data
    eprintln!("Building raw index ({} docs)...", 100_000);
    let mut indexer = Indexer::<f32>::new(
        &tmpdir,
        &BuilderOptions {
            in_memory_threshold: 128,
            checkpoint_frequency: 0,
            checkpoint_flush_ratio: 0.5,
        },
    );

    const NUM_DOCS: u64 = 1_000_000;
    const VOCABULARY_SIZE: usize = 10_000;

    // Simple deterministic pseudo-random for reproducibility
    let mut seed: u64 = 42;
    let mut next_rand = || -> f32 {
        seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1);
        ((seed >> 33) as f32) / (u32::MAX as f32) * 2.0
    };

    for doc_id in 0..NUM_DOCS {
        let num_terms = 5 + ((next_rand() * 30.0) as usize).min(50);
        // Use a set to avoid duplicate term indices
        let mut term_set = std::collections::BTreeSet::new();
        let mut term_vals = Vec::new();
        for _ in 0..num_terms {
            let t = (next_rand() * VOCABULARY_SIZE as f32) as TermIndex % VOCABULARY_SIZE;
            let v = next_rand().abs() + 0.1;
            if term_set.insert(t) {
                term_vals.push((t, v));
            }
        }
        let terms: Array1<TermIndex> = Array1::from_iter(term_vals.iter().map(|(t, _)| *t));
        let values: Array1<f32> = Array1::from_iter(term_vals.iter().map(|(_, v)| *v));
        indexer.add(doc_id, &terms, &values).unwrap();
    }
    indexer.build().unwrap();
    let raw_index = indexer.to_index(true);

    // Build compressed index
    eprintln!("Building compressed index...");
    let transform = CompressionTransform {
        max_block_size: 128,
        doc_ids_compressor_factory: Box::new(BitPackingCompressor {}),
        impacts_compressor_factory: Box::new(GlobalQuantizerFactory { nbits: 16 }),
    };
    let compressed_path = tmpdir.join("compressed");
    transform.process(&compressed_path, &raw_index).unwrap();
    let index = load_index(&compressed_path, true);

    // Multiple queries with different term combinations
    let queries: Vec<HashMap<TermIndex, ImpactValue>> = (0..20)
        .map(|i| {
            let mut q = HashMap::new();
            for j in 0..5 {
                q.insert((i * 7 + j * 13) % VOCABULARY_SIZE, 1.0 + (j as f32) * 0.5);
            }
            q
        })
        .collect();

    // Warmup
    eprintln!("Warming up...");
    for q in &queries {
        for _ in 0..100 {
            let _ = search_maxscore(&*index, q, 100, MaxScoreOptions::default());
        }
    }

    eprintln!("=== PROFILING LOOP START ===");
    eprintln!("PID: {}", std::process::id());
    eprintln!("Run: sample {} 10 -f /tmp/profile.txt", std::process::id());

    // Tight search loop
    let start = std::time::Instant::now();
    let mut iterations = 0u64;
    while start.elapsed().as_secs() < 30 {
        for q in &queries {
            let _ = search_maxscore(&*index, q, 100, MaxScoreOptions::default());
            iterations += 1;
        }
    }
    let elapsed = start.elapsed().as_secs_f64();
    eprintln!(
        "=== DONE: {} iterations in {:.1}s ({:.0} q/s) ===",
        iterations,
        elapsed,
        iterations as f64 / elapsed
    );

    // Cleanup
    let _ = std::fs::remove_dir_all(&tmpdir);
}
