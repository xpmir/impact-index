use impact_index::{
    base::{DocId, ImpactValue, TermIndex},
    builder::BuilderOptions,
    index::SparseIndex,
    search::{maxscore::MaxScoreOptions, ScoredDocument, TopScoredDocuments},
    transforms::IndexTransform,
};
use log::{debug, info};
use rstest::rstest;

use helpers::index::TestIndex;
use impact_index::base::SearchFn;
use impact_index::search::{maxscore::search_maxscore, wand::search_wand};
use rand_distr::{Distribution, LogNormal};
use std::{
    collections::{HashMap, HashSet},
    fmt::Display,
};

/// Initialize the logger
fn init_logger() {
    let _ = env_logger::builder().is_test(true).try_init();
}

trait ApproxEq {
    fn approx_eq(&self, other: &Self, delta: f64) -> bool;
}

impl ApproxEq for ScoredDocument {
    fn approx_eq(&self, other: &Self, delta: f64) -> bool {
        (self.docid == other.docid) && ((self.score - other.score).abs() < (delta as f32))
    }
}

fn vec_compare<T>(observed: &Vec<T>, expected: &Vec<T>, delta: f64)
where
    T: ApproxEq + Display,
{
    assert!(
        observed.len() == expected.len(),
        "Size differ {} vs {}",
        observed.len(),
        expected.len()
    );
    for i in 0..expected.len() {
        assert!(
            observed[i].approx_eq(&expected[i], delta),
            "{}th element differ: {} vs {}",
            i,
            observed[i],
            expected[i]
        );
    }
}

#[test]
fn test_index() {
    let mut data = TestIndex::new(
        100,
        1000,
        5.,
        10,
        None,
        BuilderOptions {
            checkpoint_frequency: 0,
            in_memory_threshold: 10,
            checkpoint_flush_ratio: 0.5,
        },
        &HashSet::<DocId>::from([]),
    );
    let index = data.indexer.to_index(true);

    eprintln!("Index built in {}", &data.dir.path().display());
    // Verify the index
    for term_ix in 0..data.vocabulary_size {
        let mut iter = match data.all_terms.get(&term_ix) {
            Some(v) => Box::new(v.iter()),
            None => Box::new([].iter()),
        };

        for (ix, observed) in index.block_iterator(term_ix).enumerate() {
            let expected = iter.next().expect(&format!(
                "The index has too many elements for term {}",
                term_ix
            ));
            info!(
                "DocID {} vs {} for term {} [entry {}]",
                expected.docid, observed.docid, term_ix, ix
            );
            assert!(
                expected.docid == observed.docid,
                "Expected doc ID {}, got {} for term {} [entry {}]",
                expected.docid,
                observed.docid,
                term_ix,
                ix
            );
            assert!(expected.value == observed.value);
        }
    }
}

#[test]
fn test_heap() {
    let mut top = TopScoredDocuments::new(3);
    assert!(top.add(0, 0.1) == ImpactValue::NEG_INFINITY);

    let mut max = top.add(1, 0.2);
    assert!(
        max == ImpactValue::NEG_INFINITY,
        "Expected -inf got {}",
        max
    );

    max = top.add(2, 0.3);
    assert!(max == 0.1, "Expected 0.1 got {}", max);

    max = top.add(2, 0.05);
    assert!(max == 0.1, "Expected 0.1 got {}", max);

    max = top.add(3, 0.5);
    assert!(max == 0.2, "Expected 0.2 got {}", max);

    // Further tests
    let top_k = 10;
    let mut rng = rand::thread_rng();
    let log_normal = LogNormal::new(0., 1.).unwrap();

    let mut scored_documents: Vec<ScoredDocument> = Vec::new();
    top = TopScoredDocuments::new(top_k);
    for doc_id in 0..10000 {
        let score = log_normal.sample(&mut rng);
        top.add(doc_id, score);
        scored_documents.push(ScoredDocument {
            docid: doc_id,
            score,
        });
    }

    // Compare
    scored_documents.sort();
    let expected = scored_documents[0..top_k].to_vec();
    let observed = top.into_sorted_vec();

    vec_compare(&observed, &expected, 1e-2);
}

fn search_maxscore_default<'a>(
    index: &'a dyn SparseIndex,
    query: &HashMap<TermIndex, ImpactValue>,
    top_k: usize,
) -> Vec<ScoredDocument> {
    search_maxscore(index, query, top_k, MaxScoreOptions::default())
}

enum IndexType {
    InMemory,
    Disk,
    Split2,
    BitPacked,
    PFor,
    /// BitPacking + 16-bit quantized impacts (block_size=128) — the recommended config
    BitPacked16,
    /// BitPacking + 8-bit quantized impacts (block_size=128)
    BitPacked8,
    /// QuantizedBitPacked 8-bit (quantize then adaptive bitpack — for SPLADE)
    QuantizedBitPacked8,
}

#[rstest]
#[case(IndexType::InMemory, 100, 1000, 50., 50, 10, None, 0, vec![])]
// Checkpoint at 800, interrupts at 820
#[case(IndexType::InMemory, 100, 1000, 50., 50, 10, None, 800, vec![820])]
// Checkpoint every 300 documents, interrupts at 820
#[case(IndexType::InMemory, 100, 1000, 50., 50, 10, None, 300, vec![820])]
#[case(IndexType::InMemory, 100, 1000, 50., 50, 1, None, 0, vec![])]
// Sparse documents (max 8 postings, 5 in average) )
#[case(IndexType::InMemory, 500, 500, 5., 8, 10, Some(1), 0, vec![])]
#[case(IndexType::Disk, 500, 500, 5., 8, 10, Some(1), 0, vec![])]
#[case(IndexType::Split2, 500, 500, 5., 8, 10, Some(1), 0, vec![])]
#[case(IndexType::BitPacked, 500, 500, 5., 8, 10, Some(1), 0, vec![])]
// BitPacking + 16-bit quantized impacts (recommended production config)
#[case(IndexType::BitPacked16, 500, 500, 5., 8, 10, Some(1), 0, vec![])]
// BitPacking + 8-bit quantized impacts (fast path)
#[case(IndexType::BitPacked8, 500, 500, 5., 8, 10, Some(1), 0, vec![])]
// PFOR-delta compression
#[case(IndexType::PFor, 500, 500, 5., 8, 10, Some(1), 0, vec![])]
// QuantizedBitPacked 8-bit (for SPLADE-style float impacts)
#[case(IndexType::QuantizedBitPacked8, 500, 500, 5., 8, 10, Some(1), 0, vec![])]
fn test_search(
    #[case] index_type: IndexType,
    #[case] vocabulary_size: usize,
    #[case] document_count: DocId,
    #[case] lambda_words: f32,
    #[case] max_words: usize,
    #[case] top_k: usize,
    #[case] seed: Option<u64>,
    #[case] checkpoint_frequency: DocId,
    #[case] index_interruptions: Vec<DocId>,
    #[values(search_wand, search_maxscore_default)] search_fn: SearchFn,
) {
    use std::collections::HashSet;

    use impact_index::{
        base::load_index,
        builder::{load_forward_index, BuilderOptions},
        compress::{docid::EliasFanoCompressor, impact::Identity, CompressionTransform},
        transforms::split::SplitIndexTransform,
    };

    init_logger();
    // std::env::set_var("RUST_LOG", "trace");
    debug!("Search test start");
    let index_interruptions_set = HashSet::<DocId>::from_iter(index_interruptions.iter().copied());

    let mut data = TestIndex::new(
        vocabulary_size,
        document_count,
        lambda_words,
        max_words,
        seed,
        // Use small pages
        BuilderOptions {
            in_memory_threshold: 10,
            checkpoint_frequency: checkpoint_frequency,
            checkpoint_flush_ratio: 0.5,
        },
        &index_interruptions_set,
    );
    let index = match index_type {
        IndexType::InMemory => Box::new(data.indexer.to_index(true)),
        IndexType::Disk => Box::new(load_forward_index::<f32>(data.dir.path(), true)),
        IndexType::Split2 => {
            let transform = SplitIndexTransform {
                sink: Box::new(CompressionTransform {
                    max_block_size: 1024,
                    doc_ids_compressor_factory: Box::new(EliasFanoCompressor {}),
                    impacts_compressor_factory: Box::new(Identity {}),
                }),
                quantiles: [63. / 64.].to_vec(),
            };
            let split_index_path = data.dir.path().join("split");
            transform
                .process(&split_index_path, &data.indexer.to_index(true))
                .expect("Could not build the new index");
            load_index(&split_index_path, true)
        }
        IndexType::BitPacked => {
            use impact_index::compress::docid::BitPackingCompressor;
            let transform = CompressionTransform {
                max_block_size: 128,
                doc_ids_compressor_factory: Box::new(BitPackingCompressor {}),
                impacts_compressor_factory: Box::new(Identity {}),
            };
            let compressed_path = data.dir.path().join("bitpacked");
            transform
                .process(&compressed_path, &data.indexer.to_index(true))
                .expect("Could not build compressed index");
            load_index(&compressed_path, true)
        }
        IndexType::BitPacked16 => {
            use impact_index::compress::docid::BitPackingCompressor;
            use impact_index::compress::impact::GlobalQuantizerFactory;
            let transform = CompressionTransform {
                max_block_size: 128,
                doc_ids_compressor_factory: Box::new(BitPackingCompressor {}),
                impacts_compressor_factory: Box::new(GlobalQuantizerFactory { nbits: 16 }),
            };
            let compressed_path = data.dir.path().join("bitpacked16");
            transform
                .process(&compressed_path, &data.indexer.to_index(true))
                .expect("Could not build compressed index");
            load_index(&compressed_path, true)
        }
        IndexType::BitPacked8 => {
            use impact_index::compress::docid::BitPackingCompressor;
            use impact_index::compress::impact::GlobalQuantizerFactory;
            let transform = CompressionTransform {
                max_block_size: 128,
                doc_ids_compressor_factory: Box::new(BitPackingCompressor {}),
                impacts_compressor_factory: Box::new(GlobalQuantizerFactory { nbits: 8 }),
            };
            let compressed_path = data.dir.path().join("bitpacked8");
            transform
                .process(&compressed_path, &data.indexer.to_index(true))
                .expect("Could not build compressed index");
            load_index(&compressed_path, true)
        }
        IndexType::PFor => {
            use impact_index::compress::docid::PForCompressor;
            let transform = CompressionTransform {
                max_block_size: 128,
                doc_ids_compressor_factory: Box::new(PForCompressor {}),
                impacts_compressor_factory: Box::new(Identity {}),
            };
            let compressed_path = data.dir.path().join("pfor");
            transform
                .process(&compressed_path, &data.indexer.to_index(true))
                .expect("Could not build compressed index");
            load_index(&compressed_path, true)
        }
        IndexType::QuantizedBitPacked8 => {
            use impact_index::compress::docid::BitPackingCompressor;
            use impact_index::compress::impact::QuantizedBitPackedFactory;
            let transform = CompressionTransform {
                max_block_size: 128,
                doc_ids_compressor_factory: Box::new(BitPackingCompressor {}),
                impacts_compressor_factory: Box::new(QuantizedBitPackedFactory { nbits: 8 }),
            };
            let compressed_path = data.dir.path().join("qbp8");
            transform
                .process(&compressed_path, &data.indexer.to_index(true))
                .expect("Could not build compressed index");
            load_index(&compressed_path, true)
        }
    };

    // Builds a query from a document
    let query: HashMap<usize, ImpactValue> = data.documents[10]
        .terms
        .iter()
        .map(|x| (x.term_ix, x.weight as ImpactValue))
        .collect();

    // Search with WAND
    let observed = search_fn(&*index, &query, top_k);
    eprintln!("(1) observed results");
    for (ix, result) in observed.iter().enumerate() {
        eprintln!(
            " [{}] document {}, score {}",
            ix, result.docid, result.score
        );
    }

    // Searching by iterating
    let mut top = TopScoredDocuments::new(top_k);
    for (doc_id, document) in data.documents.iter().enumerate() {
        let mut score = 0.;
        for tw in document.terms.iter() {
            score += match query.get(&tw.term_ix) {
                Some(s) => *s,
                None => 0.,
            } * (tw.weight as ImpactValue)
        }
        if score > 0. {
            top.add(doc_id.try_into().unwrap(), score);
        }
    }
    eprintln!("(2) expected results");
    let expected = top.into_sorted_vec();
    for (ix, result) in expected.iter().enumerate() {
        eprintln!(
            " [{}] document {}, score {}",
            ix, result.docid, result.score
        );
    }

    // Quantized indices have lossy compression — use wider tolerance
    let delta = match index_type {
        IndexType::BitPacked8 | IndexType::QuantizedBitPacked8 => 5e-2,
        IndexType::BitPacked16 => 1e-2,
        _ => 1e-2,
    };
    vec_compare(&observed, &expected, delta);
}

/// Test that compares the output of legacy and streaming BMP conversion.
///
/// The two methods should produce byte-identical output files.
#[test]
fn test_bmp_conversion_comparison() {
    use std::fs;

    init_logger();

    // Create a test index
    let mut data = TestIndex::new(
        50,       // vocabulary_size
        200,      // document_count
        5.,       // lambda_words
        10,       // max_words
        Some(42), // seed for reproducibility
        BuilderOptions {
            checkpoint_frequency: 0,
            in_memory_threshold: 10,
            checkpoint_flush_ratio: 0.5,
        },
        &HashSet::<DocId>::from([]),
    );

    let index = data.indexer.to_index(true);

    // Create temp files for both outputs
    let legacy_path = data.dir.path().join("legacy.bmp");
    let streaming_path = data.dir.path().join("streaming.bmp");

    // Convert using legacy method
    index
        .convert_to_bmp(&legacy_path, 32, false)
        .expect("Legacy conversion failed");

    // Convert using streaming method
    index
        .convert_to_bmp_streaming(&streaming_path, 32, false)
        .expect("Streaming conversion failed");

    // Read both files and compare
    let legacy_bytes = fs::read(&legacy_path).expect("Failed to read legacy file");
    let streaming_bytes = fs::read(&streaming_path).expect("Failed to read streaming file");

    // Compare file sizes
    assert_eq!(
        legacy_bytes.len(),
        streaming_bytes.len(),
        "File sizes differ: legacy={}, streaming={}",
        legacy_bytes.len(),
        streaming_bytes.len()
    );

    // Compare contents byte by byte
    let mut diff_count = 0;
    let mut first_diff = None;
    for (i, (a, b)) in legacy_bytes.iter().zip(streaming_bytes.iter()).enumerate() {
        if a != b {
            diff_count += 1;
            if first_diff.is_none() {
                first_diff = Some(i);
            }
        }
    }

    if diff_count > 0 {
        eprintln!(
            "Files differ: {} bytes differ, first difference at offset {}",
            diff_count,
            first_diff.unwrap()
        );

        // For debugging: show the region around the first difference
        if let Some(offset) = first_diff {
            let start = offset.saturating_sub(16);
            let end = (offset + 16).min(legacy_bytes.len());
            eprintln!("Legacy   @ {}: {:?}", start, &legacy_bytes[start..end]);
            eprintln!("Streaming @ {}: {:?}", start, &streaming_bytes[start..end]);
        }
    }

    assert_eq!(
        diff_count, 0,
        "Files should be byte-identical, but {} bytes differ",
        diff_count
    );
}

/// Test streaming BMP conversion with compression enabled.
#[test]
fn test_bmp_streaming_with_compression() {
    use std::fs;

    init_logger();

    let mut data = TestIndex::new(
        30,
        100,
        3.,
        8,
        Some(123),
        BuilderOptions {
            checkpoint_frequency: 0,
            in_memory_threshold: 10,
            checkpoint_flush_ratio: 0.5,
        },
        &HashSet::<DocId>::from([]),
    );

    let index = data.indexer.to_index(true);
    let output_path = data.dir.path().join("compressed.bmp");

    // Convert with compression enabled
    index
        .convert_to_bmp_streaming(&output_path, 64, true)
        .expect("Streaming conversion with compression failed");

    // Verify the file was created and has content
    let metadata = fs::metadata(&output_path).expect("Failed to get file metadata");
    assert!(metadata.len() > 0, "Output file should not be empty");

    eprintln!("Compressed BMP file size: {} bytes", metadata.len());
}

/// Stress test comparing legacy and streaming BMP conversion with various parameters.
///
/// Tests different block sizes, document counts, and posting densities to ensure
/// streaming and legacy methods produce identical output across edge cases.
#[rstest]
// Small block size (8) - many blocks per term
#[case(100, 500, 10., 20, 8, false, Some(1))]
#[case(100, 500, 10., 20, 8, true, Some(2))]
// Very small block size (4) - stress test block boundaries
#[case(50, 200, 8., 15, 4, false, Some(3))]
#[case(50, 200, 8., 15, 4, true, Some(4))]
// Large number of documents spanning many blocks
#[case(200, 2000, 15., 30, 16, false, Some(5))]
#[case(200, 2000, 15., 30, 16, true, Some(6))]
// Dense postings (high lambda) - more postings per document
#[case(100, 500, 50., 100, 32, false, Some(7))]
#[case(100, 500, 50., 100, 32, true, Some(8))]
// Sparse postings with small blocks
#[case(500, 1000, 3., 5, 8, false, Some(9))]
// Large vocabulary with moderate documents
#[case(1000, 500, 5., 10, 16, false, Some(10))]
// Edge case: documents exactly filling blocks
#[case(50, 128, 10., 20, 32, false, Some(11))]
#[case(50, 256, 10., 20, 64, false, Some(12))]
// High posting density with tiny blocks
#[case(30, 100, 20., 30, 4, false, Some(13))]
#[case(30, 100, 20., 30, 4, true, Some(14))]
fn test_bmp_conversion_stress(
    #[case] vocabulary_size: usize,
    #[case] document_count: DocId,
    #[case] lambda_words: f32,
    #[case] max_words: usize,
    #[case] bsize: usize,
    #[case] compress_range: bool,
    #[case] seed: Option<u64>,
) {
    use std::fs;

    init_logger();

    eprintln!(
        "Testing: vocab={}, docs={}, lambda={}, max_words={}, bsize={}, compress={}",
        vocabulary_size, document_count, lambda_words, max_words, bsize, compress_range
    );

    let mut data = TestIndex::new(
        vocabulary_size,
        document_count,
        lambda_words,
        max_words,
        seed,
        BuilderOptions {
            checkpoint_frequency: 0,
            in_memory_threshold: 10,
            checkpoint_flush_ratio: 0.5,
        },
        &HashSet::<DocId>::from([]),
    );

    let index = data.indexer.to_index(true);

    // Create temp files for both outputs
    let legacy_path = data.dir.path().join("legacy.bmp");
    let streaming_path = data.dir.path().join("streaming.bmp");

    // Convert using legacy method
    index
        .convert_to_bmp(&legacy_path, bsize, compress_range)
        .expect("Legacy conversion failed");

    // Convert using streaming method
    index
        .convert_to_bmp_streaming(&streaming_path, bsize, compress_range)
        .expect("Streaming conversion failed");

    // Read both files and compare
    let legacy_bytes = fs::read(&legacy_path).expect("Failed to read legacy file");
    let streaming_bytes = fs::read(&streaming_path).expect("Failed to read streaming file");

    eprintln!(
        "File sizes: legacy={} bytes, streaming={} bytes",
        legacy_bytes.len(),
        streaming_bytes.len()
    );

    // Compare file sizes
    assert_eq!(
        legacy_bytes.len(),
        streaming_bytes.len(),
        "File sizes differ: legacy={}, streaming={}",
        legacy_bytes.len(),
        streaming_bytes.len()
    );

    // Compare contents byte by byte
    let mut diff_count = 0;
    let mut first_diff = None;
    for (i, (a, b)) in legacy_bytes.iter().zip(streaming_bytes.iter()).enumerate() {
        if a != b {
            diff_count += 1;
            if first_diff.is_none() {
                first_diff = Some(i);
            }
        }
    }

    if diff_count > 0 {
        eprintln!(
            "Files differ: {} bytes differ, first difference at offset {}",
            diff_count,
            first_diff.unwrap()
        );

        // For debugging: show the region around the first difference
        if let Some(offset) = first_diff {
            let start = offset.saturating_sub(16);
            let end = (offset + 16).min(legacy_bytes.len());
            eprintln!("Legacy    @ {}: {:?}", start, &legacy_bytes[start..end]);
            eprintln!("Streaming @ {}: {:?}", start, &streaming_bytes[start..end]);
        }
    }

    assert_eq!(
        diff_count, 0,
        "Files should be byte-identical, but {} bytes differ",
        diff_count
    );
}
