use helpers::index::{check_same_index, TestIndex};
use impact_index::index::SparseIndex;
use impact_index::{
    base::load_index,
    compress::{
        docid::EliasFanoCompressor, impact::Identity, impact::Quantizer, CompressionTransform,
    },
    transforms::IndexTransform,
};
use rstest::rstest;
use temp_dir::TempDir;

#[rstest]
#[case(5. / ((2 << 4) as f64), CompressionTransform {
    max_block_size: 1024,
    doc_ids_compressor_factory: Box::new(EliasFanoCompressor {}),
    impacts_compressor_factory: Box::new(Quantizer::new(4, 0., 5.)),
    positions_codec: None,
})]
#[case(0., CompressionTransform {
    max_block_size: 1024,
    doc_ids_compressor_factory: Box::new(EliasFanoCompressor {}),
    impacts_compressor_factory: Box::new(Identity {}),
    positions_codec: None,
})]
fn test_compressed_index(#[case] step: f64, #[case] transform: CompressionTransform) {
    use std::collections::HashSet;

    use impact_index::{base::DocId, builder::BuilderOptions};

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
            positions: false,
        },
        &HashSet::<DocId>::from([]),
    );
    let index = data.indexer.to_index(true);

    let dir = TempDir::new().expect("Could not create temporary directory");

    transform
        .process(dir.path(), &index)
        .expect("An error occurred");

    let c_index = load_index(dir.path(), true);

    check_same_index(
        c_index.block_iterator(0).as_mut(),
        index.block_iterator(0).as_mut(),
        step,
    );
}

/// Block (and term) maxima must bound the *decoded* impacts: a quantised
/// value decodes to its bin centre, up to half a step above the raw value,
/// so maxima computed from raw values let block-max pruning drop top-k
/// documents.
#[test]
fn test_block_max_bounds_decoded_values() {
    use std::collections::HashSet;

    use impact_index::{base::DocId, builder::BuilderOptions};

    let mut data = TestIndex::new(
        50,
        2000,
        5.,
        10,
        Some(7),
        BuilderOptions {
            checkpoint_frequency: 0,
            in_memory_threshold: 100,
            checkpoint_flush_ratio: 0.5,
            positions: false,
        },
        &HashSet::<DocId>::from([]),
    );
    let index = data.indexer.to_index(true);
    let dir = TempDir::new().expect("Could not create temporary directory");
    CompressionTransform {
        max_block_size: 16,
        doc_ids_compressor_factory: Box::new(EliasFanoCompressor {}),
        impacts_compressor_factory: Box::new(Quantizer::new(4, 0., 5.)),
        positions_codec: None,
    }
    .process(dir.path(), &index)
    .expect("An error occurred");
    let c_index = load_index(dir.path(), true);

    for term in 0..c_index.len() {
        let mut it = c_index.block_iterator(term);
        let term_max = it.max_value();
        while it.next_min_doc_id(0).is_some() {
            let value = it.current().value;
            assert!(value <= it.max_block_value(), "term {term}: block max");
            assert!(value <= term_max, "term {term}: term max");
        }
    }
}
