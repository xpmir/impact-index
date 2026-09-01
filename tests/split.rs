use std::collections::HashSet;

use helpers::index::{check_same_index, TestIndex};
use impact_index::base::DocId;
use impact_index::builder::BuilderOptions;
use impact_index::index::SparseIndex;
use impact_index::transforms::split::SplitIndexTransform;
use impact_index::{
    base::load_index,
    compress::{docid::EliasFanoCompressor, impact::Identity, CompressionTransform},
    transforms::IndexTransform,
};
use temp_dir::TempDir;

#[test]
fn test_split_index() {
    let mut data = TestIndex::new(
        100,
        10_000,
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

    let sink = Box::new(CompressionTransform {
        max_block_size: 1024,
        doc_ids_compressor_factory: Box::new(EliasFanoCompressor {}),
        impacts_compressor_factory: Box::new(Identity {}),
        positions_codec: None,
    });

    let transform = SplitIndexTransform {
        sink,
        quantiles: [50. / 64., 63. / 64.].to_vec(),
    };

    transform
        .process(dir.path(), &index)
        .expect("An error occurred");

    let c_index = load_index(dir.path(), true);

    for term_ix in [0, 5, 27, 99] {
        check_same_index(
            index.block_iterator(term_ix).as_mut(),
            c_index.block_iterator(term_ix).as_mut(),
            0.,
        );
    }
}
