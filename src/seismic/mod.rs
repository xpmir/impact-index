//! Seismic approximate retrieval (Bruch et al., SIGIR 2024), via the
//! upstream [`seismic`] crate.
//!
//! Like [`crate::bmp`], a Seismic index is a *separate* artefact built from
//! an existing impact index: it stores the learned impact values as-is
//! (dot-product scoring only), so it cannot serve query-time scoring models
//! (BM25, LM) nor structured queries. Search is **approximate** -- results
//! depend on the build parameters (`n_postings`, `summary_energy`, ...) and
//! on the query-time `query_cut` / `heap_factor`.
//!
//! ## On-disk layout
//!
//! A Seismic index is a directory containing:
//! - `index.seismic`: the serialized [`seismic`] index (bincode, owned by
//!   the upstream crate)
//! - `manifest.json`: an [`IndexKind::Seismic`](crate::manifest::IndexKind)
//!   manifest; `builder.codecs` records [`SEISMIC_FORMAT`] and the
//!   component type
//!
//! Seismic document ids are positional; empty documents are pushed as empty
//! vectors so that the positional id *is* the impact-index [`DocId`](crate::base::DocId).
//!
//! ## Versioning
//!
//! The `index.seismic` bytes belong to the pinned upstream crate and cannot
//! be migrated: when [`SEISMIC_FORMAT`] changes (i.e. the pinned revision is
//! bumped), loading an older directory fails and the index must be rebuilt
//! with [`convert_to_seismic`].

mod builder;
mod searcher;

pub use builder::convert_to_seismic;
pub use searcher::SeismicSearcher;

use serde::{Deserialize, Serialize};

use seismic::configurations::{
    BlockingStrategy, ClusteringAlgorithm, Configuration, KnnConfiguration, PruningStrategy,
    SummarizationStrategy,
};

/// Name of the serialized Seismic index within the directory.
pub const SEISMIC_INDEX_FILE: &str = "index.seismic";

/// Identifies the on-disk format of `index.seismic`: bump together with the
/// pinned `seismic` git revision in `Cargo.toml`.
pub const SEISMIC_FORMAT: &str = "seismic@3c267137";

/// Largest vocabulary stored with `u16` components.
pub(crate) const U16_MAX_DIM: usize = u16::MAX as usize + 1;

/// Build parameters of a Seismic index.
///
/// Defaults follow the upstream recommendations for SPLADE on MS MARCO.
#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct SeismicConfig {
    /// Average number of postings kept per term (global threshold pruning).
    pub n_postings: usize,
    /// Maximum posting list length, as a factor of `n_postings`.
    pub max_fraction: f32,
    /// Number of blocks (k-means centroids) per list, as a fraction of its length.
    pub centroid_fraction: f32,
    /// Minimum number of postings per block.
    pub min_cluster_size: usize,
    /// Number of top document components used when clustering.
    pub doc_cut: usize,
    /// Fraction of the L1 mass kept in each block summary.
    pub summary_energy: f32,
    /// Number of nearest neighbours stored per document (0 = no kNN graph).
    pub knn: usize,
}

impl Default for SeismicConfig {
    fn default() -> Self {
        Self {
            n_postings: 6000,
            max_fraction: 1.5,
            centroid_fraction: 0.1,
            min_cluster_size: 2,
            doc_cut: 15,
            summary_energy: 0.4,
            knn: 0,
        }
    }
}

impl SeismicConfig {
    pub(crate) fn to_configuration(&self) -> Configuration {
        Configuration::default()
            .pruning_strategy(PruningStrategy::GlobalThreshold {
                n_postings: self.n_postings,
                max_fraction: self.max_fraction,
            })
            .blocking_strategy(BlockingStrategy::RandomKmeans {
                centroid_fraction: self.centroid_fraction,
                min_cluster_size: self.min_cluster_size,
                clustering_algorithm: ClusteringAlgorithm::RandomKmeansInvertedIndexApprox {
                    doc_cut: self.doc_cut,
                },
            })
            .summarization_strategy(SummarizationStrategy::EnergyPreserving {
                summary_energy: self.summary_energy,
            })
            .knn(KnnConfiguration::new(self.knn, None))
    }
}

/// Query-time parameters of a Seismic search.
#[derive(Debug, Clone, Copy)]
pub struct SeismicSearchParams {
    /// Only the `query_cut` highest-weighted query terms are traversed.
    pub query_cut: usize,
    /// Blocks whose summary score is below `heap_factor` times the current
    /// k-th score are skipped (1.0 = no approximation from summaries).
    pub heap_factor: f32,
    /// Number of kNN neighbours used to refine the results (needs `knn > 0`
    /// at build time).
    pub n_knn: usize,
}

impl Default for SeismicSearchParams {
    fn default() -> Self {
        Self {
            query_cut: 10,
            heap_factor: 0.7,
            n_knn: 0,
        }
    }
}
