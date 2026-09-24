//! Loading and querying a Seismic index.

use std::io::{self, Result};
use std::path::Path;

use half::f16;
use vectorium::{Distance, IndexSerializer, SparseVectorView};

use seismic::ScalarInvertedIndex;

use super::{SeismicSearchParams, SEISMIC_FORMAT, SEISMIC_INDEX_FILE};
use crate::base::{ImpactValue, TermIndex};
use crate::manifest::{read_manifest, IndexKind};
use crate::search::ScoredDocument;

enum Inner {
    U16(ScalarInvertedIndex<u16, f32, f16>),
    U32(ScalarInvertedIndex<u32, f32, f16>),
}

/// A loaded Seismic index (see [`super`]).
pub struct SeismicSearcher {
    inner: Inner,
}

fn invalid(msg: String) -> io::Error {
    io::Error::new(io::ErrorKind::InvalidData, msg)
}

macro_rules! run_search {
    ($index:expr, $C:ty, $query:expr, $k:expr, $params:expr) => {{
        let index = $index;
        let dim = index.dim();
        let mut terms: Vec<(TermIndex, ImpactValue)> = $query
            .iter()
            .copied()
            .filter(|&(t, v)| t < dim && v > 0.)
            .collect();
        terms.sort_unstable_by_key(|&(t, _)| t);
        terms.dedup_by_key(|&mut (t, _)| t);
        let components: Vec<$C> = terms.iter().map(|&(t, _)| t as $C).collect();
        let values: Vec<f32> = terms.iter().map(|&(_, v)| v).collect();
        if components.is_empty() {
            return Vec::new();
        }
        index
            .search(
                SparseVectorView::new(&components, &values),
                $k,
                $params.query_cut,
                $params.heap_factor,
                $params.n_knn,
                false,
            )
            .into_iter()
            .map(|scored| ScoredDocument {
                docid: scored.vector,
                score: scored.distance.distance(),
            })
            .collect()
    }};
}

impl SeismicSearcher {
    /// Loads a Seismic index directory written by [`super::convert_to_seismic`].
    pub fn load(path: &Path) -> Result<Self> {
        let manifest = read_manifest(path)?.ok_or_else(|| {
            invalid(format!(
                "{} has no manifest.json: not a Seismic index directory",
                path.display()
            ))
        })?;
        if manifest.index_kind != IndexKind::Seismic {
            return Err(invalid(format!(
                "{} is a {} index, not a Seismic index",
                path.display(),
                manifest.index_kind
            )));
        }
        let codecs = manifest.builder.codecs.unwrap_or_default();
        let mut parts = codecs.split_whitespace();
        let format = parts.next().unwrap_or("");
        if format != SEISMIC_FORMAT {
            return Err(invalid(format!(
                "Seismic index at {} was built with format '{}', but this build reads '{}'. \
                 Seismic indices cannot be migrated: rebuild it with to_seismic / convert_to_seismic.",
                path.display(),
                format,
                SEISMIC_FORMAT
            )));
        }

        let file = path.join(SEISMIC_INDEX_FILE);
        let file = file.to_string_lossy();
        let load_err = |e| invalid(format!("Failed to load Seismic index {}: {:?}", file, e));
        let inner = match parts.next() {
            Some("components=u16") => {
                Inner::U16(ScalarInvertedIndex::load_index(&file).map_err(load_err)?)
            }
            Some("components=u32") => {
                Inner::U32(ScalarInvertedIndex::load_index(&file).map_err(load_err)?)
            }
            other => {
                return Err(invalid(format!(
                    "Unknown Seismic component type in manifest: {:?}",
                    other
                )))
            }
        };
        Ok(Self { inner })
    }

    /// Number of documents in the index.
    pub fn num_documents(&self) -> usize {
        match &self.inner {
            Inner::U16(index) => index.len(),
            Inner::U32(index) => index.len(),
        }
    }

    /// Approximate top-`k` search by dot product.
    ///
    /// Query terms outside the index vocabulary or with a non-positive
    /// weight are ignored; duplicate terms keep their first weight.
    pub fn search(
        &self,
        query: &[(TermIndex, ImpactValue)],
        k: usize,
        params: &SeismicSearchParams,
    ) -> Vec<ScoredDocument> {
        match &self.inner {
            Inner::U16(index) => run_search!(index, u16, query, k, params),
            Inner::U32(index) => run_search!(index, u32, query, k, params),
        }
    }
}
