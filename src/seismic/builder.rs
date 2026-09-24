//! Conversion of an impact index into a Seismic index.

use std::io::{self, Result};
use std::path::Path;

use half::f16;
use indicatif::{ProgressBar, ProgressStyle};
use vectorium::{
    DatasetGrowable, DotProduct, IndexSerializer, ScalarSparseDataset, ScalarSparseDatasetGrowable,
    ScalarSparseQuantizer, SparseVectorView,
};

use seismic::ScalarInvertedIndex;

use super::{SeismicConfig, SEISMIC_FORMAT, SEISMIC_INDEX_FILE, U16_MAX_DIM};
use crate::index::SparseIndexView;
use crate::manifest::{write_manifest, BuilderInfo, IndexKind};

/// Document-oriented (CSR) copy of an inverted index, with components
/// already narrowed to Seismic's component type `C` (halves the memory of
/// the copy for `u16`).
struct ForwardCsr<C> {
    offsets: Vec<usize>,
    components: Vec<C>,
    values: Vec<f32>,
}

fn progress(len: usize, msg: &'static str) -> ProgressBar {
    let pb = ProgressBar::new(len as u64);
    pb.set_style(
        ProgressStyle::default_bar()
            .template("{spinner:.green} [{elapsed_precise}] {msg} [{bar:40.cyan/blue}] {pos}/{len} ({eta})")
            .progress_chars("=> "),
    );
    pb.set_message(msg);
    pb
}

/// Transposes the inverted index in two passes (count, then fill).
///
/// Terms are visited in increasing order, so each document's components
/// come out strictly increasing, as Seismic requires.
fn transpose<C: Copy + Default + TryFrom<usize>>(index: &dyn SparseIndexView) -> ForwardCsr<C> {
    let num_docs = (index.max_doc_id() + 1) as usize;
    let num_terms = index.len();

    let mut offsets = vec![0usize; num_docs + 1];
    let pb = progress(num_terms, "Counting postings");
    for term_ix in 0..num_terms {
        for posting in index.iterator(term_ix) {
            offsets[posting.docid as usize + 1] += 1;
        }
        pb.inc(1);
    }
    pb.finish();
    for i in 0..num_docs {
        offsets[i + 1] += offsets[i];
    }

    let nnz = offsets[num_docs];
    let mut components = vec![C::default(); nnz];
    let mut values = vec![0f32; nnz];
    let mut cursor = offsets[..num_docs].to_vec();
    let pb = progress(num_terms, "Transposing");
    for term_ix in 0..num_terms {
        let component = C::try_from(term_ix)
            .unwrap_or_else(|_| panic!("term {} does not fit the component type", term_ix));
        for posting in index.iterator(term_ix) {
            let pos = &mut cursor[posting.docid as usize];
            components[*pos] = component;
            values[*pos] = posting.value;
            *pos += 1;
        }
        pb.inc(1);
    }
    pb.finish();

    ForwardCsr {
        offsets,
        components,
        values,
    }
}

macro_rules! build_and_save {
    ($C:ty, $index:expr, $dim:expr, $cfg:expr, $path:expr) => {{
        let csr = transpose::<$C>($index);
        // Values are converted to f16 as they are pushed, so no full f32
        // copy of the collection is made besides the CSR one.
        let quantizer = ScalarSparseQuantizer::<$C, f32, f16, DotProduct>::new($dim, $dim);
        let mut data: ScalarSparseDatasetGrowable<$C, f32, f16, DotProduct> =
            DatasetGrowable::with_capacity(quantizer, csr.offsets.len() - 1);
        for doc in 0..csr.offsets.len() - 1 {
            let (start, end) = (csr.offsets[doc], csr.offsets[doc + 1]);
            data.push(SparseVectorView::new(
                &csr.components[start..end],
                &csr.values[start..end],
            ));
        }
        drop(csr);

        let dataset: ScalarSparseDataset<$C, f32, f16, DotProduct> = data.into();
        let index = ScalarInvertedIndex::<$C, f32, f16>::build(dataset, $cfg.to_configuration());
        index
            .save_index(&$path.to_string_lossy())
            .map_err(|e| io::Error::new(io::ErrorKind::Other, format!("{:?}", e)))
    }};
}

/// Builds a Seismic index from `index` into the directory `output`.
///
/// The whole collection is held in memory while building (a CSR copy, then
/// Seismic's own f16 forward index), as Seismic keeps its forward index in
/// RAM: peak is about `nnz * (sizeof(C) + 4 + sizeof(C) + 2)` bytes plus
/// Seismic's own build structures.
pub fn convert_to_seismic(
    index: &dyn SparseIndexView,
    output: &Path,
    config: &SeismicConfig,
) -> Result<()> {
    std::fs::create_dir_all(output)?;
    let dim = index.len().max(1);
    let index_path = output.join(SEISMIC_INDEX_FILE);

    let component_type = if dim <= U16_MAX_DIM {
        build_and_save!(u16, index, dim, config, index_path)?;
        "u16"
    } else {
        if dim > u32::MAX as usize {
            return Err(io::Error::new(
                io::ErrorKind::InvalidInput,
                format!("Vocabulary too large for Seismic ({} terms)", dim),
            ));
        }
        build_and_save!(u32, index, dim, config, index_path)?;
        "u32"
    };

    let config_json = serde_json::to_string(config)
        .map_err(|e| io::Error::new(io::ErrorKind::Other, e.to_string()))?;
    write_manifest(
        output,
        IndexKind::Seismic,
        BuilderInfo::new().with_codecs(format!(
            "{} components={} values=f16 config={}",
            SEISMIC_FORMAT, component_type, config_json
        )),
    )
}
