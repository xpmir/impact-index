use log::debug;
use pyo3::PyClassInitializer;
use pyo3::{
    pyclass, pymethods, pymodule,
    types::{PyAnyMethods, PyDict, PyModule, PyModuleMethods},
    Bound, Py, PyAny, PyRef, PyResult, Python,
};

use std::collections::HashMap;
use std::future::IntoFuture;
use std::path::{Path, PathBuf};
use std::str::FromStr;
use std::sync::Arc;
use tokio::sync::Mutex;
use tokio::task;

use crate::bow::BOWIndexBuilder;
use crate::builder::BuilderOptions;
use crate::compress;
use crate::compress::docid::{BitPackingCompressor, EliasFanoCompressor, PForCompressor};
use crate::compress::CompressionTransform;
use crate::docmeta::DocMetadata;
use crate::docstore;
use crate::scoring::bm25::BM25Scoring;
use crate::scoring::ScoredIndex;
use crate::transforms::split::SplitIndexTransform;
use crate::vocab::analyzer::TextAnalyzer;
use crate::vocab::stemmer::{PorterStemmer, SnowballStemmer};

use bmp::index::posting_list::PostingListIterator;
use bmp::query::MAX_TERM_WEIGHT;
use bmp::search::b_search_verbose;

use crate::base::load_index;
use crate::base::{DocId, ImpactValue, PostingValue, TermIndex};
use crate::index::SparseIndex;
use crate::search::maxscore::{search_maxscore, MaxScoreOptions};
use crate::transforms::IndexTransform;
use crate::{
    base::TermImpactIterator, builder::Indexer as SparseIndexer, search::wand::search_wand,
};

use numpy::{PyArray1, PyArrayMethods};
#[cfg(feature = "stub-gen")]
use pyo3_stub_gen::derive::*;

/// Type alias for search functions
type SearchFn = fn(
    index: &dyn SparseIndex,
    query: &HashMap<TermIndex, ImpactValue>,
    top_k: usize,
) -> Vec<crate::search::ScoredDocument>;

/// A single term impact: a (document ID, impact value) pair.
#[cfg_attr(feature = "stub-gen", gen_stub_pyclass)]
#[pyclass(name = "TermImpact")]
struct PyTermImpact {
    /// The impact value.
    #[pyo3(get)]
    value: ImpactValue,

    /// The document identifier.
    #[pyo3(get)]
    docid: DocId,
}

/// A document with its retrieval score, returned by search methods.
#[cfg_attr(feature = "stub-gen", gen_stub_pyclass)]
#[pyclass]
pub struct PyScoredDocument {
    /// The relevance score.
    #[pyo3(get)]
    score: ImpactValue,

    /// The document identifier.
    #[pyo3(get)]
    docid: DocId,
}

/// Iterator over term impacts in a posting list.
///
/// Yields TermImpact objects with (docid, value) pairs.
/// Also provides metadata: length(), max_value(), max_doc_id().
#[cfg_attr(feature = "stub-gen", gen_stub_pyclass)]
#[pyclass(name = "SparseIndexIterator", unsendable)]
struct PySparseIndexIterator {
    // Use dead code to ensure we have a valid index when iterating
    #[allow(dead_code)]
    index: Arc<Box<dyn SparseIndex>>,
    iter: TermImpactIterator<'static>,
}

#[cfg_attr(feature = "stub-gen", gen_stub_pymethods)]
#[pymethods]
impl PySparseIndexIterator {
    fn __next__(&mut self) -> PyResult<Option<PyTermImpact>> {
        if let Some(r) = self.iter.next() {
            return Ok(Some(PyTermImpact {
                value: r.value,
                docid: r.docid,
            }));
        }
        Ok(None)
    }

    fn __iter__(slf: PyRef<Self>) -> PyRef<Self> {
        slf
    }

    /// Returns the total number of postings for this term.
    fn length(&self) -> usize {
        self.iter.length()
    }

    /// Returns the maximum impact value for this term.
    fn max_value(&self) -> ImpactValue {
        self.iter.max_value()
    }

    /// Returns the maximum document ID in this posting list.
    fn max_doc_id(&self) -> DocId {
        self.iter.max_doc_id()
    }
}

/// Base class for index views.
#[cfg_attr(feature = "stub-gen", gen_stub_pyclass)]
#[pyclass(subclass, name = "IndexView")]
pub struct PyIndexView {}

/// A loaded sparse index that supports searching and iteration.
///
/// Use ``Index.load(folder, in_memory)`` to load an existing index,
/// or build one with ``IndexBuilder``.
///
/// Example:
///
/// ```python,ignore
/// import impact_index
/// index = impact_index.Index.load("/path/to/index", in_memory=True)
/// results = index.search_wand({42: 1.5, 100: 0.8}, top_k=10)
/// for doc in results:
///     print(doc.docid, doc.score)
/// ```
#[cfg_attr(feature = "stub-gen", gen_stub_pyclass)]
#[pyclass(name = "Index", extends=PyIndexView)]
pub struct PySparseIndex {
    index: Arc<Box<dyn SparseIndex>>,
}

impl PySparseIndex {
    fn _search(
        &self,
        py: Python<'_>,
        py_query: &Bound<'_, PyDict>,
        top_k: usize,
        search_fn: SearchFn,
    ) -> PyResult<Py<PyAny>> {
        let query: HashMap<usize, ImpactValue> = py_query.extract()?;
        let results = search_fn(&**self.index, &query, top_k);

        let v: Vec<PyScoredDocument> = results
            .iter()
            .map(|r| PyScoredDocument {
                docid: r.docid,
                score: r.score,
            })
            .collect();
        Ok(pyo3::IntoPyObject::into_pyobject(v, py)?.into())
    }

    fn _aio_search<'a>(
        &self,
        py: Python<'a>,
        py_query: &Bound<'_, PyDict>,
        top_k: usize,
        search_fn: SearchFn,
    ) -> PyResult<Bound<'a, PyAny>> {
        let index = self.index.clone();

        let query: HashMap<usize, ImpactValue> = py_query.extract()?;

        pyo3_async_runtimes::tokio::future_into_py(py, async move {
            let results = task::spawn(async move { search_fn(&**index, &query, top_k) })
                .into_future()
                .await
                .expect("Error while searching");

            let v: Vec<PyScoredDocument> = results
                .iter()
                .map(|r| PyScoredDocument {
                    docid: r.docid,
                    score: r.score,
                })
                .collect();
            Ok(v)
        })
    }
}

#[cfg_attr(feature = "stub-gen", gen_stub_pymethods)]
#[pymethods]
impl PySparseIndex {
    /// Returns an iterator over the posting list for the given term index.
    fn postings(&self, term: TermIndex) -> PyResult<PySparseIndexIterator> {
        Ok(PySparseIndexIterator {
            index: self.index.clone(),
            // TODO: ugly but works since index is up here
            iter: unsafe { extend_lifetime(self.index.block_iterator(term)) },
        })
    }

    /// Returns the number of distinct terms in the index.
    fn num_postings(&self) -> usize {
        self.index.len()
    }

    /// Search the index (deprecated, use search_wand instead).
    fn search(
        &self,
        py: Python<'_>,
        py_query: &Bound<'_, PyDict>,
        top_k: usize,
    ) -> PyResult<Py<PyAny>> {
        self._search(py, py_query, top_k, search_wand)
    }

    /// Search using the WAND algorithm.
    fn search_wand(
        &self,
        py: Python<'_>,
        py_query: &Bound<'_, PyDict>,
        top_k: usize,
    ) -> PyResult<Py<PyAny>> {
        self._search(py, py_query, top_k, search_wand)
    }

    /// Search using the MaxScore algorithm.
    fn search_maxscore(
        &self,
        py: Python<'_>,
        py_query: &Bound<'_, PyDict>,
        top_k: usize,
    ) -> PyResult<Py<PyAny>> {
        self._search(py, py_query, top_k, |index, query, top_k| {
            let options = MaxScoreOptions::default();
            search_maxscore(index, query, top_k, options)
        })
    }

    /// Async version of search_wand.
    fn aio_search_wand<'a>(
        &self,
        py: Python<'a>,
        py_query: &Bound<'_, PyDict>,
        top_k: usize,
    ) -> PyResult<Bound<'a, PyAny>> {
        self._aio_search(py, py_query, top_k, search_wand)
    }

    /// Async version of search_maxscore.
    fn aio_search_maxscore<'a>(
        &self,
        py: Python<'a>,
        py_query: &Bound<'_, PyDict>,
        top_k: usize,
    ) -> PyResult<Bound<'a, PyAny>> {
        self._aio_search(py, py_query, top_k, |index, query, top_k| {
            let options = MaxScoreOptions::default();
            search_maxscore(index, query, top_k, options)
        })
    }

    /// Convert the index into BMP format.
    fn to_bmp(&self, output: &str, bsize: usize, compress_range: bool) -> PyResult<()> {
        let index = self.index.clone();
        let output_path = PathBuf::from_str(output).expect("cannot use path");
        index
            .convert_to_bmp(&output_path, bsize, compress_range)
            .expect("Failed to write the BMP file");
        Ok(())
    }

    /// Convert into a BMP index using streaming (memory-efficient) method.
    fn to_bmp_streaming(&self, output: &str, bsize: usize, compress_range: bool) -> PyResult<()> {
        let index = self.index.clone();
        let output_path = PathBuf::from_str(output).expect("cannot use path");
        index
            .convert_to_bmp_streaming(&output_path, bsize, compress_range)
            .expect("Failed to write the BMP file");
        Ok(())
    }

    /// Get the text analyzer for this index (stemmer, stop words, vocabulary).
    ///
    /// Get the text analyzer for this index (if available).
    fn analyzer(&self) -> PyResult<PyTextAnalyzer> {
        let source = self.index.source_path().ok_or_else(|| {
            pyo3::exceptions::PyRuntimeError::new_err(
                "Index has no source path — cannot load analyzer.",
            )
        })?;
        PyTextAnalyzer::from_index(source.to_str().unwrap())
    }

    /// Create a scored index that applies a scoring model to raw postings.
    ///
    /// Document metadata (lengths) is loaded automatically from the index.
    fn with_scoring(&self, py: Python<'_>, scoring: &PyBM25Scoring) -> PyResult<Py<PyAny>> {
        let doc_meta = crate::index::SparseIndex::doc_meta(&**self.index).ok_or_else(|| {
            pyo3::exceptions::PyRuntimeError::new_err(
                "Index has no document metadata. Build with BOWIndexBuilder for BM25.",
            )
        })?;
        // Clone into Arc for ScoredIndex
        let doc_meta = Arc::new(crate::docmeta::DocMetadata::from_lengths(
            doc_meta.doc_lengths.clone(),
        ));

        let model = Box::new(BM25Scoring::with_params(scoring.k1, scoring.b));
        let scored = ScoredIndex::new(self.index.clone(), doc_meta, model);
        let scored_box: Arc<Box<dyn SparseIndex>> = Arc::new(Box::new(scored));

        let base = PyClassInitializer::from(PyIndexView {});
        let sub = base.add_subclass(PyScoredIndex { index: scored_box });
        Ok(Py::new(py, sub)?.into_any())
    }

    /// Compress the index with SIMD bitpacking and quantization.
    ///
    /// Creates a compressed index at the given path with block-max metadata
    /// for efficient pruning during search. Returns the compressed index.
    ///
    /// Args:
    ///     output_folder: Directory to write the compressed index to.
    ///     block_size: Number of postings per block (default: 128).
    ///     nbits: Quantization bits for impact values. Default 0 means
    ///         lossless integer bitpacking (best for BM25 with integer TF).
    ///         Set to 8 or 16 for quantized float compression.
    ///     in_memory: Load the compressed index in memory (default: True).
    #[pyo3(signature = (output_folder, block_size=128, nbits=0, in_memory=true))]
    fn compress(
        &self,
        py: Python<'_>,
        output_folder: &str,
        block_size: usize,
        nbits: u32,
        in_memory: bool,
    ) -> PyResult<Py<PyAny>> {
        let impacts_factory: Box<dyn compress::ImpactCompressorFactory> = if nbits == 0 {
            // Lossless integer bitpacking (best for BM25 integer TF counts)
            Box::new(compress::impact::BitPackedIntCompressor {})
        } else {
            // Quantized float compression
            Box::new(compress::impact::QuantizedBitPackedFactory { nbits })
        };
        let transform = CompressionTransform {
            max_block_size: block_size,
            doc_ids_compressor_factory: Box::new(PForCompressor {}),
            impacts_compressor_factory: impacts_factory,
        };
        let path = Path::new(output_folder);
        transform
            .process(path, &**self.index)
            .map_err(|e| pyo3::exceptions::PyIOError::new_err(format!("{}", e)))?;

        // Let the source index save its auxiliary data to the compressed dir
        self.index
            .save_auxiliary(path)
            .map_err(|e| pyo3::exceptions::PyIOError::new_err(format!("{}", e)))?;

        let base = PyClassInitializer::from(PyIndexView {});
        let sub = base.add_subclass(PySparseIndex {
            index: Arc::new(load_index(path, in_memory)),
        });
        Ok(Py::new(py, sub)?.into_any())
    }

    /// Reorder documents by recursive graph bisection (BP), then compress.
    ///
    /// Renumbers document ids so that documents sharing many terms end up
    /// with nearby ids: posting-list gaps shrink (smaller/faster
    /// compressed index) and per-block impact maxima / minimum document
    /// lengths become skewed instead of near-uniform, which sharpens
    /// block-max and P1a `min_dl` pruning (see `optimizations.md`, P2).
    ///
    /// Reordering is transparent to search: results returned by
    /// ``search_maxscore``/``search_wand`` on a reordered index carry the
    /// ORIGINAL document ids (translated automatically), so no caller-side
    /// mapping is needed. ``reorder_map()`` exposes the raw internal
    /// mapping for advanced uses.
    ///
    /// Args:
    ///     output_folder: Directory to write the reordered, compressed
    ///         index to.
    ///     block_size: Number of postings per block (default: 128).
    ///     nbits: Quantization bits for impact values (see ``compress``).
    ///     in_memory: Load the resulting index in memory (default: True).
    ///     leaf_size: Stop recursing once a subtree has at most this many
    ///         documents (default: 64).
    ///     max_iters: Maximum swap iterations per recursion level
    ///         (default: 20).
    #[pyo3(signature = (output_folder, block_size=128, nbits=0, in_memory=true, leaf_size=64, max_iters=20))]
    fn reorder(
        &self,
        py: Python<'_>,
        output_folder: &str,
        block_size: usize,
        nbits: u32,
        in_memory: bool,
        leaf_size: usize,
        max_iters: usize,
    ) -> PyResult<Py<PyAny>> {
        let impacts_factory: Box<dyn compress::ImpactCompressorFactory> = if nbits == 0 {
            Box::new(compress::impact::BitPackedIntCompressor {})
        } else {
            Box::new(compress::impact::QuantizedBitPackedFactory { nbits })
        };
        let sink = Box::new(CompressionTransform {
            max_block_size: block_size,
            doc_ids_compressor_factory: Box::new(PForCompressor {}),
            impacts_compressor_factory: impacts_factory,
        });
        let transform = crate::transforms::reorder::ReorderTransform {
            sink,
            options: crate::transforms::reorder::BpOptions {
                leaf_size,
                max_iters,
                ..Default::default()
            },
        };
        let path = Path::new(output_folder);
        transform
            .process(path, &**self.index)
            .map_err(|e| pyo3::exceptions::PyIOError::new_err(format!("{}", e)))?;

        // `ReorderTransform` already wrote the (permuted) docmeta itself;
        // copy vocab/analyzer only -- `save_auxiliary` would overwrite
        // docmeta with the *original*, un-reordered lengths.
        self.index
            .save_vocab_and_analyzer(path)
            .map_err(|e| pyo3::exceptions::PyIOError::new_err(format!("{}", e)))?;

        let base = PyClassInitializer::from(PyIndexView {});
        let sub = base.add_subclass(PySparseIndex {
            index: Arc::new(load_index(path, in_memory)),
        });
        Ok(Py::new(py, sub)?.into_any())
    }

    /// Document-id reorder map (`new_docid -> original_docid`), if this
    /// index was produced by ``reorder()`` (or a ``ReorderTransform``).
    ///
    /// Search results are already translated to original document ids
    /// automatically, so most callers never need this. It exposes the raw
    /// internal permutation (``reorder_map()[internal_docid] ==
    /// original_docid``) for advanced uses such as interpreting raw
    /// posting iterators. Returns ``None`` for an index that was never
    /// reordered.
    fn reorder_map(&self) -> Option<Vec<DocId>> {
        self.index.reorder_map().cloned()
    }

    /// Load an index from a directory.
    #[staticmethod]
    fn load(py: Python<'_>, folder: &str, in_memory: bool) -> PyResult<Py<PyAny>> {
        let base = PyClassInitializer::from(PyIndexView {});
        let sub = base.add_subclass(PySparseIndex {
            index: Arc::new(load_index(Path::new(folder), in_memory)),
        });
        Ok(Py::new(py, sub)?.into_any())
    }

    /// Migrate an index directory to the format version this library expects.
    ///
    /// Raised by ``Index.load`` (and internal loaders) when an index
    /// directory's ``manifest.json`` records an older ``format_version``
    /// than this library supports: call ``Index.update(path)`` first, then
    /// retry loading.
    ///
    /// Args:
    ///     path: Directory containing the index to migrate.
    ///     dest: If given, write the migrated index there instead of
    ///         migrating in place (``path`` is left untouched).
    ///
    /// Returns:
    ///     The directory holding the migrated index (``dest`` if given,
    ///     otherwise ``path``).
    #[staticmethod]
    #[pyo3(signature = (path, dest=None))]
    fn update(path: &str, dest: Option<&str>) -> PyResult<String> {
        let dest_path = dest.map(PathBuf::from);
        let result_path = crate::manifest::update_index(Path::new(path), dest_path.as_deref())
            .map_err(|e| pyo3::exceptions::PyIOError::new_err(e.to_string()))?;
        Ok(result_path.to_string_lossy().into_owned())
    }
}

/// Configuration options for IndexBuilder.
#[cfg_attr(feature = "stub-gen", gen_stub_pyclass)]
#[pyclass(name = "BuilderOptions")]
struct PyBuilderOptions(BuilderOptions);

#[cfg_attr(feature = "stub-gen", gen_stub_pymethods)]
#[pymethods]
impl PyBuilderOptions {
    #[new]
    fn new() -> Self {
        PyBuilderOptions {
            0: BuilderOptions::default(),
        }
    }
    #[getter]
    fn checkpoint_frequency(&self) -> DocId {
        self.0.checkpoint_frequency
    }

    #[setter]
    fn set_checkpoint_frequency(&mut self, value: DocId) {
        self.0.checkpoint_frequency = value;
    }

    #[getter]
    fn in_memory_threshold(&self) -> usize {
        self.0.in_memory_threshold
    }

    #[setter]
    fn set_in_memory_threshold(&mut self, value: usize) {
        self.0.in_memory_threshold = value;
    }
}

/// Builds a sparse index from document impact vectors.
#[cfg_attr(feature = "stub-gen", gen_stub_pyclass)]
#[pyclass(name = "IndexBuilder")]
pub struct PyIndexBuilder {
    inner: Arc<Mutex<IndexerEnum>>,
}

/// Type-erased wrapper around generic `Indexer<V>`.
enum IndexerEnum {
    F32(SparseIndexer<f32>),
    F64(SparseIndexer<f64>),
    F16(SparseIndexer<half::f16>),
    BF16(SparseIndexer<half::bf16>),
    I32(SparseIndexer<i32>),
    I64(SparseIndexer<i64>),
}

/// Helper to add a document with f32 values to a typed indexer.
fn add_to_indexer<V: PostingValue>(
    indexer: &mut SparseIndexer<V>,
    docid: DocId,
    terms: &Bound<'_, PyArray1<TermIndex>>,
    values: &Bound<'_, PyArray1<f32>>,
) -> PyResult<()> {
    let terms_array = unsafe { terms.as_array() };
    let values_f32 = unsafe { values.as_array() };

    let converted: Vec<V> = values_f32
        .iter()
        .map(|&v| convert_f64_to_posting_value::<V>(v as f64))
        .collect();
    let values_array = ndarray::Array::from_vec(converted);
    indexer.add(docid, &terms_array, &values_array)?;
    Ok(())
}

/// Convert an f64 to a PostingValue type.
fn convert_f64_to_posting_value<V: PostingValue>(v: f64) -> V {
    use std::any::TypeId;
    let id = TypeId::of::<V>();
    unsafe {
        if id == TypeId::of::<f32>() {
            let val = v as f32;
            *(&val as *const f32 as *const V)
        } else if id == TypeId::of::<f64>() {
            *(&v as *const f64 as *const V)
        } else if id == TypeId::of::<half::f16>() {
            let val = half::f16::from_f64(v);
            *(&val as *const half::f16 as *const V)
        } else if id == TypeId::of::<half::bf16>() {
            let val = half::bf16::from_f64(v);
            *(&val as *const half::bf16 as *const V)
        } else if id == TypeId::of::<i32>() {
            let val = v as i32;
            *(&val as *const i32 as *const V)
        } else if id == TypeId::of::<i64>() {
            let val = v as i64;
            *(&val as *const i64 as *const V)
        } else {
            panic!("Unknown PostingValue type")
        }
    }
}

unsafe fn extend_lifetime<'b>(r: TermImpactIterator<'b>) -> TermImpactIterator<'static> {
    std::mem::transmute::<TermImpactIterator<'b>, TermImpactIterator<'static>>(r)
}

// gen_stub_pymethods skipped: PyArray1<usize> not supported
// https://github.com/Jij-Inc/pyo3-stub-gen/issues/97
// gen_stub_pymethods skipped: (Self, Parent) return in #[new] unsupported
#[pymethods]
impl PyIndexBuilder {
    /// Create a new IndexBuilder.
    #[new]
    #[pyo3(signature = (folder, options=None, dtype=None))]
    fn new(
        folder: &str,
        options: Option<&PyBuilderOptions>,
        dtype: Option<&Bound<'_, PyAny>>,
    ) -> PyResult<Self> {
        let builder_options = match &options {
            Some(_options) => _options.0.clone(),
            None => BuilderOptions::default(),
        };

        let dtype_str: String = match dtype {
            None => "float32".to_string(),
            Some(obj) => {
                if let Ok(s) = obj.extract::<String>() {
                    s
                } else {
                    match obj.getattr("name") {
                        Ok(name) => name.extract::<String>()?,
                        Err(_) => obj.str()?.extract::<String>()?,
                    }
                }
            }
        };

        let path = Path::new(folder);
        let inner = match dtype_str.as_str() {
            "float32" | "f32" => IndexerEnum::F32(SparseIndexer::new(path, &builder_options)),
            "float64" | "f64" => IndexerEnum::F64(SparseIndexer::new(path, &builder_options)),
            "float16" | "f16" => IndexerEnum::F16(SparseIndexer::new(path, &builder_options)),
            "bfloat16" | "bf16" => IndexerEnum::BF16(SparseIndexer::new(path, &builder_options)),
            "int32" | "i32" => IndexerEnum::I32(SparseIndexer::new(path, &builder_options)),
            "int64" | "i64" => IndexerEnum::I64(SparseIndexer::new(path, &builder_options)),
            other => {
                return Err(pyo3::exceptions::PyValueError::new_err(format!(
                    "Unknown dtype '{}', expected one of: float32, float64, float16, bfloat16, int32, int64",
                    other
                )));
            }
        };

        Ok(PyIndexBuilder {
            inner: Arc::new(Mutex::new(inner)),
        })
    }

    /// Add a document to the index.
    fn add(
        &mut self,
        docid: DocId,
        terms: &Bound<'_, PyArray1<TermIndex>>,
        values: &Bound<'_, PyAny>,
    ) -> PyResult<()> {
        let py = values.py();
        let np = py.import("numpy")?;
        let values_f32: Bound<'_, PyArray1<f32>> = np
            .call_method1("asarray", (values,))?
            .call_method1("astype", ("float32",))?
            .extract()?;

        let mut inner = self.inner.blocking_lock();
        match &mut *inner {
            IndexerEnum::F32(indexer) => add_to_indexer(indexer, docid, terms, &values_f32),
            IndexerEnum::F64(indexer) => add_to_indexer(indexer, docid, terms, &values_f32),
            IndexerEnum::F16(indexer) => add_to_indexer(indexer, docid, terms, &values_f32),
            IndexerEnum::BF16(indexer) => add_to_indexer(indexer, docid, terms, &values_f32),
            IndexerEnum::I32(indexer) => add_to_indexer(indexer, docid, terms, &values_f32),
            IndexerEnum::I64(indexer) => add_to_indexer(indexer, docid, terms, &values_f32),
        }
    }

    /// Returns the document ID from the last checkpoint, or None.
    fn get_checkpoint_doc_id(&self) -> Option<DocId> {
        let inner = self.inner.blocking_lock();
        match &*inner {
            IndexerEnum::F32(indexer) => indexer.get_checkpoint_doc_id(),
            IndexerEnum::F64(indexer) => indexer.get_checkpoint_doc_id(),
            IndexerEnum::F16(indexer) => indexer.get_checkpoint_doc_id(),
            IndexerEnum::BF16(indexer) => indexer.get_checkpoint_doc_id(),
            IndexerEnum::I32(indexer) => indexer.get_checkpoint_doc_id(),
            IndexerEnum::I64(indexer) => indexer.get_checkpoint_doc_id(),
        }
    }

    /// Finalize the index and return a searchable Index.
    fn build(&mut self, py: Python<'_>, in_memory: bool) -> PyResult<Py<PyAny>> {
        let mut inner = self.inner.blocking_lock();

        macro_rules! build_index {
            ($indexer:expr) => {{
                let folder = $indexer.folder().to_path_buf();
                $indexer.build().expect("Error while building index");
                let base = PyClassInitializer::from(PyIndexView {});
                let index = $indexer.to_index(in_memory);
                let sub = base.add_subclass(PySparseIndex {
                    index: Arc::new(Box::new(index)),
                });
                Ok(Py::new(py, sub)?.into_any())
            }};
        }

        match &mut *inner {
            IndexerEnum::F32(indexer) => build_index!(indexer),
            IndexerEnum::F64(indexer) => build_index!(indexer),
            IndexerEnum::F16(indexer) => build_index!(indexer),
            IndexerEnum::BF16(indexer) => build_index!(indexer),
            IndexerEnum::I32(indexer) => build_index!(indexer),
            IndexerEnum::I64(indexer) => build_index!(indexer),
        }
    }
}

/// Base class for document ID compressors.
#[cfg_attr(feature = "stub-gen", gen_stub_pyclass)]
#[pyclass(subclass)]
pub struct PyDocIdCompressor {
    inner: Arc<Box<dyn compress::DocIdCompressorFactory>>,
}

/// Elias-Fano encoding for document ID compression.
#[cfg_attr(feature = "stub-gen", gen_stub_pyclass)]
#[pyclass(name="EliasFanoCompressor", extends=PyDocIdCompressor)]
pub struct PyEliasFanoCompressor {}

// gen_stub_pymethods skipped: (Self, Parent) return in #[new] unsupported
#[pymethods]
impl PyEliasFanoCompressor {
    #[new]
    fn new() -> (Self, PyDocIdCompressor) {
        (
            Self {},
            PyDocIdCompressor {
                inner: Arc::new(Box::new(EliasFanoCompressor {})),
            },
        )
    }
}

/// SIMD bitpacking for document ID compression (faster than Elias-Fano).
#[cfg_attr(feature = "stub-gen", gen_stub_pyclass)]
#[pyclass(name="BitPackingCompressor", extends=PyDocIdCompressor)]
pub struct PyBitPackingCompressor {}

// gen_stub_pymethods skipped: (Self, Parent) return in #[new] unsupported
#[pymethods]
impl PyBitPackingCompressor {
    #[new]
    fn new() -> (Self, PyDocIdCompressor) {
        (
            Self {},
            PyDocIdCompressor {
                inner: Arc::new(Box::new(BitPackingCompressor {})),
            },
        )
    }
}

/// PFOR-delta doc ID compressor (better compression than BitPacking with outliers).
#[cfg_attr(feature = "stub-gen", gen_stub_pyclass)]
#[pyclass(name="PForCompressor", extends=PyDocIdCompressor)]
pub struct PyPForCompressor {}

// gen_stub_pymethods skipped: (Self, Parent) return in #[new] unsupported
#[pymethods]
impl PyPForCompressor {
    #[new]
    fn new() -> (Self, PyDocIdCompressor) {
        (
            Self {},
            PyDocIdCompressor {
                inner: Arc::new(Box::new(PForCompressor {})),
            },
        )
    }
}

/// Base class for impact value compressors.
#[cfg_attr(feature = "stub-gen", gen_stub_pyclass)]
#[pyclass(name = "ImpactCompressor", subclass)]
pub struct PyImpactCompressorFactory {
    inner: Arc<Box<dyn compress::ImpactCompressorFactory>>,
}

/// Fixed-range quantizer for impact values.
#[cfg_attr(feature = "stub-gen", gen_stub_pyclass)]
#[pyclass(name="ImpactQuantizer", extends=PyImpactCompressorFactory)]
pub struct PyImpactQuantizer {}

// gen_stub_pymethods skipped: (Self, Parent) return in #[new] unsupported
#[pymethods]
impl PyImpactQuantizer {
    #[new]
    fn new(nbits: u32, min: ImpactValue, max: ImpactValue) -> (Self, PyImpactCompressorFactory) {
        (
            PyImpactQuantizer {},
            PyImpactCompressorFactory {
                inner: Arc::new(Box::new(compress::impact::Quantizer::new(nbits, min, max))),
            },
        )
    }
}

/// Auto-ranging quantizer that determines min/max from the index.
#[cfg_attr(feature = "stub-gen", gen_stub_pyclass)]
#[pyclass(name="GlobalImpactQuantizer", extends=PyImpactCompressorFactory)]
pub struct PyGlobalQuantizerFactory {}

// gen_stub_pymethods skipped: (Self, Parent) return in #[new] unsupported
#[pymethods]
impl PyGlobalQuantizerFactory {
    #[new]
    fn new(nbits: u32) -> (Self, PyImpactCompressorFactory) {
        (
            PyGlobalQuantizerFactory {},
            PyImpactCompressorFactory {
                inner: Arc::new(Box::new(compress::impact::GlobalQuantizerFactory { nbits })),
            },
        )
    }
}

/// SIMD bitpacked integer compressor for raw TF counts.
///
/// For BM25 indices with integer term frequencies, uses ~2-3 bits per value
/// (adaptive per block) vs 8 bits for quantized floats.
#[cfg_attr(feature = "stub-gen", gen_stub_pyclass)]
#[pyclass(name="BitPackedIntCompressor", extends=PyImpactCompressorFactory)]
pub struct PyBitPackedIntCompressor {}

#[pymethods]
impl PyBitPackedIntCompressor {
    #[new]
    fn new() -> (Self, PyImpactCompressorFactory) {
        (
            PyBitPackedIntCompressor {},
            PyImpactCompressorFactory {
                inner: Arc::new(Box::new(compress::impact::BitPackedIntCompressor {})),
            },
        )
    }
}

/// Quantized + adaptive bitpacked compressor for neural IR (SPLADE).
///
/// Quantizes float impacts to N-bit integers, then compresses with
/// adaptive SIMD bitpacking (~3-4 bits/value instead of fixed N bits).
#[cfg_attr(feature = "stub-gen", gen_stub_pyclass)]
#[pyclass(name="QuantizedBitPackedCompressor", extends=PyImpactCompressorFactory)]
pub struct PyQuantizedBitPackedCompressor {}

#[pymethods]
impl PyQuantizedBitPackedCompressor {
    #[new]
    fn new(nbits: u32) -> (Self, PyImpactCompressorFactory) {
        (
            PyQuantizedBitPackedCompressor {},
            PyImpactCompressorFactory {
                inner: Arc::new(Box::new(compress::impact::QuantizedBitPackedFactory {
                    nbits,
                })),
            },
        )
    }
}

trait PyTransformFactory: Send + Sync {
    fn create(&self, py: Python<'_>) -> Box<dyn IndexTransform>;
}

/// Base class for index transforms.
#[cfg_attr(feature = "stub-gen", gen_stub_pyclass)]
#[pyclass(subclass)]
pub struct PyTransform {
    factory: Box<dyn PyTransformFactory>,
}

#[cfg_attr(feature = "stub-gen", gen_stub_pymethods)]
#[pymethods]
impl PyTransform {
    /// Apply this transform to an index, writing the result to path.
    fn process(&self, py: Python<'_>, path: &str, index: &PySparseIndex) -> PyResult<()> {
        let transform = self.factory.create(py);
        let view = index.index.as_view();
        transform.process(Path::new(path), view)?;
        Ok(())
    }
}

struct PyCompressionTransformFactory {
    max_block_size: usize,
    doc_ids_compressor: Py<PyDocIdCompressor>,
    impacts_compressor: Py<PyImpactCompressorFactory>,
}

impl PyTransformFactory for PyCompressionTransformFactory {
    fn create(&self, py: Python<'_>) -> Box<dyn IndexTransform> {
        let impacts = self.impacts_compressor.bind(py).borrow();
        let docids = self.doc_ids_compressor.bind(py).borrow();
        Box::new(CompressionTransform {
            max_block_size: self.max_block_size,
            impacts_compressor_factory: (*impacts.inner).clone(),
            doc_ids_compressor_factory: (*docids.inner).clone(),
        })
    }
}

/// Transform that compresses an index using block-based encoding.
#[cfg_attr(feature = "stub-gen", gen_stub_pyclass)]
#[pyclass(extends=PyTransform, name="CompressionTransform")]
pub struct PyCompressionTransform {}

// gen_stub_pymethods skipped: (Self, Parent) return in #[new] unsupported
#[pymethods]
impl PyCompressionTransform {
    #[new]
    fn new(
        max_block_size: usize,
        doc_ids_compressor: Py<PyDocIdCompressor>,
        impacts_compressor: Py<PyImpactCompressorFactory>,
    ) -> (Self, PyTransform) {
        let factory = Box::new(PyCompressionTransformFactory {
            max_block_size,
            doc_ids_compressor,
            impacts_compressor,
        });
        (PyCompressionTransform {}, PyTransform { factory })
    }
}

struct PySplitIndexTransformFactory {
    sink: Py<PyTransform>,
    quantiles: Vec<f64>,
}
impl PyTransformFactory for PySplitIndexTransformFactory {
    fn create(&self, py: Python<'_>) -> Box<dyn IndexTransform> {
        let sink_ref = self.sink.bind(py).borrow();
        let sink = sink_ref.factory.create(py);
        Box::new(SplitIndexTransform {
            sink,
            quantiles: self.quantiles.clone(),
        })
    }
}

/// Transform that splits posting lists by impact quantiles.
#[cfg_attr(feature = "stub-gen", gen_stub_pyclass)]
#[pyclass(name="SplitIndexTransform", extends=PyTransform)]
struct PySplitIndexTransform {}

// gen_stub_pymethods skipped: (Self, Parent) return in #[new] unsupported
#[pymethods]
impl PySplitIndexTransform {
    #[new]
    fn new(quantiles: Vec<f64>, sink: Py<PyTransform>) -> (Self, PyTransform) {
        let factory = Box::new(PySplitIndexTransformFactory { sink, quantiles });
        (PySplitIndexTransform {}, PyTransform { factory })
    }
}

struct PyReorderTransformFactory {
    sink: Py<PyTransform>,
    options: crate::transforms::reorder::BpOptions,
}

impl PyTransformFactory for PyReorderTransformFactory {
    fn create(&self, py: Python<'_>) -> Box<dyn IndexTransform> {
        let sink_ref = self.sink.bind(py).borrow();
        let sink = sink_ref.factory.create(py);
        Box::new(crate::transforms::reorder::ReorderTransform {
            sink,
            options: self.options.clone(),
        })
    }
}

/// Transform that renumbers document ids by recursive graph bisection
/// (BP, see `optimizations.md` P2) before delegating to `sink` (typically
/// a `CompressionTransform`) to write the reordered postings.
///
/// Note: composing this generically (rather than through
/// `Index.reorder(...)`) does not copy vocab/analyzer auxiliary files --
/// callers that need those must copy them separately, the way
/// `Index.compress(...)` does. Permuted `docmeta` and `reorder_map.dat`
/// are always written directly by the transform itself.
#[cfg_attr(feature = "stub-gen", gen_stub_pyclass)]
#[pyclass(name="ReorderTransform", extends=PyTransform)]
struct PyReorderTransform {}

// gen_stub_pymethods skipped: (Self, Parent) return in #[new] unsupported
#[pymethods]
impl PyReorderTransform {
    #[new]
    #[pyo3(signature = (sink, leaf_size=64, max_iters=20, min_df=2, max_df_ratio=0.5))]
    fn new(
        sink: Py<PyTransform>,
        leaf_size: usize,
        max_iters: usize,
        min_df: usize,
        max_df_ratio: f64,
    ) -> (Self, PyTransform) {
        let factory = Box::new(PyReorderTransformFactory {
            sink,
            options: crate::transforms::reorder::BpOptions {
                leaf_size,
                max_iters,
                min_df,
                max_df_ratio,
            },
        });
        (PyReorderTransform {}, PyTransform { factory })
    }
}

/// BMP (Block-Max Pruning) Searcher for fast approximate search
#[cfg_attr(feature = "stub-gen", gen_stub_pyclass)]
#[pyclass(name = "BmpSearcher")]
pub struct PyBmpSearcher {
    index: bmp::index::inverted_index::Index,
    bfwd: bmp::index::forward_index::BlockForwardIndex,
}

#[cfg_attr(feature = "stub-gen", gen_stub_pymethods)]
#[pymethods]
impl PyBmpSearcher {
    #[new]
    fn new(path: &str) -> PyResult<Self> {
        let path_buf = PathBuf::from_str(path).expect("Invalid path");
        let (index, bfwd) = bmp::index::from_file(path_buf).map_err(|e| {
            pyo3::exceptions::PyIOError::new_err(format!("Failed to load BMP index: {}", e))
        })?;
        Ok(PyBmpSearcher { index, bfwd })
    }

    #[pyo3(signature = (query, k, alpha=1.0, beta=1.0))]
    fn search(
        &self,
        query: HashMap<String, f32>,
        k: usize,
        alpha: f32,
        beta: f32,
    ) -> PyResult<(Vec<String>, Vec<f32>)> {
        let max_tok_weight = query
            .iter()
            .map(|p| *p.1)
            .filter(|&value| !value.is_nan())
            .max_by(|a, b| a.partial_cmp(b).unwrap())
            .unwrap_or(1.0);

        let mut quant_query: HashMap<String, u32> = HashMap::new();
        let scale: f32 = MAX_TERM_WEIGHT as f32 / max_tok_weight;
        for (key, value) in &query {
            quant_query.insert(key.clone(), (value * scale).ceil() as u32);
        }

        let cursors: Vec<PostingListIterator> = quant_query
            .iter()
            .flat_map(|(token, freq)| self.index.get_cursor(token, *freq))
            .collect();
        let wrapped_cursors = vec![cursors; 1];

        let mut results = b_search_verbose(wrapped_cursors, &self.bfwd, k, alpha, beta, false);

        let doc_lexicon = self.index.documents();
        let mut docnos: Vec<String> = Vec::new();
        let mut scores: Vec<f32> = Vec::new();
        for r in results[0].to_sorted_vec().iter() {
            docnos.push(doc_lexicon[r.doc_id.0 as usize].clone());
            scores.push(r.score.into());
        }
        Ok((docnos, scores))
    }

    fn num_documents(&self) -> usize {
        self.index.num_documents()
    }
}

// --- DocumentStore Python bindings ---

#[cfg_attr(feature = "stub-gen", gen_stub_pyclass)]
#[pyclass(name = "Document")]
pub struct PyDocument {
    inner: docstore::Document,
}

// gen_stub_pymethods skipped: &[u8] not supported
// https://github.com/Jij-Inc/pyo3-stub-gen/issues/97
// gen_stub_pymethods skipped: (Self, Parent) return in #[new] unsupported
#[pymethods]
impl PyDocument {
    #[getter]
    fn internal_id(&self) -> u64 {
        self.inner.internal_id
    }

    #[getter]
    fn keys(&self) -> HashMap<String, String> {
        self.inner.keys.clone()
    }

    #[getter]
    fn content(&self) -> &[u8] {
        &self.inner.content
    }
}

#[cfg_attr(feature = "stub-gen", gen_stub_pyclass)]
#[pyclass(name = "DocumentStoreBuilder")]
pub struct PyDocumentStoreBuilder {
    builder: Option<docstore::builder::DocumentStoreBuilder>,
}

// gen_stub_pymethods skipped: &[u8] not supported
// https://github.com/Jij-Inc/pyo3-stub-gen/issues/97
// gen_stub_pymethods skipped: (Self, Parent) return in #[new] unsupported
#[pymethods]
impl PyDocumentStoreBuilder {
    /// Create a new DocumentStoreBuilder.
    ///
    /// Args:
    ///     folder: Directory to write the store into.
    ///     block_size: Uncompressed block size in bytes before flushing.
    ///     zstd_level: zstd compression level.
    ///     checkpoint_frequency: Controls checkpointing/recovery.
    ///         - ``0`` (default): disabled — output files are truncated on
    ///           open and any existing checkpoint is removed.
    ///         - ``N > 0``: recover from any existing checkpoint, then
    ///           automatically checkpoint every ``N`` added documents.
    ///         - ``None``: recover from any existing checkpoint, but never
    ///           auto-checkpoint — call ``checkpoint()`` manually.
    #[new]
    #[pyo3(signature = (folder, block_size=4096, zstd_level=3, checkpoint_frequency=Some(0)))]
    fn new(
        folder: &str,
        block_size: usize,
        zstd_level: i32,
        checkpoint_frequency: Option<u64>,
    ) -> PyResult<Self> {
        let opts = docstore::builder::BuilderOptions {
            block_size,
            zstd_level,
            checkpoint_frequency,
        };
        let builder =
            docstore::builder::DocumentStoreBuilder::new_with_options(Path::new(folder), &opts)
                .map_err(|e| {
                    pyo3::exceptions::PyIOError::new_err(format!(
                        "Failed to create DocumentStoreBuilder: {}",
                        e
                    ))
                })?;
        Ok(Self {
            builder: Some(builder),
        })
    }

    /// Add a document. Returns ``True`` if this call triggered an automatic
    /// checkpoint (only possible when ``checkpoint_frequency`` is a positive
    /// integer).
    fn add(&mut self, keys: HashMap<String, String>, content: &[u8]) -> PyResult<bool> {
        let doc = docstore::DocumentData {
            keys,
            content: content.to_vec(),
        };
        self.builder
            .as_mut()
            .ok_or_else(|| {
                pyo3::exceptions::PyRuntimeError::new_err("Builder already consumed by build()")
            })?
            .add(&doc)
            .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(format!("{}", e)))
    }

    /// Number of documents added so far (including any restored from a checkpoint).
    fn num_documents(&self) -> PyResult<u64> {
        Ok(self
            .builder
            .as_ref()
            .ok_or_else(|| {
                pyo3::exceptions::PyRuntimeError::new_err("Builder already consumed by build()")
            })?
            .num_documents())
    }

    /// Force a checkpoint now. Only useful when the builder was created with
    /// a non-zero ``checkpoint_frequency``.
    fn checkpoint(&mut self) -> PyResult<()> {
        self.builder
            .as_mut()
            .ok_or_else(|| {
                pyo3::exceptions::PyRuntimeError::new_err("Builder already consumed by build()")
            })?
            .checkpoint()
            .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(format!("{}", e)))
    }

    fn build(&mut self) -> PyResult<()> {
        let builder = self.builder.take().ok_or_else(|| {
            pyo3::exceptions::PyRuntimeError::new_err("Builder already consumed by build()")
        })?;
        builder
            .build()
            .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(format!("{}", e)))
    }
}

#[cfg_attr(feature = "stub-gen", gen_stub_pyclass)]
#[pyclass(name = "DocumentStore")]
pub struct PyDocumentStore {
    store: Arc<docstore::store::DocumentStore>,
}

// gen_stub_pymethods skipped: &[u8] in return types
// https://github.com/Jij-Inc/pyo3-stub-gen/issues/97
// gen_stub_pymethods skipped: (Self, Parent) return in #[new] unsupported
#[pymethods]
impl PyDocumentStore {
    #[staticmethod]
    #[pyo3(signature = (folder, content_access="memory"))]
    fn load(folder: &str, content_access: &str) -> PyResult<Self> {
        let access = match content_access {
            "memory" => docstore::store::ContentAccess::Memory,
            "mmap" => docstore::store::ContentAccess::Mmap,
            "disk" => docstore::store::ContentAccess::Disk,
            other => {
                return Err(pyo3::exceptions::PyValueError::new_err(format!(
                    "Unknown content_access '{}', expected 'memory', 'mmap', or 'disk'",
                    other
                )));
            }
        };
        let store = docstore::store::DocumentStore::load(Path::new(folder), access)
            .map_err(|e| pyo3::exceptions::PyIOError::new_err(format!("{}", e)))?;
        Ok(Self {
            store: Arc::new(store),
        })
    }

    fn num_documents(&self) -> u64 {
        self.store.num_documents()
    }

    fn key_names(&self) -> Vec<String> {
        self.store.key_names().to_vec()
    }

    fn get_by_number(&self, doc_numbers: Vec<u64>) -> PyResult<Vec<PyDocument>> {
        let docs = self
            .store
            .get_by_number(&doc_numbers)
            .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(format!("{}", e)))?;
        Ok(docs.into_iter().map(|d| PyDocument { inner: d }).collect())
    }

    fn get_by_key(
        &self,
        key_name: &str,
        key_values: Vec<String>,
    ) -> PyResult<Vec<Option<PyDocument>>> {
        let refs: Vec<&str> = key_values.iter().map(|s| s.as_str()).collect();
        let docs = self
            .store
            .get_by_key(key_name, &refs)
            .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(format!("{}", e)))?;
        Ok(docs
            .into_iter()
            .map(|opt| opt.map(|d| PyDocument { inner: d }))
            .collect())
    }

    fn aio_get_by_number<'a>(
        &self,
        py: Python<'a>,
        doc_numbers: Vec<u64>,
    ) -> PyResult<Bound<'a, PyAny>> {
        let store = self.store.clone();
        pyo3_async_runtimes::tokio::future_into_py(py, async move {
            let docs = task::spawn_blocking(move || {
                store.get_by_number(&doc_numbers).map_err(|e| e.to_string())
            })
            .await
            .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(format!("{}", e)))?
            .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e))?;
            let v: Vec<PyDocument> = docs.into_iter().map(|d| PyDocument { inner: d }).collect();
            Ok(v)
        })
    }

    fn aio_get_by_key<'a>(
        &self,
        py: Python<'a>,
        key_name: String,
        key_values: Vec<String>,
    ) -> PyResult<Bound<'a, PyAny>> {
        let store = self.store.clone();
        pyo3_async_runtimes::tokio::future_into_py(py, async move {
            let docs = task::spawn_blocking(move || {
                let refs: Vec<&str> = key_values.iter().map(|s| s.as_str()).collect();
                store
                    .get_by_key(&key_name, &refs)
                    .map_err(|e| e.to_string())
            })
            .await
            .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(format!("{}", e)))?
            .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e))?;
            let v: Vec<Option<PyDocument>> = docs
                .into_iter()
                .map(|opt| opt.map(|d| PyDocument { inner: d }))
                .collect();
            Ok(v)
        })
    }
}

// --- BM25 / Scoring Python bindings ---

/// Document metadata (document lengths) for use with scoring models.
#[cfg_attr(feature = "stub-gen", gen_stub_pyclass)]
#[pyclass(name = "DocMetadata")]
pub struct PyDocMetadata {
    inner: Arc<DocMetadata>,
}

#[cfg_attr(feature = "stub-gen", gen_stub_pymethods)]
#[pymethods]
impl PyDocMetadata {
    #[staticmethod]
    fn load(folder: &str) -> PyResult<Self> {
        let meta = DocMetadata::load(std::path::Path::new(folder)).map_err(|e| {
            pyo3::exceptions::PyIOError::new_err(format!("Failed to load DocMetadata: {}", e))
        })?;
        Ok(Self {
            inner: Arc::new(meta),
        })
    }

    fn num_docs(&self) -> u64 {
        self.inner.num_docs()
    }

    fn avg_dl(&self) -> f32 {
        self.inner.avg_dl()
    }

    fn min_dl(&self) -> u32 {
        self.inner.min_dl()
    }

    #[staticmethod]
    fn copy_files(src: &str, dst: &str) -> PyResult<()> {
        DocMetadata::copy_files(Path::new(src), Path::new(dst)).map_err(|e| {
            pyo3::exceptions::PyIOError::new_err(format!(
                "Failed to copy doc metadata files: {}",
                e
            ))
        })
    }
}

/// BM25 scoring model.
#[cfg_attr(feature = "stub-gen", gen_stub_pyclass)]
#[pyclass(name = "BM25Scoring")]
pub struct PyBM25Scoring {
    k1: f32,
    b: f32,
}

#[cfg_attr(feature = "stub-gen", gen_stub_pymethods)]
#[pymethods]
impl PyBM25Scoring {
    #[new]
    #[pyo3(signature = (k1=1.2, b=0.75))]
    fn new(k1: f32, b: f32) -> Self {
        Self { k1, b }
    }
}

/// A scored index that applies a scoring model to raw postings.
#[cfg_attr(feature = "stub-gen", gen_stub_pyclass)]
#[pyclass(name = "ScoredIndex", extends = PyIndexView)]
pub struct PyScoredIndex {
    index: Arc<Box<dyn SparseIndex>>,
}

impl PyScoredIndex {
    fn _search(
        &self,
        py: Python<'_>,
        py_query: &Bound<'_, PyDict>,
        top_k: usize,
        search_fn: SearchFn,
    ) -> PyResult<Py<PyAny>> {
        let query: HashMap<usize, ImpactValue> = py_query.extract()?;
        let results = search_fn(&**self.index, &query, top_k);
        let v: Vec<PyScoredDocument> = results
            .iter()
            .map(|r| PyScoredDocument {
                docid: r.docid,
                score: r.score,
            })
            .collect();
        Ok(pyo3::IntoPyObject::into_pyobject(v, py)?.into())
    }
}

#[cfg_attr(feature = "stub-gen", gen_stub_pymethods)]
#[pymethods]
impl PyScoredIndex {
    fn search_wand(
        &self,
        py: Python<'_>,
        py_query: &Bound<'_, PyDict>,
        top_k: usize,
    ) -> PyResult<Py<PyAny>> {
        self._search(py, py_query, top_k, search_wand)
    }

    fn search_maxscore(
        &self,
        py: Python<'_>,
        py_query: &Bound<'_, PyDict>,
        top_k: usize,
    ) -> PyResult<Py<PyAny>> {
        self._search(py, py_query, top_k, |index, query, top_k| {
            let options = MaxScoreOptions::default();
            search_maxscore(index, query, top_k, options)
        })
    }
}

/// Bag-of-words index builder for traditional IR (BM25, TF-IDF, etc.).
///
/// Example:
///
/// ```python,ignore
/// builder = impact_index.BOWIndexBuilder("/path/to/index", dtype="int32")
/// builder.add(0, terms, tf_values)
/// index, doc_meta = builder.build(in_memory=True)
/// scored = index.with_scoring(impact_index.BM25Scoring(), doc_meta)
/// results = scored.search_wand(query, top_k=10)
/// ```
#[cfg_attr(feature = "stub-gen", gen_stub_pyclass)]
#[pyclass(name = "BOWIndexBuilder")]
pub struct PyBOWIndexBuilder {
    inner: Arc<Mutex<Option<BOWBuilderEnum>>>,
}

enum BOWBuilderEnum {
    I32(BOWIndexBuilder<i32>),
    I64(BOWIndexBuilder<i64>),
    F32(BOWIndexBuilder<f32>),
}

// gen_stub_pymethods skipped: PyArray1<usize> not supported
// https://github.com/Jij-Inc/pyo3-stub-gen/issues/97
// gen_stub_pymethods skipped: (Self, Parent) return in #[new] unsupported
#[pymethods]
impl PyBOWIndexBuilder {
    /// Create a new BOWIndexBuilder.
    ///
    /// Args:
    ///     folder: Directory to store the index.
    ///     options: Builder options (block size, checkpoint frequency).
    ///     dtype: Data type for term frequencies ("int32", "int64", "float32").
    ///     stemmer: Stemmer to use ("snowball" or None).
    ///     language: Language for the stemmer (default: "english").
    ///     stop_words: Stop words to filter. Either a list of strings, or
    ///         ``True`` to use the default Lucene stop words for the language.
    #[new]
    #[pyo3(signature = (folder, options=None, dtype=None, stemmer=None, language=None, stop_words=None))]
    fn new(
        folder: &str,
        options: Option<&PyBuilderOptions>,
        dtype: Option<&str>,
        stemmer: Option<&str>,
        language: Option<&str>,
        stop_words: Option<&Bound<'_, PyAny>>,
    ) -> PyResult<Self> {
        let builder_options = match options {
            Some(o) => o.0.clone(),
            None => BuilderOptions::default(),
        };
        let path = Path::new(folder);
        let dtype_str = dtype.unwrap_or("int32");
        let lang = language.unwrap_or("english");

        // Resolve stop words: True = use language default, list = explicit, None = no stop words
        let resolved_stop_words: Vec<String> = match stop_words {
            Some(obj) => {
                if let Ok(true) = obj.extract::<bool>() {
                    // True: use default stop words for the language
                    crate::vocab::stopwords::get_stop_words(lang)
                        .ok_or_else(|| {
                            pyo3::exceptions::PyValueError::new_err(format!(
                                "No built-in stop words for language '{}'",
                                lang
                            ))
                        })?
                        .into_iter()
                        .map(|s| s.to_string())
                        .collect()
                } else if let Ok(list) = obj.extract::<Vec<String>>() {
                    list
                } else {
                    return Err(pyo3::exceptions::PyTypeError::new_err(
                        "stop_words must be True, a list of strings, or None",
                    ));
                }
            }
            None => Vec::new(),
        };

        let stop_word_refs: Vec<&str> = resolved_stop_words.iter().map(|s| s.as_str()).collect();

        let make_analyzer = |stemmer_name: &str,
                             stemmer_box: Box<dyn crate::vocab::stemmer::Stemmer>,
                             english: bool|
         -> TextAnalyzer {
            let mut a = if stop_word_refs.is_empty() {
                TextAnalyzer::new(stemmer_box)
            } else {
                TextAnalyzer::with_stop_words(stemmer_box, &stop_word_refs)
            };
            // Enable English possessive filter for English stemmers
            if english {
                a.set_english_possessive_filter(true);
            }
            // Store config for later retrieval
            a.set_config(crate::vocab::analyzer::AnalyzerConfig {
                stemmer: stemmer_name.to_string(),
                language: lang.to_string(),
                stop_words: !stop_word_refs.is_empty(),
                english_possessive_filter: english,
            });
            a
        };

        let is_english = lang == "english";

        let analyzer = match stemmer {
            Some("snowball") => {
                let s = SnowballStemmer::new(lang).map_err(|e| {
                    pyo3::exceptions::PyValueError::new_err(format!("Invalid stemmer: {}", e))
                })?;
                Some(make_analyzer("snowball", Box::new(s), is_english))
            }
            Some("porter") => Some(make_analyzer(
                "porter",
                Box::new(PorterStemmer::new()),
                true,
            )),
            Some("none") | None => None,
            Some(other) => {
                return Err(pyo3::exceptions::PyValueError::new_err(format!(
                    "Unknown stemmer '{}', expected 'snowball', 'porter', or None",
                    other
                )));
            }
        };

        let inner = match (dtype_str, analyzer) {
            ("int32" | "i32", None) => {
                BOWBuilderEnum::I32(BOWIndexBuilder::new(path, &builder_options))
            }
            ("int32" | "i32", Some(a)) => {
                BOWBuilderEnum::I32(BOWIndexBuilder::with_analyzer(path, &builder_options, a))
            }
            ("int64" | "i64", None) => {
                BOWBuilderEnum::I64(BOWIndexBuilder::new(path, &builder_options))
            }
            ("int64" | "i64", Some(a)) => {
                BOWBuilderEnum::I64(BOWIndexBuilder::with_analyzer(path, &builder_options, a))
            }
            ("float32" | "f32", None) => {
                BOWBuilderEnum::F32(BOWIndexBuilder::new(path, &builder_options))
            }
            ("float32" | "f32", Some(a)) => {
                BOWBuilderEnum::F32(BOWIndexBuilder::with_analyzer(path, &builder_options, a))
            }
            (other, _) => {
                return Err(pyo3::exceptions::PyValueError::new_err(format!(
                    "Unknown dtype '{}' for BOWIndexBuilder, expected int32, int64, or float32",
                    other
                )));
            }
        };

        Ok(Self {
            inner: Arc::new(Mutex::new(Some(inner))),
        })
    }

    fn add(
        &mut self,
        docid: DocId,
        terms: &Bound<'_, PyArray1<TermIndex>>,
        values: &Bound<'_, PyAny>,
    ) -> PyResult<()> {
        let py = values.py();
        let np = py.import("numpy")?;
        let values_f32: Bound<'_, PyArray1<f32>> = np
            .call_method1("asarray", (values,))?
            .call_method1("astype", ("float32",))?
            .extract()?;

        let terms_vec: Vec<TermIndex> = unsafe { terms.as_array() }.to_vec();
        let values_vec: Vec<f32> = unsafe { values_f32.as_array() }.to_vec();

        let mut inner = self.inner.blocking_lock();
        let builder = inner.as_mut().ok_or_else(|| {
            pyo3::exceptions::PyRuntimeError::new_err("Builder already consumed by build()")
        })?;

        match builder {
            BOWBuilderEnum::I32(b) => {
                let vals: Vec<i32> = values_vec.iter().map(|&v| v as i32).collect();
                b.add(docid, &terms_vec, &vals)
                    .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(format!("{}", e)))
            }
            BOWBuilderEnum::I64(b) => {
                let vals: Vec<i64> = values_vec.iter().map(|&v| v as i64).collect();
                b.add(docid, &terms_vec, &vals)
                    .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(format!("{}", e)))
            }
            BOWBuilderEnum::F32(b) => b
                .add(docid, &terms_vec, &values_vec)
                .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(format!("{}", e))),
        }
    }

    /// Add a batch of (docid, text) pairs with parallel text analysis.
    ///
    /// Tokenization and stemming are parallelized using rayon. Much faster
    /// than calling add_text() in a loop from Python.
    ///
    /// Documents must be sorted by ascending docid.
    fn add_texts(&mut self, documents: Vec<(DocId, String)>) -> PyResult<()> {
        let mut inner = self.inner.blocking_lock();
        let builder = inner.as_mut().ok_or_else(|| {
            pyo3::exceptions::PyRuntimeError::new_err("Builder already consumed by build()")
        })?;

        let doc_refs: Vec<(DocId, &str)> =
            documents.iter().map(|(d, t)| (*d, t.as_str())).collect();

        match builder {
            BOWBuilderEnum::I32(b) => b
                .add_texts_batch(&doc_refs)
                .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(format!("{}", e))),
            BOWBuilderEnum::I64(b) => b
                .add_texts_batch(&doc_refs)
                .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(format!("{}", e))),
            BOWBuilderEnum::F32(b) => b
                .add_texts_batch(&doc_refs)
                .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(format!("{}", e))),
        }
    }

    fn add_text(&mut self, docid: DocId, text: &str) -> PyResult<()> {
        let mut inner = self.inner.blocking_lock();
        let builder = inner.as_mut().ok_or_else(|| {
            pyo3::exceptions::PyRuntimeError::new_err("Builder already consumed by build()")
        })?;

        match builder {
            BOWBuilderEnum::I32(b) => b
                .add_text(docid, text)
                .map(|_| ())
                .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(format!("{}", e))),
            BOWBuilderEnum::I64(b) => b
                .add_text(docid, text)
                .map(|_| ())
                .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(format!("{}", e))),
            BOWBuilderEnum::F32(b) => b
                .add_text(docid, text)
                .map(|_| ())
                .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(format!("{}", e))),
        }
    }

    fn analyze_query(&self, text: &str) -> PyResult<HashMap<TermIndex, f32>> {
        let inner = self.inner.blocking_lock();
        let builder = inner.as_ref().ok_or_else(|| {
            pyo3::exceptions::PyRuntimeError::new_err("Builder already consumed by build()")
        })?;

        match builder {
            BOWBuilderEnum::I32(b) => Ok(b.analyze_query(text)),
            BOWBuilderEnum::I64(b) => Ok(b.analyze_query(text)),
            BOWBuilderEnum::F32(b) => Ok(b.analyze_query(text)),
        }
    }

    fn build(&mut self, py: Python<'_>, in_memory: bool) -> PyResult<Py<PyAny>> {
        let mut inner = self.inner.blocking_lock();
        let builder = inner.take().ok_or_else(|| {
            pyo3::exceptions::PyRuntimeError::new_err("Builder already consumed by build()")
        })?;

        macro_rules! build_bow {
            ($builder:expr) => {{
                let folder = $builder.folder().to_path_buf();
                let (index, _doc_meta) = $builder.build(in_memory).map_err(|e| {
                    pyo3::exceptions::PyRuntimeError::new_err(format!("Build failed: {}", e))
                })?;

                let base = PyClassInitializer::from(PyIndexView {});
                let sub = base.add_subclass(PySparseIndex {
                    index: Arc::new(Box::new(index)),
                });
                Ok(Py::new(py, sub)?.into_any())
            }};
        }

        match builder {
            BOWBuilderEnum::I32(b) => build_bow!(b),
            BOWBuilderEnum::I64(b) => build_bow!(b),
            BOWBuilderEnum::F32(b) => build_bow!(b),
        }
    }
}

#[cfg_attr(feature = "stub-gen", gen_stub_pyclass)]
#[pyclass(name = "TextAnalyzer")]
pub struct PyTextAnalyzer {
    inner: TextAnalyzer,
}

#[cfg_attr(feature = "stub-gen", gen_stub_pymethods)]
#[pymethods]
impl PyTextAnalyzer {
    /// Load a text analyzer from a built index directory.
    ///
    /// The index directory must contain a `vocab.cbor` file (written by
    /// BOWIndexBuilder.build()).
    ///
    /// Args:
    ///     folder: Path to the index directory
    ///     stemmer: Stemmer to use ("snowball", "porter", or None)
    ///     language: Language for snowball stemmer (default "english")
    ///     stop_words: Stop words list, or None
    #[staticmethod]
    #[pyo3(signature = (folder, stemmer=None, language=None, stop_words=None))]
    fn load(
        folder: &str,
        stemmer: Option<&str>,
        language: Option<&str>,
        stop_words: Option<Vec<String>>,
    ) -> PyResult<Self> {
        let path = Path::new(folder);

        let stemmer_box: Box<dyn crate::vocab::stemmer::Stemmer> = match stemmer {
            Some("snowball") => {
                let lang = language.unwrap_or("english");
                let s = SnowballStemmer::new(lang).map_err(|e| {
                    pyo3::exceptions::PyValueError::new_err(format!("Invalid stemmer: {}", e))
                })?;
                Box::new(s)
            }
            Some("porter") => Box::new(PorterStemmer::new()),
            Some("none") | None => Box::new(crate::vocab::stemmer::NoStemmer),
            Some(other) => {
                return Err(pyo3::exceptions::PyValueError::new_err(format!(
                    "Unknown stemmer '{}', expected 'snowball', 'porter', or None",
                    other
                )));
            }
        };

        let stop_word_refs: Vec<&str> = match &stop_words {
            Some(words) => words.iter().map(|s| s.as_str()).collect(),
            None => Vec::new(),
        };

        let analyzer = if stop_word_refs.is_empty() {
            TextAnalyzer::load(path, stemmer_box)
        } else {
            TextAnalyzer::load_with_stop_words(path, stemmer_box, &stop_word_refs)
        }
        .map_err(|e| {
            pyo3::exceptions::PyIOError::new_err(format!(
                "Failed to load text analyzer from '{}': {}",
                folder, e
            ))
        })?;

        Ok(Self { inner: analyzer })
    }

    /// Load a text analyzer from a saved config (auto-configures stemmer, stop words).
    ///
    /// The index directory must contain `vocab.cbor` and `analyzer.cbor` files.
    /// All settings (stemmer, stop words, possessive filter) are restored
    /// from the saved configuration.
    #[staticmethod]
    #[pyo3(name = "from_index")]
    fn from_index(folder: &str) -> PyResult<Self> {
        let path = Path::new(folder);
        let config = TextAnalyzer::load_config(path);

        // Recreate stemmer from config
        let stemmer_box: Box<dyn crate::vocab::stemmer::Stemmer> = match config.stemmer.as_str() {
            "snowball" => {
                let s = SnowballStemmer::new(&config.language).map_err(|e| {
                    pyo3::exceptions::PyValueError::new_err(format!("Invalid stemmer: {}", e))
                })?;
                Box::new(s)
            }
            "porter" => Box::new(PorterStemmer::new()),
            _ => Box::new(crate::vocab::stemmer::NoStemmer),
        };

        // Recreate stop words from config
        let stop_words = if config.stop_words {
            crate::vocab::stopwords::get_stop_words(&config.language)
                .unwrap_or_default()
                .iter()
                .map(|s| s.to_string())
                .collect::<Vec<_>>()
        } else {
            Vec::new()
        };

        let stop_word_refs: Vec<&str> = stop_words.iter().map(|s| s.as_str()).collect();

        let mut analyzer = if stop_word_refs.is_empty() {
            TextAnalyzer::load(path, stemmer_box)
        } else {
            TextAnalyzer::load_with_stop_words(path, stemmer_box, &stop_word_refs)
        }
        .map_err(|e| {
            pyo3::exceptions::PyIOError::new_err(format!(
                "Failed to load text analyzer from '{}': {}",
                folder, e
            ))
        })?;

        analyzer.set_english_possessive_filter(config.english_possessive_filter);
        analyzer.set_config(config);

        Ok(Self { inner: analyzer })
    }

    /// Analyze a query string into term IDs and frequencies.
    ///
    /// Unknown terms (not in the vocabulary) are skipped.
    fn analyze_query(&self, text: &str) -> HashMap<TermIndex, f32> {
        self.inner.analyze_query(text)
    }
}

/// Python module for sparse index construction, compression, and search.
#[pymodule]
fn impact_index(_py: Python, module: &Bound<'_, PyModule>) -> PyResult<()> {
    // Init logging
    pyo3_log::init();
    debug!("Loading xpmir-rust extension");

    module.add_class::<PyBuilderOptions>()?;
    module.add_class::<PyIndexBuilder>()?;
    module.add_class::<PySparseIndex>()?;
    module.add_class::<PySparseIndexIterator>()?;

    module.add_class::<PyEliasFanoCompressor>()?;
    module.add_class::<PyBitPackingCompressor>()?;
    module.add_class::<PyPForCompressor>()?;
    module.add_class::<PyImpactQuantizer>()?;
    module.add_class::<PyGlobalQuantizerFactory>()?;
    module.add_class::<PyBitPackedIntCompressor>()?;
    module.add_class::<PyQuantizedBitPackedCompressor>()?;
    module.add_class::<PyCompressionTransform>()?;
    module.add_class::<PySplitIndexTransform>()?;
    module.add_class::<PyReorderTransform>()?;
    module.add_class::<PyBmpSearcher>()?;

    module.add_class::<PyDocument>()?;
    module.add_class::<PyDocumentStoreBuilder>()?;
    module.add_class::<PyDocumentStore>()?;

    // BM25 / Scoring
    module.add_class::<PyDocMetadata>()?;
    module.add_class::<PyBM25Scoring>()?;
    module.add_class::<PyScoredIndex>()?;
    module.add_class::<PyBOWIndexBuilder>()?;
    module.add_class::<PyTextAnalyzer>()?;

    // Functions
    #[pyfn(module)]
    #[pyo3(name = "get_stop_words")]
    fn py_get_stop_words(language: &str) -> PyResult<Vec<String>> {
        crate::vocab::stopwords::get_stop_words(language)
            .map(|words| words.into_iter().map(|s| s.to_string()).collect())
            .ok_or_else(|| {
                pyo3::exceptions::PyValueError::new_err(format!(
                    "No stop words for language '{}'",
                    language
                ))
            })
    }

    Ok(())
}

#[cfg(feature = "stub-gen")]
pyo3_stub_gen::define_stub_info_gatherer!(stub_info);
