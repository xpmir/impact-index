//! WASM bindings for impact-index (`wasm-pack build --target web`).
//!
//! Bytes in, bytes out: the embedding application reads and writes the
//! index files, and hands them over as `Uint8Array`s.
//!
//! ```js
//! import init, { Builder, Index } from "impact_index_wasm";
//! await init();
//! const builder = new Builder({ range: { min: 0, max: 3 } });
//! builder.add(0, new Uint32Array([12, 40]), new Float32Array([0.8, 1.5]));
//! const files = builder.finish(); // Map: file name -> Uint8Array
//! const index = Index.load(files);
//! const { docids, scores } = index.search(
//!     new Uint32Array([12, 40]), new Float32Array([1.2, 0.4]), 10);
//! ```

use std::collections::HashMap;
use std::rc::Rc;

use impact_index::base::{load_index_from_files, ImpactValue, TermIndex};
use impact_index::builder::MemoryIndexer;
use impact_index::compress::docid::{BitPackingCompressor, EliasFanoCompressor, PForCompressor};
use impact_index::compress::impact::{
    Identity, QuantizationRange, QuantizerCodec, QuantizerParams, RangedQuantizerFactory,
};
use impact_index::compress::{
    CompressionTransform, DocIdCompressorFactory, ImpactCompressorFactory,
};
use impact_index::docstore::keys::KeyStoreBuilder as KeyStoreBuilderInner;
use impact_index::docstore::store::{ContentAccess, DocumentStore as StoreInner};
use impact_index::index::{SparseIndex, SparseIndexView};
use impact_index::search::filter::{DocBitSet, Exclude};
use impact_index::search::maxscore::{search_maxscore_filtered, MaxScoreOptions};
use impact_index::search::segments::{segment_size, Segment, SegmentedSearch};
use impact_index::storage::{MemoryFiles, MemoryOutput};
use js_sys::{Array, Map, Object, Reflect, Uint8Array};
use wasm_bindgen::prelude::*;
use wasm_bindgen::JsCast;

mod io;
use io::{JsOutput, JsSpillReader, JsSpillWriter};

#[cfg(target_family = "wasm")]
extern "C" {
    /// Static constructors, which register the (typetag) codecs
    fn __wasm_call_ctors();
}

#[wasm_bindgen(start)]
fn start() {
    // Run once, before any other export: otherwise the linker may call
    // the constructors at the start of every exported function
    #[cfg(target_family = "wasm")]
    unsafe {
        __wasm_call_ctors()
    };
    console_error_panic_hook::set_once();
}

/// Size of the WASM linear memory, in bytes (it only grows)
#[wasm_bindgen(js_name = memoryBytes)]
pub fn memory_bytes() -> f64 {
    #[cfg(target_arch = "wasm32")]
    return (core::arch::wasm32::memory_size(0) * 65536) as f64;
    #[cfg(not(target_arch = "wasm32"))]
    return 0.;
}

fn js_error(e: impl std::fmt::Display) -> JsError {
    JsError::new(&e.to_string())
}

/// `{name: bytes}` from a `Map` or a plain object
fn memory_files(files: &JsValue) -> Result<MemoryFiles, JsError> {
    let entries = js_entries(files)
        .ok_or_else(|| JsError::new("index files must be a Map or an object of Uint8Array"))?;
    let mut result = MemoryFiles::new();
    for (name, bytes) in entries {
        let name = name
            .as_string()
            .ok_or_else(|| JsError::new("file names must be strings"))?;
        let bytes = bytes
            .dyn_into::<Uint8Array>()
            .map_err(|_| JsError::new(&format!("{name}: expected a Uint8Array")))?;
        result.insert(name, bytes.to_vec());
    }
    Ok(result)
}

/// Entries of a `Map` or of a plain object
fn js_entries(value: &JsValue) -> Option<Vec<(JsValue, JsValue)>> {
    if let Some(map) = value.dyn_ref::<Map>() {
        let mut entries = Vec::new();
        map.for_each(&mut |value, key| entries.push((key, value)));
        Some(entries)
    } else if value.is_object() {
        Some(
            Object::entries(value.unchecked_ref())
                .iter()
                .map(|entry| {
                    let entry: Array = entry.unchecked_into();
                    (entry.get(0), entry.get(1))
                })
                .collect(),
        )
    } else {
        None
    }
}

/// Files as a `Map` from file name to `Uint8Array`
fn files_map(files: &MemoryFiles) -> Map {
    let map = Map::new();
    for name in files.names() {
        let bytes = files.get_bytes(name).expect("listed file");
        map.set(&JsValue::from_str(name), &Uint8Array::from(bytes).into());
    }
    map
}

/// Query as a term → weight map; repeated terms add up
fn query(terms: &[u32], weights: &[f32]) -> Result<HashMap<TermIndex, ImpactValue>, JsError> {
    if terms.len() != weights.len() {
        return Err(JsError::new("terms and weights differ in length"));
    }
    let mut query = HashMap::with_capacity(terms.len());
    for (&t, &w) in terms.iter().zip(weights) {
        *query.entry(t as TermIndex).or_insert(0.) += w;
    }
    Ok(query)
}

fn docid32(docid: u64) -> Result<u32, JsError> {
    u32::try_from(docid).map_err(|_| JsError::new(&format!("document id {docid} exceeds 2^32")))
}

/// A set of document ids (e.g. deleted documents of a segment)
#[wasm_bindgen]
#[derive(Default, Clone)]
pub struct DocSet {
    inner: DocBitSet,
}

#[wasm_bindgen]
impl DocSet {
    #[wasm_bindgen(constructor)]
    pub fn new() -> DocSet {
        DocSet::default()
    }

    /// The set of the given document ids
    #[wasm_bindgen(js_name = fromDocids)]
    pub fn from_docids(docids: &[u32]) -> DocSet {
        DocSet {
            inner: docids.iter().map(|&d| d as u64).collect(),
        }
    }

    /// Adds a document, returns `true` if it was not in the set
    pub fn add(&mut self, docid: u32) -> bool {
        self.inner.insert(docid as u64)
    }

    /// Removes a document, returns `true` if it was in the set
    pub fn remove(&mut self, docid: u32) -> bool {
        self.inner.remove(docid as u64)
    }

    pub fn has(&self, docid: u32) -> bool {
        self.inner.contains(docid as u64)
    }

    #[wasm_bindgen(getter)]
    pub fn size(&self) -> usize {
        self.inner.len()
    }

    /// The document ids, in increasing order
    pub fn docids(&self) -> Result<Vec<u32>, JsError> {
        self.inner.iter().map(docid32).collect()
    }
}

/// Quantisation parameters of a compressed index
#[wasm_bindgen]
pub struct Quantizer {
    params: QuantizerParams,
}

#[wasm_bindgen]
impl Quantizer {
    #[wasm_bindgen(getter)]
    pub fn nbits(&self) -> u32 {
        self.params.nbits
    }

    #[wasm_bindgen(getter)]
    pub fn min(&self) -> f32 {
        self.params.min
    }

    #[wasm_bindgen(getter)]
    pub fn max(&self) -> f32 {
        self.params.max
    }

    #[wasm_bindgen(getter)]
    pub fn step(&self) -> f32 {
        self.params.step
    }
}

/// Search results, by decreasing score
#[wasm_bindgen]
pub struct SearchResults {
    docids: Vec<u32>,
    scores: Vec<f32>,
    segments: Option<Vec<u32>>,
}

#[wasm_bindgen]
impl SearchResults {
    #[wasm_bindgen(getter)]
    pub fn docids(&self) -> Vec<u32> {
        self.docids.clone()
    }

    #[wasm_bindgen(getter)]
    pub fn scores(&self) -> Vec<f32> {
        self.scores.clone()
    }

    /// Segment of each document (segmented search only)
    #[wasm_bindgen(getter)]
    pub fn segments(&self) -> Option<Vec<u32>> {
        self.segments.clone()
    }

    #[wasm_bindgen(getter)]
    pub fn length(&self) -> usize {
        self.docids.len()
    }
}

/// A loaded index
#[wasm_bindgen]
pub struct Index {
    inner: Rc<dyn SparseIndex>,
}

#[wasm_bindgen]
impl Index {
    /// Loads an index from its files: a `Map` (or an object) from file
    /// names, relative to the index directory (e.g. `inner/docids.dat`
    /// for a split index), to their contents
    pub fn load(files: JsValue) -> Result<Index, JsError> {
        let files = memory_files(&files)?;
        let index = load_index_from_files(&files, true).map_err(js_error)?;
        Ok(Index {
            inner: Rc::from(index),
        })
    }

    /// Number of terms
    #[wasm_bindgen(getter, js_name = numTerms)]
    pub fn num_terms(&self) -> usize {
        self.inner.len()
    }

    /// Quantisation parameters (compressed indexes with quantised impacts)
    pub fn quantizer(&self) -> Option<Quantizer> {
        self.inner.quantizer().map(|params| Quantizer { params })
    }

    /// Top-`k` documents (MaxScore) for the query `terms` / `weights`,
    /// skipping the `deleted` documents
    pub fn search(
        &self,
        terms: &[u32],
        weights: &[f32],
        k: usize,
        deleted: Option<DocSet>,
    ) -> Result<SearchResults, JsError> {
        let query = query(terms, weights)?;
        let deleted = deleted.map(|d| d.inner);
        let results = search_maxscore_filtered(
            &*self.inner,
            &query,
            k,
            MaxScoreOptions::default(),
            deleted.as_ref().map(Exclude),
        );
        Ok(SearchResults {
            docids: results
                .iter()
                .map(|d| docid32(d.docid))
                .collect::<Result<_, _>>()?,
            scores: results.iter().map(|d| d.score).collect(),
            segments: None,
        })
    }
}

/// Several indexes (segments) searched as one, each with its own deleted
/// documents; scores must be comparable across segments (see the
/// quantisation range modes)
#[wasm_bindgen]
#[derive(Default)]
pub struct SegmentedIndex {
    indexes: Vec<Rc<dyn SparseIndex>>,
    deleted: Vec<Option<DocBitSet>>,
    sizes: Vec<usize>,
}

#[wasm_bindgen]
impl SegmentedIndex {
    #[wasm_bindgen(constructor)]
    pub fn new() -> SegmentedIndex {
        SegmentedIndex::default()
    }

    /// Adds a segment, returns its position
    pub fn push(&mut self, index: &Index, deleted: Option<DocSet>) -> usize {
        self.indexes.push(index.inner.clone());
        self.deleted.push(deleted.map(|d| d.inner));
        self.sizes.push(segment_size(&*index.inner));
        self.indexes.len() - 1
    }

    /// Replaces the deleted documents of a segment
    #[wasm_bindgen(js_name = setDeleted)]
    pub fn set_deleted(&mut self, segment: usize, deleted: Option<DocSet>) -> Result<(), JsError> {
        let slot = self
            .deleted
            .get_mut(segment)
            .ok_or_else(|| JsError::new(&format!("no segment {segment}")))?;
        *slot = deleted.map(|d| d.inner);
        Ok(())
    }

    #[wasm_bindgen(getter)]
    pub fn length(&self) -> usize {
        self.indexes.len()
    }

    /// Top-`k` documents over all segments
    pub fn search(
        &self,
        terms: &[u32],
        weights: &[f32],
        k: usize,
    ) -> Result<SearchResults, JsError> {
        let query = query(terms, weights)?;
        let segments = self
            .indexes
            .iter()
            .zip(&self.deleted)
            .map(|(index, deleted)| Segment {
                index: &**index,
                filter: deleted.as_ref().map(Exclude),
            })
            .collect();
        let results = SegmentedSearch::with_sizes(segments, &self.sizes).search(&query, k);
        Ok(SearchResults {
            docids: results
                .iter()
                .map(|d| docid32(d.docid))
                .collect::<Result<_, _>>()?,
            scores: results.iter().map(|d| d.score).collect(),
            segments: Some(results.iter().map(|d| d.segment as u32).collect()),
        })
    }
}

/// Field `name` of a JS options object, if set
fn option(options: &JsValue, name: &str) -> Result<Option<JsValue>, JsError> {
    if options.is_undefined() || options.is_null() {
        return Ok(None);
    }
    let value = Reflect::get(options, &JsValue::from_str(name))
        .map_err(|_| JsError::new("options must be an object"))?;
    Ok((!value.is_undefined() && !value.is_null()).then_some(value))
}

fn number_option(options: &JsValue, name: &str) -> Result<Option<f64>, JsError> {
    option(options, name)?
        .map(|v| {
            v.as_f64()
                .ok_or_else(|| JsError::new(&format!("option {name} must be a number")))
        })
        .transpose()
}

/// Builds a compressed index in memory
#[wasm_bindgen]
pub struct Builder {
    indexer: MemoryIndexer,
    block_size: usize,
    docids: String,
    /// Bits per quantised impact; `None` stores impacts as floats
    nbits: Option<u32>,
    range: QuantizationRange,
    clamped: Option<u64>,
    /// Spill memory budget (bytes) and scratch stream writer
    spill: Option<(usize, js_sys::Function)>,
    /// Set when spilling failed: the scratch stream is then unusable
    broken: Option<String>,
}

#[wasm_bindgen]
impl Builder {
    /// Options (all optional):
    /// - `blockSize` (128): postings per block
    /// - `docids` (`"pfor"`): document id codec, `"pfor"`, `"ef"`
    ///   (Elias-Fano) or `"bitpacking"`
    /// - `impacts` (`"quantized"`): `"quantized"` or `"float"`
    /// - `nbits` (8): bits per quantised impact
    /// - `range`: quantisation range, `{min, max}` (fixed: scores are then
    ///   exactly comparable across indexes built with the same range), or a
    ///   `Quantizer` of another index (to build segments comparable with
    ///   it); by default, the range of the indexed impacts
    /// - `spill`: `{budget, write(bytes)}`, to bound the memory used by
    ///   postings: when they exceed `budget` bytes (64 MiB), the largest
    ///   posting lists are moved to the end of a scratch stream through
    ///   `write` (synchronous, append-only, `bytes` a copy), down to half the
    ///   budget; `finish` then needs `readSpill` to read them back
    #[wasm_bindgen(constructor)]
    pub fn new(options: JsValue) -> Result<Builder, JsError> {
        let block_size = number_option(&options, "blockSize")?.unwrap_or(128.) as usize;
        if block_size == 0 {
            return Err(JsError::new("blockSize must be positive"));
        }
        let docids = match option(&options, "docids")? {
            None => "pfor".to_string(),
            Some(v) => v
                .as_string()
                .filter(|s| ["pfor", "ef", "bitpacking"].contains(&s.as_str()))
                .ok_or_else(|| JsError::new("docids must be \"pfor\", \"ef\" or \"bitpacking\""))?,
        };
        let nbits = number_option(&options, "nbits")?.unwrap_or(8.) as u32;
        if !(1..=32).contains(&nbits) {
            return Err(JsError::new("nbits must be between 1 and 32"));
        }
        let nbits = match option(&options, "impacts")?
            .and_then(|v| v.as_string())
            .as_deref()
        {
            None | Some("quantized") => Some(nbits),
            Some("float") => None,
            Some(other) => return Err(JsError::new(&format!("unknown impacts codec {other}"))),
        };
        let range = match option(&options, "range")? {
            None => QuantizationRange::PerIndex,
            Some(range) => {
                let min = number_option(&range, "min")?;
                let max = number_option(&range, "max")?;
                let step = number_option(&range, "step")?;
                let nbits = number_option(&range, "nbits")?;
                match (min, max, step, nbits) {
                    // A `Quantizer` (getters on its prototype)
                    (Some(min), Some(max), Some(step), Some(nbits)) => {
                        QuantizationRange::Reference(QuantizerParams {
                            nbits: nbits as u32,
                            min: min as f32,
                            max: max as f32,
                            step: step as f32,
                        })
                    }
                    (Some(min), Some(max), None, None) if min < max => QuantizationRange::Fixed {
                        min: min as f32,
                        max: max as f32,
                    },
                    _ => {
                        return Err(JsError::new(
                            "range must be {min, max} with min < max, or a Quantizer",
                        ))
                    }
                }
            }
        };
        let spill = match option(&options, "spill")? {
            None => None,
            Some(spill) => {
                let budget = number_option(&spill, "budget")?.unwrap_or((64 << 20) as f64);
                if budget.is_nan() || budget < 0. {
                    return Err(JsError::new("spill.budget must be a number of bytes"));
                }
                let write = io::method(&spill, "write")?
                    .ok_or_else(|| JsError::new("spill needs a write(bytes) method"))?;
                Some((budget as usize, write))
            }
        };
        Ok(Builder {
            indexer: MemoryIndexer::new(),
            block_size,
            docids,
            nbits,
            range,
            clamped: None,
            spill,
            broken: None,
        })
    }

    /// Adds a document; document ids must be increasing
    pub fn add(&mut self, docid: u32, terms: &[u32], weights: &[f32]) -> Result<(), JsError> {
        self.check_usable()?;
        let terms: Vec<TermIndex> = terms.iter().map(|&t| t as TermIndex).collect();
        self.indexer
            .add(docid as u64, &terms, weights)
            .map_err(js_error)?;
        if let Some((budget, write)) = &self.spill {
            if self.indexer.memory_bytes() > *budget {
                let mut out =
                    std::io::BufWriter::with_capacity(1 << 20, JsSpillWriter(write.clone()));
                let result = self
                    .indexer
                    .spill(budget / 2, &mut out)
                    .and_then(|_| std::io::Write::flush(&mut out));
                if let Err(e) = result {
                    let message = format!("spilling failed, the builder is unusable: {e}");
                    self.broken = Some(message.clone());
                    return Err(JsError::new(&message));
                }
            }
        }
        Ok(())
    }

    /// Bytes of postings held in the WASM memory (spilled ones excluded)
    #[wasm_bindgen(getter, js_name = memoryBytes)]
    pub fn memory_bytes(&self) -> f64 {
        self.indexer.memory_bytes() as f64
    }

    /// Bytes written to the scratch stream (see the `spill` option)
    #[wasm_bindgen(getter, js_name = spilledBytes)]
    pub fn spilled_bytes(&self) -> f64 {
        self.indexer.spilled_bytes() as f64
    }

    /// Number of postings added so far
    #[wasm_bindgen(getter, js_name = numPostings)]
    pub fn num_postings(&self) -> usize {
        self.indexer.num_postings()
    }

    /// Compresses the index. Without `output`, returns its files (a `Map`
    /// from file name to `Uint8Array`), to be stored by the caller and given
    /// to `Index.load`. With `output = {create(name), write(name, bytes),
    /// close(name), chunkSize}` (only `write` required), streams the files
    /// instead, by pieces of at most `chunkSize` bytes (1 MiB): they never
    /// sit whole in the WASM memory. The callbacks are synchronous and their
    /// return value ignored (queue asynchronous writes, and wait for the
    /// queue before using the files); `bytes` is a copy, safe to keep. An
    /// exception in a callback aborts `finish` with that error.
    ///
    /// If postings were spilled (`spill` option), `readSpill(offset,
    /// length)` must return those bytes of the scratch stream, synchronously
    /// (all spill writes must have completed).
    #[wasm_bindgen(unchecked_return_type = "Map<string, Uint8Array> | undefined")]
    pub fn finish(
        &mut self,
        #[wasm_bindgen(
            unchecked_param_type = "{create?(name: string): void, write(name: string, bytes: Uint8Array): void, close?(name: string): void, chunkSize?: number}"
        )]
        output: Option<JsValue>,
        #[wasm_bindgen(js_name = readSpill, unchecked_param_type = "(offset: number, length: number) => Uint8Array")]
        read_spill: Option<js_sys::Function>,
    ) -> Result<JsValue, JsError> {
        self.check_usable()?;
        if self.indexer.num_postings() == 0 {
            return Err(JsError::new("no postings to index"));
        }
        // No clock on wasm32-unknown-unknown: the manifest needs one
        #[cfg(target_family = "wasm")]
        impact_index::manifest::set_unix_time((js_sys::Date::now() / 1000.) as u64);

        let reader = match (self.indexer.is_spilled(), read_spill) {
            (false, _) => None,
            (true, Some(read)) => Some(JsSpillReader(read)),
            (true, None) => {
                return Err(JsError::new(
                    "postings were spilled: finish needs readSpill(offset, length)",
                ))
            }
        };
        // Read-ahead within the spill budget, the memory the caller allowed
        let read_ahead = self.spill.as_ref().map_or(0, |(budget, _)| *budget);
        let spilled = reader
            .as_ref()
            .map(|r| self.indexer.with_spill(r, read_ahead));
        let index: &dyn SparseIndexView = match &spilled {
            Some(spilled) => spilled,
            None => &self.indexer,
        };
        let transform = self.transform();
        let (files, report) = match output.filter(|o| !o.is_undefined() && !o.is_null()) {
            Some(output) => {
                let out = JsOutput::new(&output)?;
                (JsValue::UNDEFINED, transform.compress_to(&out, index))
            }
            None => {
                let out = MemoryOutput::new();
                let report = transform.compress_to(&out, index);
                (files_map(&out.into_files()).into(), report)
            }
        };
        // A read error first: it explains any other
        if let Some(spilled) = &spilled {
            spilled.check().map_err(js_error)?;
        }
        self.clamped = Some(report.map_err(js_error)?.clamped_impacts);
        Ok(files)
    }

    /// Number of impacts clamped to the quantisation range by `finish`
    /// (fixed or reference range only); if too many, start a new reference
    #[wasm_bindgen(getter, js_name = clampedImpacts)]
    pub fn clamped_impacts(&self) -> Option<f64> {
        self.clamped.map(|n| n as f64)
    }
}

impl Builder {
    fn check_usable(&self) -> Result<(), JsError> {
        match &self.broken {
            Some(message) => Err(JsError::new(message)),
            None => Ok(()),
        }
    }

    fn transform(&self) -> CompressionTransform {
        let doc_ids_compressor_factory: Box<dyn DocIdCompressorFactory> = match self.docids.as_str()
        {
            "ef" => Box::new(EliasFanoCompressor {}),
            "bitpacking" => Box::new(BitPackingCompressor {}),
            _ => Box::new(PForCompressor {}),
        };
        let impacts_compressor_factory: Box<dyn ImpactCompressorFactory> = match self.nbits {
            None => Box::new(Identity {}),
            Some(nbits) => Box::new(RangedQuantizerFactory {
                nbits,
                range: self.range,
                codec: QuantizerCodec::BitPacked,
            }),
        };
        CompressionTransform {
            max_block_size: self.block_size,
            doc_ids_compressor_factory,
            impacts_compressor_factory,
            positions_codec: None,
        }
    }
}

/// Key values of a document: `{name: value}` (or a `Map`) of strings
fn document_keys(keys: &JsValue) -> Result<HashMap<String, String>, JsError> {
    let entries =
        js_entries(keys).ok_or_else(|| JsError::new("keys must be an object of strings"))?;
    entries
        .into_iter()
        .map(
            |(name, value)| match (name.as_string(), value.as_string()) {
                (Some(name), Some(value)) => Ok((name, value)),
                _ => Err(JsError::new("key names and values must be strings")),
            },
        )
        .collect()
}

/// Builds a key-only document store: maps document keys (e.g. a `docno`)
/// to document ids (e.g. those given to `Builder.add`) and back
#[wasm_bindgen]
pub struct KeyStoreBuilder {
    inner: Option<KeyStoreBuilderInner>,
}

#[wasm_bindgen]
impl KeyStoreBuilder {
    #[wasm_bindgen(constructor)]
    pub fn new() -> KeyStoreBuilder {
        KeyStoreBuilder {
            inner: Some(KeyStoreBuilderInner::new()),
        }
    }

    /// Adds a document with its keys (e.g. `{docno: "D1"}`); with `docid`
    /// (larger than the previous ones), at that document id, otherwise at
    /// the next one. Returns the document id.
    pub fn add(&mut self, keys: JsValue, docid: Option<u32>) -> Result<u32, JsError> {
        let keys = document_keys(&keys)?;
        let builder = self
            .inner
            .as_mut()
            .ok_or_else(|| JsError::new("finish() was already called"))?;
        let docid = match docid {
            Some(docid) => builder
                .add_with_docid(docid as u64, &keys)
                .map(|_| docid as u64),
            None => builder.add(&keys),
        }
        .map_err(js_error)?;
        docid32(docid)
    }

    /// Number of document ids used so far (largest + 1)
    #[wasm_bindgen(getter, js_name = numDocuments)]
    pub fn num_documents(&self) -> f64 {
        self.inner.as_ref().map_or(0, |b| b.num_documents()) as f64
    }

    /// The store files (a `Map` from file name to `Uint8Array`), to be
    /// stored by the caller and given to `DocumentStore.load`
    pub fn finish(&mut self) -> Result<Map, JsError> {
        let builder = self
            .inner
            .take()
            .ok_or_else(|| JsError::new("finish() was already called"))?;
        let out = MemoryOutput::new();
        builder.build_to(&out).map_err(js_error)?;
        Ok(files_map(&out.into_files()))
    }
}

/// A document store: key lookups both ways. Stores built natively with
/// content also load, but their content cannot be read in WASM (no zstd).
#[wasm_bindgen]
pub struct DocumentStore {
    inner: StoreInner,
}

#[wasm_bindgen]
impl DocumentStore {
    /// Loads a store from its files (a `Map` or an object, from file name
    /// to `Uint8Array`)
    pub fn load(files: JsValue) -> Result<DocumentStore, JsError> {
        let files = memory_files(&files)?;
        let inner = StoreInner::load_from(&files, ContentAccess::Memory)
            .map_err(|e| JsError::new(&e.to_string()))?;
        Ok(DocumentStore { inner })
    }

    /// Number of document ids (largest + 1)
    #[wasm_bindgen(getter, js_name = numDocuments)]
    pub fn num_documents(&self) -> f64 {
        self.inner.num_documents() as f64
    }

    #[wasm_bindgen(getter, js_name = keyNames)]
    pub fn key_names(&self) -> Vec<String> {
        self.inner.key_names().to_vec()
    }

    #[wasm_bindgen(getter, js_name = hasContent)]
    pub fn has_content(&self) -> bool {
        self.inner.has_content()
    }

    /// Whether there is a document with this id
    pub fn contains(&self, docid: u32) -> bool {
        self.inner.contains(docid as u64)
    }

    /// Values of the key `keyName` for the given document ids (`undefined`
    /// if there is no such document, or it has no such key)
    #[wasm_bindgen(js_name = getKeys)]
    pub fn get_keys(&self, key_name: &str, docids: &[u32]) -> Result<Vec<JsValue>, JsError> {
        let docids: Vec<u64> = docids.iter().map(|&d| d as u64).collect();
        let keys = self
            .inner
            .get_keys(key_name, &docids)
            .map_err(|e| JsError::new(&e.to_string()))?;
        Ok(keys
            .into_iter()
            .map(|k| k.map_or(JsValue::UNDEFINED, |k| JsValue::from_str(&k)))
            .collect())
    }

    /// Document ids of the given values of the key `keyName` (`undefined`
    /// if not found)
    #[wasm_bindgen(js_name = getDocids)]
    pub fn get_docids(&self, key_name: &str, values: Vec<String>) -> Result<Vec<JsValue>, JsError> {
        let refs: Vec<&str> = values.iter().map(|s| s.as_str()).collect();
        let numbers = self
            .inner
            .get_numbers(key_name, &refs)
            .map_err(|e| JsError::new(&e.to_string()))?;
        Ok(numbers
            .into_iter()
            .map(|n| n.map_or(JsValue::UNDEFINED, |n| JsValue::from_f64(n as f64)))
            .collect())
    }
}
