//! Index construction from document impact vectors.
//!
//! Provides [`Indexer`] for building a forward index on disk, with optional
//! checkpointing for crash recovery during long indexing runs. Once built,
//! the index can be loaded as a [`SparseBuilderIndex`] for searching or
//! further transformation (compression, splitting).

use std::{
    cell::RefCell,
    fs::{self, File},
    io::{BufReader, BufWriter, Seek, Write},
    path::{Path, PathBuf},
};

use byteorder::{BigEndian, ReadBytesExt, WriteBytesExt};
use derivative::Derivative;
use log::{debug, info};
use ndarray::{ArrayBase, Data, Ix1};
use serde::{Deserialize, Serialize};

use super::{
    index::{BlockTermImpactIterator, SparseIndex},
    index::{IndexInformation, TermIndexPageInformation},
};
use crate::{
    base::{BoxResult, DocId, GenericTermImpact, ImpactValue, PostingValue, TermIndex, ValueType},
    index::{SparseIndexInformation, TermIndexInformation},
};
use crate::{
    base::{Len, TermImpact},
    utils::buffer::{Buffer, MemoryBuffer, MmapBuffer, Slice},
};

/*
* ---- First phase data structure
*
*/

/// Configuration options for the [`Indexer`].
#[derive(Derivative, Clone)]
#[derivative(Default)]
pub struct BuilderOptions {
    /// Build a checkpoint every N documents (0 disables checkpointing).
    ///
    /// Checkpoints allow resuming indexing after a crash by persisting
    /// the current state to disk periodically.
    #[derivative(Default(value = "0"))]
    pub checkpoint_frequency: DocId,

    /// Maximum number of in-memory postings per term before flushing to disk.
    ///
    /// Default is 128, matching the optimal block size for block-max pruning.
    /// With buffered I/O, small blocks have minimal overhead.
    #[derivative(Default(value = "128"))]
    pub in_memory_threshold: usize,

    /// Ratio of `in_memory_threshold` used as flush threshold during checkpoints.
    ///
    /// Posting lists with length >= `in_memory_threshold * checkpoint_flush_ratio`
    /// will be flushed to disk before checkpointing to reduce checkpoint size.
    #[derivative(Default(value = "0.5"))]
    pub checkpoint_flush_ratio: f64,
}

/// Holds a block of impacts together
/// with the information
#[derive(Serialize, Deserialize)]
#[serde(bound(deserialize = "V: PostingValue"))]
struct PostingsInformation<V: PostingValue> {
    postings: Vec<GenericTermImpact<V>>,
    information: TermIndexPageInformation,
}
impl<V: PostingValue> PostingsInformation<V> {
    fn new() -> Self {
        Self {
            postings: Vec::new(),
            information: TermIndexPageInformation::new(),
        }
    }

    #[inline]
    pub fn len(&self) -> usize {
        self.postings.len()
    }
}

struct TermsImpacts<V: PostingValue> {
    /// When using checkpointing, this is the last doc ID
    checkpoint_doc_id: Option<DocId>,
    options: BuilderOptions,
    folder: PathBuf,
    postings_file: std::io::BufWriter<std::fs::File>,
    /// Track file position manually to avoid flushing BufWriter on stream_position()
    postings_file_position: u64,
    information: IndexInformation,
    postings_information: Vec<PostingsInformation<V>>,
}

impl<V: PostingValue> TermsImpacts<V> {
    /**
     * Create a new term impacts in memory structure
     */
    fn new(folder: &Path, options: &BuilderOptions) -> TermsImpacts<V> {
        let path = folder.join(format!("postings.dat"));

        let mut file_options = File::options();
        file_options.read(true).write(true).create(true);

        if options.checkpoint_frequency == 0 {
            file_options.truncate(true);
        } else {
            file_options.truncate(false);
        }

        let mut _self = TermsImpacts {
            checkpoint_doc_id: None,
            options: options.clone(),
            folder: folder.to_path_buf(),
            postings_file: std::io::BufWriter::with_capacity(
                1 << 20, // 1 MB buffer
                file_options
                    .open(path)
                    .expect("Error while creating postings file."),
            ),
            postings_file_position: 0,
            postings_information: Vec::new(),
            information: IndexInformation::new(),
        };

        // Check if there is a checkpoint
        if options.checkpoint_frequency > 0 {
            let info_path = _self.folder.join(format!("checkpoint.cbor"));
            if fs::exists(&info_path).expect("error while checking if checkpoint exists") {
                let ckpt_file = File::options()
                    .read(true)
                    .open(info_path)
                    .expect("Error while opening checkpoint file");

                // Use zstd decompression
                let mut decoder = zstd::stream::Decoder::new(BufReader::new(ckpt_file))
                    .expect("Error creating zstd decoder");

                // Read binary IndexInformation first, then CBOR for the rest
                let pos: u64;
                let (_value_type, info) =
                    IndexInformation::read_binary(&mut decoder).unwrap_or_else(|_| {
                        // Fall back to old all-CBOR format
                        panic!("Cannot read checkpoint: incompatible format. Delete checkpoint.cbor and re-index.");
                    });
                _self.information = info;
                (_self.postings_information, pos, _self.checkpoint_doc_id) =
                    ciborium::de::from_reader(decoder)
                        .expect("error while reading checkpoint postings");

                // Note that we don't truncate the file since we will overwrite
                // everything from there
                _self
                    .postings_file
                    .seek(std::io::SeekFrom::Start(pos))
                    .expect("Error while moving in the posting file");
                _self.postings_file_position = pos;

                info!(
                    "Read checkpoint (current doc ID {})",
                    _self.checkpoint_doc_id.unwrap()
                );
            }
        }

        _self
    }

    /// Build a checkpoint
    fn checkpoint(&mut self, doc_id: DocId) {
        info!("Check pointing index ({})", doc_id);

        // Flush large posting lists to reduce checkpoint size
        let flush_threshold = (self.options.in_memory_threshold as f64
            * self.options.checkpoint_flush_ratio) as usize;
        for term_ix in 0..self.postings_information.len() {
            if self.postings_information[term_ix].len() >= flush_threshold {
                self.flush(term_ix)
                    .expect("error when flushing term during checkpoint");
            }
        }

        self.postings_file
            .flush()
            .expect("error when flushing the posting file");

        let info_path = self.folder.join("checkpoint.cbor");
        let tmp_info_path = self.folder.join("checkpoint.cbor.tmp");
        let info_file = File::options()
            .write(true)
            .create(true)
            .truncate(true)
            .open(&tmp_info_path)
            .expect("Error while creating checkpoint file");

        // Use zstd compression
        let encoder = zstd::stream::Encoder::new(BufWriter::new(info_file), 3)
            .expect("Error creating zstd encoder");
        let mut encoder = encoder.auto_finish();

        let pos = self.postings_file_position;

        // Write IndexInformation as compact binary (the bulk of the data)
        let value_type = crate::base::value_type_of::<V>();
        self.information
            .write_binary(value_type, &mut encoder)
            .expect("Error writing binary metadata in checkpoint");

        // Write remaining state as CBOR (in-memory postings + position + doc_id)
        ciborium::ser::into_writer(&(&self.postings_information, pos, doc_id), &mut encoder)
            .expect("Error while serializing checkpoint");

        // Ensure encoder is flushed before rename
        drop(encoder);

        // Atomic rename
        fs::rename(tmp_info_path, info_path).expect("Error when moving checkpoint.cbor in place");

        self.checkpoint_doc_id = Some(doc_id);
    }
    /// Adds a term for a given document
    fn add_impact(
        &mut self,
        term_ix: TermIndex,
        docid: DocId,
        value: V,
    ) -> Result<(), std::io::Error> {
        assert!(
            value.is_positive(),
            "Impact values should be greater than 0"
        );

        // Adds new vectors for missing words
        if term_ix >= self.postings_information.len() {
            let d = term_ix - self.postings_information.len() + 1;
            for _i in 1..=d {
                self.postings_information.push(PostingsInformation::new());
                self.information.terms.push(TermIndexInformation {
                    pages: Vec::new(),
                    length: 0,
                    max_value: ImpactValue::NEG_INFINITY,
                    max_doc_id: 0,
                });
            }
        }

        // Update the postings information
        let p_info = &mut self.postings_information[term_ix];

        p_info.postings.push(GenericTermImpact {
            docid: docid,
            value: value,
        });

        let value_f32 = value.to_f32();
        if p_info.information.max_value < value_f32 {
            p_info.information.max_value = value_f32;
        }

        // Update the term information
        let info = &mut self.information.terms[term_ix];
        info.length += 1;

        if info.max_value < value_f32 {
            info.max_value = value_f32
        }

        assert!(
            info.length == 1 || (info.max_doc_id < docid),
            "Doc ID should be increasing and this is not the case: {} vs {}",
            info.max_doc_id,
            docid
        );
        info.max_doc_id = docid;

        // Flush if needed
        if self.postings_information[term_ix].postings.len() > self.options.in_memory_threshold {
            self.flush(term_ix)?;
        }
        Ok(())
    }

    /// Flush a term into disk
    fn flush(&mut self, term_ix: TermIndex) -> Result<(), std::io::Error> {
        let len_postings = self.postings_information[term_ix].len();
        if len_postings == 0 {
            return Ok(());
        }

        // Use tracked position (avoids flushing BufWriter for stream_position)
        let position = self.postings_file_position;
        debug!(
            "Flush {} at {} (length {})",
            term_ix, position, len_postings
        );

        // Starts with a fresh PostingsInformation for this term
        let mut postings_info = std::mem::replace(
            &mut self.postings_information[term_ix],
            PostingsInformation::new(),
        );
        postings_info.information.docid_position = position;
        postings_info.information.max_doc_id = postings_info
            .postings
            .last()
            .expect("should not be empty")
            .docid;
        postings_info.information.value_position = position; // will be ignored anyways
        postings_info.information.length = len_postings;
        self.information.terms[term_ix]
            .pages
            .push(postings_info.information);

        // outputs the postings for this term
        let record_size = (std::mem::size_of::<DocId>() + V::BYTE_SIZE) as u64;
        for ti in postings_info.postings.iter() {
            self.postings_file.write_u64::<BigEndian>(ti.docid)?;
            ti.value.write_be(&mut self.postings_file)?;
        }
        self.postings_file_position += record_size * len_postings as u64;

        Ok(())
    }

    fn flush_all(&mut self) -> Result<(), std::io::Error> {
        for term_ix in 0..self.postings_information.len() {
            self.flush(term_ix)?;
        }
        self.postings_file.flush()?;
        Ok(())
    }
}

/// Builds a sparse index from document impact vectors.
///
/// Documents are added one at a time via [`add`](Indexer::add). Once all
/// documents have been added, call [`build`](Indexer::build) to finalize
/// the on-disk structure, then [`to_index`](Indexer::to_index) to obtain
/// a searchable index.
///
/// The type parameter `V` specifies the value type stored on disk.
/// Common choices are `f32`, `f16`, `bf16`, `f64`, `i32`, `i64`.
pub struct Indexer<V: PostingValue = f32> {
    impacts: TermsImpacts<V>,
    folder: PathBuf,
    built: bool,
}

impl<V: PostingValue> Indexer<V> {
    /// Creates a new indexer writing to the given directory.
    pub fn new(folder: &Path, options: &BuilderOptions) -> Indexer<V> {
        Indexer {
            impacts: TermsImpacts::new(folder, options),
            folder: folder.to_path_buf(),
            built: false,
        }
    }

    /// Get the index directory path.
    pub fn folder(&self) -> &Path {
        &self.folder
    }

    /// Returns the document ID from the last checkpoint, or `None` if
    /// no checkpoint exists. Useful to resume indexing after a crash.
    pub fn get_checkpoint_doc_id(&self) -> Option<DocId> {
        self.impacts.checkpoint_doc_id
    }

    /// Adds a document's sparse impact vector to the index.
    ///
    /// # Arguments
    ///
    /// * `docid` - Unique document identifier (must be strictly increasing)
    /// * `terms` - Array of term indices with non-zero impacts
    /// * `values` - Array of corresponding impact values (must be > 0)
    ///
    /// # Panics
    ///
    /// Panics if `terms` and `values` have different lengths, or if the index
    /// has already been built.
    pub fn add<S, T>(
        &mut self,
        docid: DocId,
        terms: &ArrayBase<S, Ix1>,
        values: &ArrayBase<T, Ix1>,
    ) -> Result<(), std::io::Error>
    where
        S: Data<Elem = TermIndex>,
        T: Data<Elem = V>,
    {
        assert!(
            !self.built,
            "Index cannot be changed since it has been built"
        );
        assert!(
            terms.len() == values.len(),
            "Value and term lists should have the same length"
        );
        for ix in 0..terms.len() {
            self.impacts.add_impact(terms[ix], docid, values[ix])?;
        }

        // Flush terms that have not been flushed for a long time (recovery)
        if (self.impacts.options.checkpoint_frequency > 0)
            && (docid
                >= self.impacts.options.checkpoint_frequency
                    + self.impacts.checkpoint_doc_id.unwrap_or(0))
        {
            // Perform a checkpoint
            self.impacts.checkpoint(docid);
        }

        Ok(())
    }

    /// Finalizes the index, flushing all remaining postings to disk.
    ///
    /// Must be called before [`to_index`](Indexer::to_index). Calling this
    /// multiple times is safe (subsequent calls are no-ops).
    pub fn build(&mut self) -> BoxResult<()> {
        if !self.built {
            // Flush the last impacts
            self.impacts.flush_all()?;
            self.built = true;

            // Write compact binary metadata
            let value_type = crate::base::value_type_of::<V>();
            let bin_path = self.folder.join("information.bin");
            let bin_file = File::options()
                .write(true)
                .create(true)
                .truncate(true)
                .open(&bin_path)
                .expect("Error creating information.bin");
            self.impacts
                .information
                .write_binary(value_type, &mut std::io::BufWriter::new(bin_file))
                .expect("Error writing binary metadata");

            // Write stub CBOR (just value type, empty terms) for backward compat.
            // Real data is in information.bin.
            let info_path = self.folder.join("information.cbor");
            let info_file = File::options()
                .write(true)
                .create(true)
                .truncate(true)
                .open(info_path)
                .expect("Error while creating file");
            let stub = IndexInformation::new(); // empty
            ciborium::ser::into_writer(&(value_type, &stub), info_file)
                .expect("Error while serializing stub");

            // Remove old checkpoint
            for s in ["checkpoint.cbor", "checkpoint.cbor.tmp"] {
                let cpkt_path = self.folder.join(s);
                if fs::exists(&cpkt_path).expect("error while checking if checkpoint exists") {
                    fs::remove_file(&cpkt_path).expect("error while removing checkpoint file");
                }
            }
        } else {
            println!("Already built")
        }

        Ok(())
    }

    /// Returns a sparse index (the index has to be built)
    pub fn to_index(&mut self, in_memory: bool) -> SparseBuilderIndex<V> {
        assert!(self.built, "Index is not built");
        assert!(
            self.impacts.information.terms.len() > 0,
            "Index has already been consumed into a forward index"
        );

        load_forward_index(self.impacts.folder.as_path(), in_memory)
    }
}

/// The raw forward index structure created during index construction.
///
/// Supports searching via block-based iteration. Typically used as an
/// intermediate step before applying compression or splitting transforms.
pub struct SparseBuilderIndex<V: PostingValue = f32> {
    /// Term information
    terms: Vec<TermIndexInformation>,

    /// View on the postings
    buffer: Box<dyn Buffer>,

    /// Phantom for the value type
    _phantom: std::marker::PhantomData<V>,

    /// Source directory path
    source_dir: Option<PathBuf>,

    /// Document metadata (if loaded from a BOW index)
    pub doc_meta: Option<crate::docmeta::DocMetadata>,

    /// Analyzer config (if loaded from a BOW index)
    pub analyzer_config: Option<crate::vocab::analyzer::AnalyzerConfig>,
}

impl<V: PostingValue> SparseBuilderIndex<V> {
    fn new(terms: Vec<TermIndexInformation>, path: &PathBuf, in_memory: bool) -> Self {
        // Auto-detect auxiliary files
        let dir = path.parent().unwrap_or(path);
        let doc_meta = crate::docmeta::DocMetadata::load(dir).ok();
        let analyzer_config = crate::vocab::analyzer::TextAnalyzer::load_config(dir);
        let analyzer_config = if analyzer_config.stemmer == "none" && !analyzer_config.stop_words {
            None // default config = no analyzer was configured
        } else {
            Some(analyzer_config)
        };

        Self {
            terms: terms,
            source_dir: Some(dir.to_path_buf()),
            doc_meta,
            analyzer_config,
            buffer: if in_memory {
                Box::new(MemoryBuffer::new(path))
            } else {
                Box::new(MmapBuffer::new(path))
            },
            _phantom: std::marker::PhantomData,
        }
    }
}

/// Loads a forward index from disk.
///
/// # Arguments
///
/// * `path` - Directory containing `information.cbor` and `postings.dat`
/// * `in_memory` - If `true`, loads postings into memory; otherwise uses mmap
pub fn load_forward_index<V: PostingValue>(path: &Path, in_memory: bool) -> SparseBuilderIndex<V> {
    // Prefer binary format (compact), fall back to CBOR
    let bin_path = path.join("information.bin");
    let ti: IndexInformation = if bin_path.exists() {
        let mut reader =
            std::io::BufReader::new(File::open(&bin_path).expect("Error opening information.bin"));
        let (_vtype, info) =
            IndexInformation::read_binary(&mut reader).expect("Error reading binary metadata");
        info
    } else {
        // Fall back to CBOR
        let mut buf = Vec::new();
        let mut file = File::options()
            .read(true)
            .open(path.join("information.cbor"))
            .expect("Error reading info");
        std::io::Read::read_to_end(&mut file, &mut buf).expect("Error reading info");

        if let Ok((_vtype, info)) =
            ciborium::de::from_reader::<(ValueType, IndexInformation), _>(&buf[..])
        {
            info
        } else {
            ciborium::de::from_reader(&buf[..]).expect("Error loading term index information")
        }
    };

    let postings_path = path.join("postings.dat");
    SparseBuilderIndex::new(ti.terms, &postings_path, in_memory)
}

/// Loads a forward index from disk with type-erased value type.
///
/// Reads the `ValueType` tag from metadata and dispatches to the correct
/// typed `SparseBuilderIndex<V>`, returning a `Box<dyn SparseIndex>`.
pub fn load_forward_index_dynamic(path: &Path, in_memory: bool) -> Box<dyn SparseIndex> {
    // Get value type: prefer binary, fall back to CBOR
    let bin_path = path.join("information.bin");
    let vtype = if bin_path.exists() {
        let mut reader =
            std::io::BufReader::new(File::open(&bin_path).expect("Error opening information.bin"));
        let (vtype, _info) =
            IndexInformation::read_binary(&mut reader).expect("Error reading binary metadata");
        vtype
    } else {
        let mut buf = Vec::new();
        let mut file = File::options()
            .read(true)
            .open(path.join("information.cbor"))
            .expect("Error reading info");
        std::io::Read::read_to_end(&mut file, &mut buf).expect("Error reading info");

        if let Ok((vtype, _info)) =
            ciborium::de::from_reader::<(ValueType, IndexInformation), _>(&buf[..])
        {
            vtype
        } else {
            ValueType::F32
        }
    };

    match vtype {
        ValueType::F32 => Box::new(load_forward_index::<f32>(path, in_memory)),
        ValueType::F64 => Box::new(load_forward_index::<f64>(path, in_memory)),
        ValueType::F16 => Box::new(load_forward_index::<half::f16>(path, in_memory)),
        ValueType::BF16 => Box::new(load_forward_index::<half::bf16>(path, in_memory)),
        ValueType::I32 => Box::new(load_forward_index::<i32>(path, in_memory)),
        ValueType::I64 => Box::new(load_forward_index::<i64>(path, in_memory)),
    }
}

pub struct SparseBuilderIndexIterator<'a, V: PostingValue> {
    /// Iterator over page information
    info_iter: Box<std::slice::Iter<'a, TermIndexPageInformation>>,

    /// Current info
    info: Option<&'a TermIndexPageInformation>,

    /// Current slice
    slice: Option<Box<dyn Slice + 'a>>,

    index: usize,
    /// Term index (for reference)
    term_ix: TermIndex,

    /// Our sparse index
    sparse_index: &'a SparseBuilderIndex<V>,
}

impl<'a, V: PostingValue> SparseBuilderIndexIterator<'a, V> {
    fn new<'c: 'a>(index: &'c SparseBuilderIndex<V>, term_ix: TermIndex) -> Self {
        let mut iter = if term_ix < index.terms.len() {
            Box::new(index.terms[term_ix].pages.iter())
        } else {
            Box::new([].iter())
        };

        let info = iter.next();

        Self {
            // The index
            sparse_index: &index,

            // Iterator over term index page information
            info_iter: iter,

            // Current term index page information
            info: info,

            // Current memory slice (None if not loaded)
            slice: None,

            // The current impact index
            index: 0,

            // Just for information purpose
            term_ix: term_ix,
        }
    }

    /// Move the iterator to the first block where a document of
    /// at least `min_doc_id` is present
    fn move_iterator(&mut self, min_doc_id: DocId) -> Option<DocId> {
        // Loop until the condition is met
        while let Some(info) = self.info {
            if info.max_doc_id >= min_doc_id {
                debug!(
                    "[{}] Moving iterator OK - max(doc_id) = {} >= {}",
                    self.term_ix, info.max_doc_id, min_doc_id
                );
                return Some(min_doc_id);
            }

            // Go to the next block
            self.next_block();

            if let Some(info) = self.info {
                debug!("[{}] Read the next block (move): {}", self.term_ix, info);
            } else {
                debug!("[{}] EOF for blocks (move)", self.term_ix);
            }
        }

        // No result
        None
    }

    const RECORD_SIZE: usize = std::mem::size_of::<DocId>() + V::BYTE_SIZE;

    fn read_block(&mut self, info: &TermIndexPageInformation) {
        let start = info.docid_position as usize;
        let end = info.docid_position as usize + info.length * Self::RECORD_SIZE;

        let slice = self.sparse_index.buffer.slice(start, end);
        self.slice = Some(slice);
    }

    fn next_block(&mut self) {
        self.info = self.info_iter.next();
        self.slice = None;
        self.index = 0;
    }
}

impl<'a, V: PostingValue> Iterator for SparseBuilderIndexIterator<'a, V> {
    type Item = TermImpact;

    /// Iterate to the next doc id
    fn next(&mut self) -> Option<Self::Item> {
        if let Some(info) = self.info {
            // We are over, load the next block
            if self.index >= info.length {
                self.next_block();
            }
            if self.info.is_none() {
                debug!("[{}] EOF for blocks", self.term_ix);
            }
        }

        if let Some(info) = self.info {
            if self.slice.is_none() {
                self.read_block(info);
                debug!("[{}] Loading block data: {}", self.term_ix, info);
            }

            let data = self.slice.as_ref().expect("").as_ref().data();
            let start = (self.index as usize) * Self::RECORD_SIZE;
            let end = start + Self::RECORD_SIZE;
            let mut slice_ptr = &data[start..end];

            let docid: DocId = slice_ptr
                .read_u64::<BigEndian>()
                .expect("Erreur de lecture");
            let value = V::read_be(&mut slice_ptr);
            self.index += 1;
            Some(TermImpact {
                docid: docid,
                value: value.to_f32(),
            })
        } else {
            None
        }
    }
}

// --- Block index

struct SparseBuilderBlockTermImpactIterator<'a, V: PostingValue> {
    iterator: RefCell<SparseBuilderIndexIterator<'a, V>>,
    current_min_docid: Option<DocId>,
    // We need a RefCell for method current()
    current_value: RefCell<Option<TermImpact>>,
    max_value: ImpactValue,
    max_doc_id: DocId,
    length: usize,
}

impl<'a, V: PostingValue> SparseBuilderBlockTermImpactIterator<'a, V> {
    fn new(index: &'a SparseBuilderIndex<V>, term_ix: TermIndex) -> Self {
        let info = &index.terms[term_ix];
        Self {
            iterator: RefCell::new(SparseBuilderIndexIterator::new(index, term_ix)),
            current_value: RefCell::new(None),
            max_value: info.max_value,
            max_doc_id: info.max_doc_id,
            length: info.length,
            current_min_docid: None,
        }
    }
}

impl<'a, V: PostingValue> BlockTermImpactIterator for SparseBuilderBlockTermImpactIterator<'a, V> {
    fn next_min_doc_id(&mut self, min_doc_id: DocId) -> Option<DocId> {
        // Sets the current minimum document ID
        self.current_min_docid = Some(min_doc_id.max(
            if let Some(impact) = self.current_value.get_mut() {
                impact.docid + 1
            } else {
                0
            },
        ));

        let min_doc_id = self.current_min_docid.expect("Should not be None");

        // Move to the block having at least one document greater that min_doc_id
        if let Some(min_doc_id) = self.iterator.get_mut().move_iterator(min_doc_id) {
            debug!(
                "[{}] We have a candidate for doc_id >= {}",
                self.iterator.get_mut().term_ix,
                min_doc_id
            );

            Some(min_doc_id)
        } else {
            debug!("[{}] End of iterator", self.iterator.get_mut().term_ix);
            None
        }
    }

    /// Returns the current document ID
    fn current(&self) -> TermImpact {
        let min_docid = self.current_min_docid.expect("Should not be null");
        {
            let iterator = self.iterator.borrow();
            debug!("[{}] Searching for next {}", iterator.term_ix, min_docid);
        }

        let mut current_value = self.current_value.borrow_mut();

        if current_value
            .and_then(|x| Some(x.docid < min_docid))
            .or(Some(true))
            .unwrap()
        {
            let mut iterator = self.iterator.borrow_mut();
            debug!(
                "[{}] Current DOC ID value is {}",
                iterator.term_ix,
                if let Some(cv) = current_value.as_ref() {
                    cv.docid as i64
                } else {
                    -1
                },
            );

            *current_value = None;
            while let Some(v) = iterator.next() {
                if v.docid >= min_docid {
                    debug!("[{}] Returning {} ({})", iterator.term_ix, v.docid, v.value);

                    *current_value = Some(v);
                    break;
                }
                debug!(
                    "[{}] Skipping {} ({}) / {}",
                    iterator.term_ix, v.docid, v.value, min_docid
                );
            }

            assert!(current_value.is_some(), "Did not find current impact");
        } else {
            let iterator = self.iterator.borrow();
            debug!(
                "[{}] Current value was good {} >= {}",
                iterator.term_ix,
                current_value.expect("").docid,
                min_docid
            );
        }

        return current_value.expect("No current value");
    }

    fn max_value(&self) -> ImpactValue {
        return self.max_value;
    }

    fn max_block_doc_id(&self) -> DocId {
        self.iterator
            .borrow()
            .info
            .expect("Iterator was over")
            .max_doc_id
    }

    fn max_block_value(&self) -> ImpactValue {
        self.iterator
            .borrow()
            .info
            .expect("Iterator was over")
            .max_value
    }

    fn max_doc_id(&self) -> DocId {
        return self.max_doc_id;
    }

    fn length(&self) -> usize {
        return self.length;
    }
}

impl<V: PostingValue> SparseIndex for SparseBuilderIndex<V> {
    fn block_iterator(&'_ self, term_ix: TermIndex) -> Box<dyn BlockTermImpactIterator + '_> {
        Box::new(SparseBuilderBlockTermImpactIterator::new(self, term_ix))
    }

    fn max_doc_id(&self) -> DocId {
        self.terms
            .iter()
            .map(|term| term.max_doc_id)
            .max()
            .unwrap_or(0)
    }

    fn doc_meta(&self) -> Option<&crate::docmeta::DocMetadata> {
        self.doc_meta.as_ref()
    }

    fn analyzer_config(&self) -> Option<&crate::vocab::analyzer::AnalyzerConfig> {
        self.analyzer_config.as_ref()
    }

    fn source_path(&self) -> Option<&std::path::Path> {
        self.source_dir.as_deref()
    }
}

impl<V: PostingValue> SparseIndexInformation for SparseBuilderIndex<V> {
    fn value_range(&self, term_ix: TermIndex) -> (ImpactValue, ImpactValue) {
        return (0., self.terms[term_ix].max_value);
    }
}

impl<V: PostingValue> Len for SparseBuilderIndex<V> {
    fn len(&self) -> usize {
        return self.terms.len();
    }
}
