//! Core index traits and data structures.
//!
//! Defines the main abstractions for sparse indices:
//! - [`SparseIndex`]: The primary trait for searchable indices with block-based iteration
//! - [`BlockTermImpactIterator`]: Block-level iterator over term postings
//! - [`SparseIndexView`]: Simplified iterator-based view for transforms
//! - [`SparseIndexInformation`]: Access to per-term statistics

use std::fmt;
use std::fs::File;
use std::io::{BufWriter, Result, Write};
use std::path::Path;

use crate::base::{DocId, ImpactValue, Len, TermIndex};
use bmp::index::forward_index::ForwardIndexBuilder;
use bmp::index::inverted_index::IndexBuilder;
use bmp::proto::{DocRecord, Header, Posting, PostingsList};
use indicatif::{ProgressBar, ProgressStyle};
use protobuf::CodedOutputStream;
use serde::{Deserialize, Serialize};

use crate::base::TermImpact;

/// Metadata for a single page (block) within a term's posting list.
#[derive(Serialize, Deserialize)]
pub struct TermIndexPageInformation {
    /// Position for the document ID stream
    pub docid_position: u64,

    /// Position for the impact value stream
    pub value_position: u64,

    /// Number of records
    pub length: usize,

    /// Maximum value for this page
    pub max_value: ImpactValue,

    /// Maximum document ID for this page
    pub max_doc_id: DocId,
    // /// Minimum document ID for this page
    // pub min_doc_id: DocId
}

impl TermIndexPageInformation {
    pub fn new() -> Self {
        Self {
            docid_position: 0,
            value_position: 0,
            length: 0,
            max_value: 0.,
            max_doc_id: 0,
            // min_doc_id: 0
        }
    }
}

impl std::fmt::Display for TermIndexPageInformation {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(
            f,
            "(pos: {}/{}, len: {}, max_v: {}, max_docid: {})",
            self.docid_position, self.value_position, self.length, self.max_value, self.max_doc_id
        )
    }
}

/// Aggregate metadata for a single term's posting list across all pages.
#[derive(Serialize, Deserialize)]
pub struct TermIndexInformation {
    /// Block-level metadata for each page in the posting list.
    pub pages: Vec<TermIndexPageInformation>,
    /// Maximum impact value across all postings for this term.
    pub max_value: ImpactValue,
    /// Maximum document ID in the posting list.
    pub max_doc_id: DocId,
    /// Total number of postings for this term.
    pub length: usize,
}

/// Global information on the index structure
#[derive(Serialize, Deserialize)]
pub struct IndexInformation {
    pub terms: Vec<TermIndexInformation>,
}

const FORWARD_INDEX_MAGIC: u32 = 0x46494458; // "FIDX"
const FORWARD_INDEX_VERSION: u32 = 1;

impl IndexInformation {
    /// Creates a new index information
    pub fn new() -> IndexInformation {
        IndexInformation { terms: Vec::new() }
    }

    /// Write in compact binary format.
    ///
    /// Per-term: 20 bytes (num_pages:u32, max_value:f32, max_doc_id:u64, length:u32)
    /// Per-page: 20 bytes (docid_position:u64, length:u32, max_value:f32, max_doc_id:u64)
    /// (value_position is same as docid_position for forward index, omitted)
    pub fn write_binary(
        &self,
        value_type: crate::base::ValueType,
        writer: &mut dyn std::io::Write,
    ) -> std::io::Result<()> {
        use byteorder::{LittleEndian, WriteBytesExt};

        writer.write_u32::<LittleEndian>(FORWARD_INDEX_MAGIC)?;
        writer.write_u32::<LittleEndian>(FORWARD_INDEX_VERSION)?;
        writer.write_u32::<LittleEndian>(self.terms.len() as u32)?;
        writer.write_u32::<LittleEndian>(value_type as u32)?;

        for term in &self.terms {
            writer.write_u32::<LittleEndian>(term.pages.len() as u32)?;
            writer.write_f32::<LittleEndian>(term.max_value)?;
            writer.write_u64::<LittleEndian>(term.max_doc_id)?;
            writer.write_u32::<LittleEndian>(term.length as u32)?;

            for page in &term.pages {
                writer.write_u64::<LittleEndian>(page.docid_position)?;
                writer.write_u32::<LittleEndian>(page.length as u32)?;
                writer.write_f32::<LittleEndian>(page.max_value)?;
                writer.write_u64::<LittleEndian>(page.max_doc_id)?;
            }
        }
        Ok(())
    }

    /// Read from compact binary format.
    pub fn read_binary(
        reader: &mut dyn std::io::Read,
    ) -> std::io::Result<(crate::base::ValueType, Self)> {
        use byteorder::{LittleEndian, ReadBytesExt};

        let magic = reader.read_u32::<LittleEndian>()?;
        if magic != FORWARD_INDEX_MAGIC {
            return Err(std::io::Error::new(
                std::io::ErrorKind::InvalidData,
                "Not a forward index binary file",
            ));
        }
        let version = reader.read_u32::<LittleEndian>()?;
        if version != FORWARD_INDEX_VERSION {
            return Err(std::io::Error::new(
                std::io::ErrorKind::InvalidData,
                format!("Unsupported forward index version: {}", version),
            ));
        }
        let num_terms = reader.read_u32::<LittleEndian>()? as usize;
        let value_type_u32 = reader.read_u32::<LittleEndian>()?;
        let value_type = match value_type_u32 {
            0 => crate::base::ValueType::F16,
            1 => crate::base::ValueType::BF16,
            2 => crate::base::ValueType::F32,
            3 => crate::base::ValueType::F64,
            4 => crate::base::ValueType::I32,
            5 => crate::base::ValueType::I64,
            _ => {
                return Err(std::io::Error::new(
                    std::io::ErrorKind::InvalidData,
                    format!("Unknown value type: {}", value_type_u32),
                ))
            }
        };

        let mut terms = Vec::with_capacity(num_terms);
        for _ in 0..num_terms {
            let num_pages = reader.read_u32::<LittleEndian>()? as usize;
            let max_value = reader.read_f32::<LittleEndian>()?;
            let max_doc_id = reader.read_u64::<LittleEndian>()?;
            let length = reader.read_u32::<LittleEndian>()? as usize;

            let mut pages = Vec::with_capacity(num_pages);
            for _ in 0..num_pages {
                let docid_position = reader.read_u64::<LittleEndian>()?;
                let page_length = reader.read_u32::<LittleEndian>()? as usize;
                let page_max_value = reader.read_f32::<LittleEndian>()?;
                let page_max_doc_id = reader.read_u64::<LittleEndian>()?;

                pages.push(TermIndexPageInformation {
                    docid_position,
                    value_position: docid_position, // same for forward index
                    length: page_length,
                    max_value: page_max_value,
                    max_doc_id: page_max_doc_id,
                });
            }

            terms.push(TermIndexInformation {
                pages,
                max_value,
                max_doc_id,
                length,
            });
        }

        Ok((value_type, IndexInformation { terms }))
    }
}

/// Provides per-term statistics (e.g., value range).
pub trait SparseIndexInformation: Len {
    /// Returns the (min, max) impact value range for the given term.
    fn value_range(&self, term_ix: TermIndex) -> (ImpactValue, ImpactValue);
}

/// A simple, sequential view over an index for use in transforms and exports.
pub trait SparseIndexView: Send + Sync + SparseIndexInformation {
    /// Basic iterator
    fn iterator<'a>(&'a self, term_ix: TermIndex) -> Box<dyn Iterator<Item = TermImpact> + 'a>;

    /// num_docs
    fn max_doc_id(&self) -> DocId;
}

/// Generic trait for block-based term impact iterators
pub trait BlockTermImpactIterator: Send {
    /// Moves to the next document whose id is greater or equal than doc_id.
    ///
    ///  The move can be "shallow", i.e. the index is read when any function
    /// that involves actual posting is invoked
    /// Returns the minimum that can reached (which is greater or equal than doc_id)
    fn next_min_doc_id(&mut self, doc_id: DocId) -> Option<DocId>;

    /// Returns the current term impact (can panic)
    fn current(&self) -> TermImpact;

    /// Returns the term maximum impact
    fn max_value(&self) -> ImpactValue;

    /// Returns the maximum document ID
    fn max_doc_id(&self) -> DocId;

    /// Max block impact value (by default, returns the maximum over all impacts)
    fn max_block_value(&self) -> ImpactValue {
        // If just one block...
        self.max_value()
    }

    /// Returns the minimum document ID for this block
    fn min_block_doc_id(&self) -> DocId {
        0
    }

    /// Max block document ID (by default, returns the maximum over all doc IDs)
    fn max_block_doc_id(&self) -> DocId {
        // If just one block...
        self.max_doc_id()
    }

    /// Returns the total number of records
    fn length(&self) -> usize;
}

impl<'a> Iterator for dyn BlockTermImpactIterator + 'a {
    type Item = TermImpact;
    fn next(&mut self) -> Option<TermImpact> {
        if self.next_min_doc_id(0).is_some() {
            Some(self.current())
        } else {
            None
        }
    }
}

pub trait AsSparseIndexView {
    fn as_view(&self) -> &dyn SparseIndexView;
}

fn pb_style() -> ProgressStyle {
    const DEFAULT_PROGRESS_TEMPLATE: &str =
        "{spinner:.green} [{elapsed_precise}] [{bar:40.cyan/blue}] {count}/{total} ({eta})";

    ProgressStyle::default_bar()
        .template(DEFAULT_PROGRESS_TEMPLATE)
        .progress_chars("=> ")
}

pub trait SparseIndex: Send + Sync + SparseIndexView + AsSparseIndexView {
    /// Returns an iterator for a given term
    ///
    /// ## Arguments
    ///
    /// * `term_ix` The index of the term
    fn block_iterator<'a>(&'a self, term_ix: TermIndex) -> Box<dyn BlockTermImpactIterator + 'a>;

    /// Returns the maximum document ID
    fn max_doc_id(&self) -> DocId;

    /// Returns all the iterators for a term (if split list)
    fn block_iterators(&self, term_ix: TermIndex) -> Vec<Box<dyn BlockTermImpactIterator + '_>> {
        let mut v = Vec::new();
        v.push(self.block_iterator(term_ix));
        v
    }

    /// Convert to BMP file format
    fn convert_to_bmp(&self, output: &Path, bsize: usize, compress_range: bool) -> Result<()> {
        let mut builder: IndexBuilder;
        let index = self.as_view();
        const LEVELS: i32 = 256;
        let num_documents = (index.max_doc_id() + 1) as usize;

        // Setup quantization

        let mut min_value = f32::INFINITY;
        let mut max_value = 0.;

        for term_ix in 0..index.len() {
            let (_min_value, _max_value) = index.value_range(term_ix);
            max_value = f32::max(max_value, _max_value);
            min_value = f32::min(min_value, _min_value);
        }
        let step = (LEVELS as f32) / (max_value - min_value);
        let quantize = move |value: f32| -> u32 {
            (((value - min_value) * step) as i32).clamp(0, LEVELS - 1) as u32
        };

        {
            builder = IndexBuilder::new(num_documents as usize, bsize);

            eprintln!("Processing postings");
            let progress = ProgressBar::new(u64::try_from(index.len()).expect("error"));
            progress.set_style(pb_style());
            progress.set_draw_delta(10);

            for term_ix in 0..index.len() {
                let postings: Vec<(u32, u32)> = index
                    .iterator(term_ix)
                    .map(|p| (p.docid as u32, quantize(p.value)))
                    .collect();
                builder.insert_term(&term_ix.to_string(), postings);
                progress.inc(1);
            }
            progress.finish();

            eprintln!("Processing document names");
            let progress = ProgressBar::new(u64::try_from(num_documents).expect(""));
            progress.set_style(pb_style());
            progress.set_draw_delta((num_documents / 100) as u64);

            for doc_ix in 0..num_documents {
                builder.insert_document(&doc_ix.to_string());
                progress.inc(1);
            }
            progress.finish();
        }
        let inverted_index = builder.build(compress_range);

        // --- Forward index

        eprintln!("Building forward index");
        let progress = ProgressBar::new(index.len() as u64);
        progress.set_style(pb_style());
        progress.set_draw_delta((index.len() / 100) as u64);

        let mut fwd_builder = ForwardIndexBuilder::new(num_documents as usize);

        for term_id in 0..index.len() {
            let posting_list: Vec<(u32, u32)> = index
                .iterator(term_id)
                .map(|p| (p.docid as u32, quantize(p.value)))
                .collect();

            fwd_builder.insert_posting_list(term_id as u32, &posting_list);
            progress.inc(1);
        }
        progress.finish();

        eprintln!("Converting to blocked forward index");
        let forward_index = fwd_builder.build();
        let b_forward_index = bmp::index::forward_index::fwd2bfwd(&forward_index, bsize);
        eprintln!("block numbers: {}", b_forward_index.data.len());
        let mut tot = 0;
        let mut tot_avg_docs = 0.0;
        for (_, block) in b_forward_index.data.iter().enumerate() {
            tot += block.len();
            tot_avg_docs +=
                block.iter().map(|(_, v)| v.len()).sum::<usize>() as f32 / block.len() as f32;
        }
        eprintln!("avg terms per block: {}", tot / b_forward_index.data.len());
        eprintln!(
            "avg docs per term: {}",
            tot_avg_docs / b_forward_index.data.len() as f32
        );
        let file = File::create(output).expect("Failed to create file");
        let writer = BufWriter::new(file);
        // Serialize the index directly into a file using bincode
        bincode::serialize_into(writer, &(&inverted_index, &b_forward_index))
            .expect("Failed to serialize");

        Ok(())
    }

    /// Convert to BMP file format using streaming (memory-efficient) builders.
    ///
    /// This is an alternative to `convert_to_bmp` that uses O(num_terms * num_blocks)
    /// memory instead of O(total_postings) memory. It performs two passes over the
    /// index but avoids storing raw postings in memory.
    ///
    /// # Arguments
    /// * `output` - Output path for the BMP index file
    /// * `bsize` - Block size for BMP partitioning
    /// * `compress_range` - Whether to compress block max scores
    fn convert_to_bmp_streaming(
        &self,
        output: &Path,
        bsize: usize,
        compress_range: bool,
    ) -> Result<()> {
        crate::bmp::convert_to_bmp_streaming(self.as_view(), output, bsize, compress_range)
    }

    fn to_ciff(&self, writer: &mut dyn Write, quantization: u128) {
        let mut output = CodedOutputStream::new(writer);
        let index_view = self.as_view();

        let mut header = Header::new();
        header.set_version(1);
        let num_documents = (index_view.max_doc_id() + 1) as i32;
        header.set_num_docs(num_documents);
        header.set_num_postings_lists(index_view.len() as i32);
        header.set_total_postings_lists(index_view.len() as i32);
        header.set_total_docs(num_documents);
        header.set_description("".to_string());

        header.set_total_terms_in_collection(0);
        header.set_average_doclength(0.);

        // Write header
        output.write_message_no_tag::<Header>(&header).ok();

        // Setup quantization

        let mut min_value = f32::INFINITY;
        let mut max_value = 0.;

        for term_ix in 0..index_view.len() {
            let (_min_value, _max_value) = index_view.value_range(term_ix);
            max_value = f32::max(max_value, _max_value);
            min_value = f32::min(min_value, _min_value);
        }
        let step = (max_value - min_value) * (quantization as f32);

        // Write posting lists
        for term_ix in 0..index_view.len() {
            let mut list = PostingsList::default();
            list.set_term(term_ix.to_string());
            list.set_df(0);
            list.set_cf(0);

            let iterator = index_view.iterator(term_ix);
            for impact in iterator {
                let mut posting = Posting::default();
                posting.set_docid(impact.docid as i32);
                let tf = ((impact.value - min_value) * step).clamp(0.0, (quantization - 1) as f32);
                posting.set_tf(tf as i32);
                list.postings.push(posting);
            }

            output.write_message_no_tag::<PostingsList>(&list).ok();
        }

        // Writer documents
        for doc_ix in 0..(1 + index_view.max_doc_id()) {
            let mut doc = DocRecord::default();
            doc.set_collection_docid(doc_ix.to_string());
            doc.set_docid(doc_ix as i32);
            output.write_message_no_tag::<DocRecord>(&doc).ok();
        }

        todo!("not working yet");
    }
}

struct SparseIndexViewIterator<'a>(Box<dyn BlockTermImpactIterator + 'a>);
impl<'a> Iterator for SparseIndexViewIterator<'a> {
    type Item = TermImpact;
    fn next(&mut self) -> Option<TermImpact> {
        self.0.next()
    }
}

impl<T: SparseIndex> AsSparseIndexView for T {
    fn as_view(&self) -> &dyn SparseIndexView {
        self
    }
}

impl<T> SparseIndexView for T
where
    T: SparseIndex,
{
    fn iterator<'a>(&'a self, term_ix: TermIndex) -> Box<dyn Iterator<Item = TermImpact> + 'a> {
        Box::new(SparseIndexViewIterator(self.block_iterator(term_ix)))
    }

    fn max_doc_id(&self) -> DocId {
        SparseIndex::max_doc_id(self)
    }
}

/// Iterator that yields only the impact values (discarding document IDs).
pub struct ValueIterator<'a> {
    iterator: Box<dyn BlockTermImpactIterator + 'a>,
}

impl<'a> Iterator for ValueIterator<'a> {
    type Item = ImpactValue;

    fn next(&mut self) -> Option<ImpactValue> {
        if let Some(ti) = self.iterator.next() {
            Some(ti.value)
        } else {
            None
        }
    }

    fn max(self) -> Option<Self::Item> {
        return Some(self.iterator.max_value());
    }
}
