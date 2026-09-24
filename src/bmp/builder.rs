//! Main streaming BMP conversion builder.
//!
//! Provides memory-efficient conversion from SparseIndex to BMP format
//! using streaming builders that don't store raw postings in memory.

use std::fs::File;
use std::io::{BufWriter, Result, Write};
use std::path::Path;

use indicatif::{ProgressBar, ProgressStyle};

use crate::base::ImpactValue;
use crate::index::SparseIndexView;

use super::index::Index;
use super::posting_list_builder::StreamingPostingListManager;

const DEFAULT_PROGRESS_TEMPLATE: &str =
    "{spinner:.green} [{elapsed_precise}] [{bar:40.cyan/blue}] {pos}/{len} ({eta})";

fn pb_style() -> ProgressStyle {
    ProgressStyle::default_bar()
        .template(DEFAULT_PROGRESS_TEMPLATE)
        .progress_chars("=> ")
}

/// Quantization levels for impact scores (8-bit quantization)
const LEVELS: i32 = 256;

/// Computes the global value range across all terms.
fn compute_value_range(index: &dyn SparseIndexView) -> (f32, f32) {
    let mut min_value = f32::INFINITY;
    let mut max_value = 0.0f32;

    for term_ix in 0..index.len() {
        let (_min, _max) = index.value_range(term_ix);
        max_value = max_value.max(_max);
        min_value = min_value.min(_min);
    }

    (min_value, max_value)
}

/// Creates a quantization function for impact scores.
fn make_quantizer(min_value: f32, max_value: f32) -> impl Fn(f32) -> u8 {
    let step = (LEVELS as f32) / (max_value - min_value);
    move |value: f32| (((value - min_value) * step) as i32).clamp(0, LEVELS - 1) as u8
}

/// Documents per pass-2 chunk: bounds the block forward index held in memory
/// (a few hundred MB for SPLADE on MS MARCO).
const DOCS_PER_CHUNK: usize = 1 << 18;

/// Converts a SparseIndex to BMP format using streaming (memory-efficient) builders.
///
/// This is a two-pass algorithm that avoids storing raw postings in memory:
/// - Pass 1: Build posting lists (block max scores, k-th percentiles), then
///   write the inverted index to the output file
/// - Pass 2: Build the block forward index chunk by chunk (a range of
///   blocks at a time) and append each chunk to the file as soon as it is
///   built
///
/// Memory usage is O(num_terms * num_blocks) for pass 1 and O(chunk
/// postings) for pass 2, instead of O(total_postings). The output is
/// byte-identical to [`crate::index::SparseIndex::convert_to_bmp`].
///
/// # Arguments
/// * `index` - The source sparse index
/// * `output` - Output path for the BMP index file
/// * `bsize` - Block size for BMP partitioning
/// * `compress_range` - Whether to compress block max scores
pub fn convert_to_bmp_streaming(
    index: &dyn SparseIndexView,
    output: &Path,
    bsize: usize,
    compress_range: bool,
) -> Result<()> {
    let blocks_per_chunk = (DOCS_PER_CHUNK / bsize).max(1);
    convert_to_bmp_streaming_chunked(index, output, bsize, compress_range, blocks_per_chunk)
}

/// [`convert_to_bmp_streaming`] with an explicit pass-2 chunk size (in
/// blocks); exposed for tests.
pub fn convert_to_bmp_streaming_chunked(
    index: &dyn SparseIndexView,
    output: &Path,
    bsize: usize,
    compress_range: bool,
    blocks_per_chunk: usize,
) -> Result<()> {
    let num_terms = index.len();
    let num_documents = (index.max_doc_id() + 1) as usize;
    let num_blocks = (num_documents + bsize - 1) / bsize;

    if num_terms > u16::MAX as usize + 1 {
        return Err(std::io::Error::new(
            std::io::ErrorKind::InvalidInput,
            format!("BMP supports at most 65536 terms, the index has {num_terms}"),
        ));
    }

    eprintln!(
        "Streaming BMP conversion: {} terms, {} documents, block size {}",
        num_terms, num_documents, bsize
    );

    // Compute quantization parameters
    let (min_value, max_value) = compute_value_range(index);
    let quantize = make_quantizer(min_value, max_value);

    // === Pass 1: Build posting lists ===
    eprintln!("Pass 1: Building posting lists (streaming)");
    let mut posting_manager = StreamingPostingListManager::new(num_terms, num_documents, bsize);

    let progress = ProgressBar::new(num_terms as u64);
    progress.set_style(pb_style());

    for term_ix in 0..num_terms {
        for posting in index.iterator(term_ix) {
            let quantized_score = quantize(posting.value);
            posting_manager.add_posting(term_ix, posting.docid as u32, quantized_score);
        }
        progress.inc(1);
    }
    progress.finish();

    let posting_lists = posting_manager.build(compress_range);
    eprintln!("  Built {} posting lists", posting_lists.len());

    // Generate term names (using numeric IDs as strings, matching legacy behavior)
    let term_names: Vec<String> = (0..num_terms).map(|i| i.to_string()).collect();

    // Generate document names
    let documents: Vec<String> = (0..num_documents).map(|i| i.to_string()).collect();

    let inverted_index = Index::new(num_documents, posting_lists, term_names, documents);

    // The file is bincode of `(Index, BlockForwardIndex)`. bincode lays a
    // tuple/struct out as its fields in order and a `Vec` as a u64 length
    // followed by its elements, so the forward index can be written piece
    // by piece: `data.len()`, each block, then `block_size`.
    eprintln!("Serializing to {}", output.display());
    let mut writer = BufWriter::new(File::create(output)?);
    bincode_write(&mut writer, &inverted_index)?;
    drop(inverted_index);
    bincode_write(&mut writer, &(num_blocks as u64))?;

    // === Pass 2: Build block forward index ===
    eprintln!("Pass 2: Building block forward index (streaming)");

    // One cursor per term, advanced chunk after chunk (postings are sorted
    // by docid)
    let mut cursors: Vec<_> = (0..num_terms)
        .map(|term_ix| index.iterator(term_ix).peekable())
        .collect();

    let progress = ProgressBar::new(num_blocks as u64);
    progress.set_style(pb_style());

    let mut stats = BlockStatistics::default();
    let mut first_block = 0;
    while first_block < num_blocks {
        let n_blocks = blocks_per_chunk.min(num_blocks - first_block);
        let first_doc = (first_block * bsize) as u64;
        let end_doc = ((first_block + n_blocks) * bsize) as u64;

        // Terms are visited in increasing order, so each block's term list
        // comes out sorted, and a term's postings in a block are contiguous
        let mut blocks: Vec<Vec<(u16, Vec<(u8, u8)>)>> = vec![Vec::new(); n_blocks];
        for (term_ix, cursor) in cursors.iter_mut().enumerate() {
            let term_id = term_ix as u16;
            while let Some(posting) = cursor.next_if(|p| p.docid < end_doc) {
                if posting.docid < first_doc {
                    return Err(std::io::Error::new(
                        std::io::ErrorKind::InvalidData,
                        format!("postings of term {term_ix} are not sorted by document id"),
                    ));
                }
                let doc = posting.docid as usize;
                let block = &mut blocks[doc / bsize - first_block];
                let entry = ((doc % bsize) as u8, quantize(posting.value));
                match block.last_mut() {
                    Some((t, impacts)) if *t == term_id => impacts.push(entry),
                    _ => block.push((term_id, vec![entry])),
                }
            }
        }

        for block in &blocks {
            stats.add(block);
            bincode_write(&mut writer, block)?;
        }
        progress.inc(n_blocks as u64);
        first_block += n_blocks;
    }
    progress.finish();

    bincode_write(&mut writer, &bsize)?;
    writer.flush()?;

    stats.print(num_blocks);

    Ok(())
}

fn bincode_write<W: Write, T: serde::Serialize + ?Sized>(writer: &mut W, value: &T) -> Result<()> {
    bincode::serialize_into(writer, value)
        .map_err(|e| std::io::Error::new(std::io::ErrorKind::Other, e))
}

/// Block statistics, accumulated while the blocks are written.
#[derive(Default)]
struct BlockStatistics {
    total_terms: usize,
    total_avg_docs: f32,
}

impl BlockStatistics {
    fn add(&mut self, block: &[(u16, Vec<(u8, u8)>)]) {
        self.total_terms += block.len();
        if !block.is_empty() {
            self.total_avg_docs +=
                block.iter().map(|(_, v)| v.len()).sum::<usize>() as f32 / block.len() as f32;
        }
    }

    fn print(&self, num_blocks: usize) {
        if num_blocks == 0 {
            return;
        }
        eprintln!("Block statistics:");
        eprintln!("  Number of blocks: {}", num_blocks);
        eprintln!("  Avg terms per block: {}", self.total_terms / num_blocks);
        eprintln!(
            "  Avg docs per term (per block): {:.2}",
            self.total_avg_docs / num_blocks as f32
        );
    }
}

/// Trait extension for SparseIndexView to provide value_range method.
pub trait SparseIndexViewExt {
    fn value_range(&self, term_ix: usize) -> (ImpactValue, ImpactValue);
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_quantizer() {
        let quantize = make_quantizer(0.0, 1.0);

        assert_eq!(quantize(0.0), 0);
        assert_eq!(quantize(1.0), 255);
        assert_eq!(quantize(0.5), 128);

        // Test clamping
        assert_eq!(quantize(-1.0), 0);
        assert_eq!(quantize(2.0), 255);
    }

    #[test]
    fn test_quantizer_with_offset() {
        let quantize = make_quantizer(10.0, 20.0);

        assert_eq!(quantize(10.0), 0);
        assert_eq!(quantize(20.0), 255);
        assert_eq!(quantize(15.0), 128);
    }
}
