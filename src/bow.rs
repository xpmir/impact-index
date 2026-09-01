//! Builder for bag-of-words IR indices (BM25, TF-IDF, etc.).
//!
//! [`BOWIndexBuilder`] wraps [`Indexer`] and automatically manages document
//! lengths. Optionally integrates a [`TextAnalyzer`] for direct text indexing.

use std::path::{Path, PathBuf};

use ndarray::Array1;
use rayon::prelude::*;

use crate::base::{BoxResult, DocId, PostingValue, TermIndex};
use crate::builder::{BuilderOptions, Indexer, SparseBuilderIndex};
use crate::docmeta::DocMetadata;
use crate::vocab::analyzer::TextAnalyzer;

/// Builder for bag-of-words IR indices (BM25, TF-IDF, etc.).
///
/// Wraps [`Indexer`] and automatically computes document lengths from TF values.
/// Optionally integrates a [`TextAnalyzer`] for direct text indexing.
pub struct BOWIndexBuilder<V: PostingValue> {
    indexer: Indexer<V>,
    doc_lengths: Vec<u32>,
    analyzer: Option<TextAnalyzer>,
    folder: PathBuf,
    /// Mirrors `options.positions`: routes `add_text`/`add_texts_batch`
    /// through the positional analyzer and gates `add`/`add_with_positions`.
    positions: bool,
}

impl<V: PostingValue> BOWIndexBuilder<V> {
    /// Create without text analysis (user provides TermIndex + values directly).
    pub fn new(folder: &Path, options: &BuilderOptions) -> Self {
        Self {
            indexer: Indexer::new(folder, options),
            doc_lengths: Vec::new(),
            analyzer: None,
            folder: folder.to_path_buf(),
            positions: options.positions,
        }
    }

    /// Create with a text analyzer (stemmer + vocabulary).
    pub fn with_analyzer(folder: &Path, options: &BuilderOptions, analyzer: TextAnalyzer) -> Self {
        Self {
            indexer: Indexer::new(folder, options),
            doc_lengths: Vec::new(),
            analyzer: Some(analyzer),
            folder: folder.to_path_buf(),
            positions: options.positions,
        }
    }

    /// Get the index directory path.
    pub fn folder(&self) -> &Path {
        &self.folder
    }

    /// Get a mutable reference to the analyzer (if present).
    pub fn analyzer_mut(&mut self) -> Option<&mut TextAnalyzer> {
        self.analyzer.as_mut()
    }

    /// Returns the document ID from the last checkpoint, or `None`.
    pub fn get_checkpoint_doc_id(&self) -> Option<DocId> {
        self.indexer.get_checkpoint_doc_id()
    }

    /// Add pre-tokenized postings. Document length = sum of values (cast to u32).
    pub fn add(
        &mut self,
        docid: DocId,
        terms: &[TermIndex],
        values: &[V],
    ) -> Result<(), std::io::Error> {
        assert!(
            !self.positions,
            "index built with positions=true: use add_with_positions"
        );
        assert_eq!(terms.len(), values.len());

        // Compute doc length as sum of values
        let doc_length: u32 = values.iter().map(|v| v.to_f32() as u32).sum();

        // Ensure doc_lengths vector is large enough
        let idx = docid as usize;
        if idx >= self.doc_lengths.len() {
            self.doc_lengths.resize(idx + 1, 0);
        }
        self.doc_lengths[idx] = doc_length;

        // Add to underlying indexer
        let terms_array = Array1::from_vec(terms.to_vec());
        let values_array = Array1::from_vec(values.to_vec());
        self.indexer.add(docid, &terms_array, &values_array)
    }

    /// Add pre-tokenized terms together with each term's token positions.
    /// Values (tf) are derived as `positions[i].len()`. Requires the index
    /// to have been built with `BuilderOptions { positions: true, .. }`.
    pub fn add_with_positions(
        &mut self,
        docid: DocId,
        terms: &[TermIndex],
        positions: &[Vec<u32>],
    ) -> Result<(), std::io::Error> {
        assert_eq!(terms.len(), positions.len());

        let values: Vec<V> = positions
            .iter()
            .map(|p| convert_f32_to_v::<V>(p.len() as f32))
            .collect();

        // Compute doc length as sum of tfs (same definition as `add`)
        let doc_length: u32 = positions.iter().map(|p| p.len() as u32).sum();

        let idx = docid as usize;
        if idx >= self.doc_lengths.len() {
            self.doc_lengths.resize(idx + 1, 0);
        }
        self.doc_lengths[idx] = doc_length;

        self.indexer
            .add_with_positions(docid, terms, &values, positions)
    }

    /// Add raw text (requires analyzer). Tokenizes, stems, computes TF,
    /// grows vocabulary, and records doc length automatically.
    ///
    /// Returns the document length.
    pub fn add_text(&mut self, docid: DocId, text: &str) -> Result<u32, std::io::Error> {
        let analyzer = self
            .analyzer
            .as_mut()
            .expect("add_text requires a TextAnalyzer");

        if self.positions {
            let terms_positions = analyzer.analyze_doc_positional(text);

            // Doc length is the total number of tokens (sum of tfs), same
            // definition as the non-positional path below.
            let doc_length: u32 = terms_positions.iter().map(|(_, p)| p.len() as u32).sum();

            let idx = docid as usize;
            if idx >= self.doc_lengths.len() {
                self.doc_lengths.resize(idx + 1, 0);
            }
            self.doc_lengths[idx] = doc_length;

            let terms: Vec<TermIndex> = terms_positions.iter().map(|(t, _)| *t).collect();
            let positions: Vec<Vec<u32>> = terms_positions.into_iter().map(|(_, p)| p).collect();
            let values: Vec<V> = positions
                .iter()
                .map(|p| convert_f32_to_v::<V>(p.len() as f32))
                .collect();

            self.indexer
                .add_with_positions(docid, &terms, &values, &positions)?;
            return Ok(doc_length);
        }

        let (term_indices, tf_values) = analyzer.analyze_doc(text);

        // Doc length is the total number of tokens (sum of TF values)
        let doc_length: u32 = tf_values.iter().map(|&v| v as u32).sum();

        // Ensure doc_lengths vector is large enough
        let idx = docid as usize;
        if idx >= self.doc_lengths.len() {
            self.doc_lengths.resize(idx + 1, 0);
        }
        self.doc_lengths[idx] = doc_length;

        // Convert to arrays and add - need V conversion from f32
        let terms_array = Array1::from_vec(term_indices);

        // Convert f32 TF values to V
        let values_v: Vec<V> = tf_values
            .iter()
            .map(|&v| convert_f32_to_v::<V>(v))
            .collect();
        let values_array = Array1::from_vec(values_v);

        self.indexer.add(docid, &terms_array, &values_array)?;
        Ok(doc_length)
    }

    /// Add a batch of (docid, text) pairs with parallel text analysis.
    ///
    /// Tokenization and stemming are done in parallel using rayon. Vocabulary
    /// growth and index insertion remain sequential (they require mutable state).
    /// This is useful when called from Python where individual `add_text` calls
    /// are GIL-bound.
    ///
    /// Documents must be sorted by ascending docid within the batch.
    pub fn add_texts_batch(&mut self, documents: &[(DocId, &str)]) -> Result<(), std::io::Error> {
        let analyzer = self
            .analyzer
            .as_ref()
            .expect("add_texts_batch requires a TextAnalyzer");

        if self.positions {
            // Phase 1: parallel tokenization + stemming, keeping positions
            // (read-only on analyzer).
            let analyzed: Vec<(DocId, Vec<(String, Vec<u32>)>)> = documents
                .par_iter()
                .map(|&(docid, text)| (docid, analyzer.tokenize_and_stem_positional(text)))
                .collect();

            // Phase 2: sequential vocabulary insertion + index building
            let analyzer = self.analyzer.as_mut().unwrap();
            for (docid, tokens) in analyzed {
                let mut term_indices = Vec::with_capacity(tokens.len());
                let mut positions: Vec<Vec<u32>> = Vec::with_capacity(tokens.len());
                let mut doc_length: u32 = 0;

                for (stemmed, pos) in tokens {
                    let idx = analyzer.vocab_mut().get_or_insert(&stemmed);
                    doc_length += pos.len() as u32;
                    term_indices.push(idx);
                    positions.push(pos);
                }

                // Record doc length
                let didx = docid as usize;
                if didx >= self.doc_lengths.len() {
                    self.doc_lengths.resize(didx + 1, 0);
                }
                self.doc_lengths[didx] = doc_length;

                // Add to indexer
                let values: Vec<V> = positions
                    .iter()
                    .map(|p| convert_f32_to_v::<V>(p.len() as f32))
                    .collect();
                self.indexer
                    .add_with_positions(docid, &term_indices, &values, &positions)?;
            }

            return Ok(());
        }

        // Phase 1: Parallel tokenization + stemming (read-only on analyzer)
        // We tokenize and stem but don't insert into vocabulary yet.
        // Each result is (docid, Vec<(stemmed_token, tf)>)
        let analyzed: Vec<(DocId, Vec<(String, f32)>)> = documents
            .par_iter()
            .map(|&(docid, text)| {
                let tokens = analyzer.tokenize_and_stem(text);
                (docid, tokens)
            })
            .collect();

        // Phase 2: Sequential vocabulary insertion + index building
        let analyzer = self.analyzer.as_mut().unwrap();
        for (docid, tokens) in analyzed {
            let mut term_indices = Vec::with_capacity(tokens.len());
            let mut tf_values = Vec::with_capacity(tokens.len());
            let mut doc_length: u32 = 0;

            for (stemmed, tf) in &tokens {
                let idx = analyzer.vocab_mut().get_or_insert(stemmed);
                term_indices.push(idx);
                tf_values.push(*tf);
                doc_length += *tf as u32;
            }

            // Record doc length
            let didx = docid as usize;
            if didx >= self.doc_lengths.len() {
                self.doc_lengths.resize(didx + 1, 0);
            }
            self.doc_lengths[didx] = doc_length;

            // Add to indexer
            let terms_array = Array1::from_vec(term_indices);
            let values_v: Vec<V> = tf_values
                .iter()
                .map(|&v| convert_f32_to_v::<V>(v))
                .collect();
            let values_array = Array1::from_vec(values_v);
            self.indexer.add(docid, &terms_array, &values_array)?;
        }

        Ok(())
    }

    /// Analyze a query text using the builder's analyzer.
    ///
    /// Returns term index -> TF mapping. Does NOT grow vocabulary.
    pub fn analyze_query(&self, text: &str) -> std::collections::HashMap<TermIndex, f32> {
        self.analyzer
            .as_ref()
            .expect("analyze_query requires a TextAnalyzer")
            .analyze_query(text)
    }

    /// Build the index and write document metadata.
    ///
    /// Returns the sparse index and document metadata.
    pub fn build(mut self, in_memory: bool) -> BoxResult<(SparseBuilderIndex<V>, DocMetadata)> {
        // Build the underlying index
        self.indexer.build()?;

        // Write document metadata and vocabulary BEFORE creating the index,
        // so that SparseBuilderIndex::new can auto-detect them from disk
        let doc_meta = DocMetadata::from_lengths(self.doc_lengths);
        doc_meta.save(&self.folder)?;

        if let Some(analyzer) = &self.analyzer {
            analyzer.save_vocab(&self.folder)?;
        }

        let index = self.indexer.to_index(in_memory);
        Ok((index, doc_meta))
    }
}

/// Convert f32 to PostingValue V.
fn convert_f32_to_v<V: PostingValue>(v: f32) -> V {
    use std::any::TypeId;
    let id = TypeId::of::<V>();
    unsafe {
        if id == TypeId::of::<f32>() {
            *(&v as *const f32 as *const V)
        } else if id == TypeId::of::<f64>() {
            let val = v as f64;
            *(&val as *const f64 as *const V)
        } else if id == TypeId::of::<half::f16>() {
            let val = half::f16::from_f32(v);
            *(&val as *const half::f16 as *const V)
        } else if id == TypeId::of::<half::bf16>() {
            let val = half::bf16::from_f32(v);
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
