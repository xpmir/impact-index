//! Vocabulary management for mapping between string terms and term indices.
//!
//! Provides [`Vocabulary`] for bidirectional term <-> [`TermIndex`] mapping.
//! Uses FST for compact on-disk storage (~26 MB for 2.6M terms).

//!
//! The text analyzers (feature `analysis`) live in the submodules;
//! [`analyzer::AnalyzerConfig`], stored with indexes, is always available.

pub mod analyzer;
#[cfg(feature = "analysis")]
pub mod icu;
#[cfg(feature = "analysis")]
pub mod language;
#[cfg(feature = "analysis")]
pub mod lucene;
#[cfg(feature = "analysis")]
pub mod lucene_porter;
#[cfg(feature = "analysis")]
pub mod stemmer;
#[cfg(feature = "analysis")]
pub mod stopwords;

use std::collections::HashMap;
use std::fs::File;
use std::io::BufWriter;
use std::path::Path;

use crate::base::TermIndex;

/// Cache directory of downloaded analysis data (ICU4X segmentation data,
/// morphological dictionaries): `$IMPACT_INDEX_CACHE`, or `impact-index` in
/// the OS cache directory (e.g. `~/.cache` on Linux, `~/Library/Caches` on
/// macOS)
#[cfg(feature = "analysis-download")]
pub fn cache_dir() -> std::path::PathBuf {
    match std::env::var_os("IMPACT_INDEX_CACHE") {
        Some(dir) => std::path::PathBuf::from(dir),
        None => dirs::cache_dir()
            .unwrap_or_else(std::env::temp_dir)
            .join("impact-index"),
    }
}

/// Bidirectional mapping between string terms and [`TermIndex`].
///
/// During building, uses a HashMap for O(1) insert/lookup.
/// On disk, uses a single FST file (term→id mapping).
pub struct Vocabulary {
    term_to_id: HashMap<String, TermIndex>,
    num_terms: usize,
}

impl Clone for Vocabulary {
    fn clone(&self) -> Self {
        Self {
            term_to_id: self.term_to_id.clone(),
            num_terms: self.num_terms,
        }
    }
}

impl Vocabulary {
    /// Create an empty vocabulary.
    pub fn new() -> Self {
        Self {
            term_to_id: HashMap::new(),
            num_terms: 0,
        }
    }

    /// Get the index for a term, inserting it if not present.
    pub fn get_or_insert(&mut self, term: &str) -> TermIndex {
        if let Some(&id) = self.term_to_id.get(term) {
            id
        } else {
            let id = self.num_terms;
            self.num_terms += 1;
            self.term_to_id.insert(term.to_string(), id);
            id
        }
    }

    /// Lookup a term's index (returns None if not present).
    pub fn get(&self, term: &str) -> Option<TermIndex> {
        self.term_to_id.get(term).copied()
    }

    /// Number of terms in the vocabulary.
    pub fn len(&self) -> usize {
        self.num_terms
    }

    /// Whether the vocabulary is empty.
    pub fn is_empty(&self) -> bool {
        self.num_terms == 0
    }

    /// Save vocabulary as a single FST file.
    pub fn save(&self, path: &Path) -> std::io::Result<()> {
        // Build FST: sorted (term, id) pairs
        let mut entries: Vec<(&str, u64)> = self
            .term_to_id
            .iter()
            .map(|(term, &id)| (term.as_str(), id as u64))
            .collect();
        entries.sort_by(|a, b| a.0.cmp(b.0));

        let fst_path = path.with_extension("fst");
        let wtr = BufWriter::new(File::create(&fst_path)?);
        let mut builder = fst::MapBuilder::new(wtr)
            .map_err(|e| std::io::Error::new(std::io::ErrorKind::Other, e.to_string()))?;
        for (term, id) in &entries {
            builder
                .insert(term.as_bytes(), *id)
                .map_err(|e| std::io::Error::new(std::io::ErrorKind::Other, e.to_string()))?;
        }
        builder
            .finish()
            .map_err(|e| std::io::Error::new(std::io::ErrorKind::Other, e.to_string()))?;

        Ok(())
    }

    /// Load vocabulary from FST file.
    pub fn load(path: &Path) -> std::io::Result<Self> {
        let fst_path = path.with_extension("fst");
        let fst_data = std::fs::read(&fst_path)?;
        let fst_map = fst::Map::new(fst_data)
            .map_err(|e| std::io::Error::new(std::io::ErrorKind::InvalidData, e.to_string()))?;

        let entries = fst_map
            .stream()
            .into_str_vec()
            .map_err(|e| std::io::Error::new(std::io::ErrorKind::InvalidData, e.to_string()))?;

        let num_terms = entries.len();
        let mut term_to_id = HashMap::with_capacity(num_terms);
        for (term, id) in entries {
            term_to_id.insert(term, id as TermIndex);
        }

        Ok(Self {
            term_to_id,
            num_terms,
        })
    }
}
