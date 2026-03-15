//! Text analysis pipeline: tokenize, lowercase, stem, vocabulary lookup.
//!
//! [`TextAnalyzer`] provides document and query analysis:
//! - Document analysis grows the vocabulary as new terms are encountered
//! - Query analysis does NOT grow the vocabulary (unknown terms are skipped)

use std::collections::{HashMap, HashSet};
use std::path::Path;

use serde::{Deserialize, Serialize};
use unicode_segmentation::UnicodeSegmentation;

use crate::base::TermIndex;

use super::stemmer::Stemmer;
use super::Vocabulary;

/// Analyzer configuration, serialized with the index for reproducibility.
#[derive(Serialize, Deserialize, Clone, Debug)]
pub struct AnalyzerConfig {
    /// Stemmer type: "porter", "snowball", or "none"
    pub stemmer: String,
    /// Language (for snowball stemmer and stop words)
    pub language: String,
    /// Whether stop words are enabled
    pub stop_words: bool,
    /// Whether English possessive filter is enabled (strip 's)
    pub english_possessive_filter: bool,
}

impl Default for AnalyzerConfig {
    fn default() -> Self {
        Self {
            stemmer: "none".to_string(),
            language: "english".to_string(),
            stop_words: false,
            english_possessive_filter: false,
        }
    }
}

/// Full text analysis pipeline matching Lucene's EnglishAnalyzer:
/// tokenize -> possessive filter -> lowercase -> stop words -> stem -> vocabulary lookup.
pub struct TextAnalyzer {
    vocab: Vocabulary,
    stemmer: Box<dyn Stemmer>,
    stop_words: HashSet<String>,
    /// Strip English possessives ('s) before tokenizing
    english_possessive_filter: bool,
    /// Analyzer config for serialization
    config: AnalyzerConfig,
}

impl TextAnalyzer {
    /// Create a new analyzer with the given stemmer (no stop words).
    pub fn new(stemmer: Box<dyn Stemmer>) -> Self {
        Self {
            vocab: Vocabulary::new(),
            stemmer,
            stop_words: HashSet::new(),
            english_possessive_filter: false,
            config: AnalyzerConfig::default(),
        }
    }

    /// Create a new analyzer with the given stemmer and stop words.
    pub fn with_stop_words(stemmer: Box<dyn Stemmer>, stop_words: &[&str]) -> Self {
        Self {
            vocab: Vocabulary::new(),
            stemmer,
            stop_words: stop_words.iter().map(|s| s.to_string()).collect(),
            english_possessive_filter: false,
            config: AnalyzerConfig::default(),
        }
    }

    /// Create from an existing vocabulary and stemmer.
    pub fn with_vocab(vocab: Vocabulary, stemmer: Box<dyn Stemmer>) -> Self {
        Self {
            vocab,
            stemmer,
            stop_words: HashSet::new(),
            english_possessive_filter: false,
            config: AnalyzerConfig::default(),
        }
    }

    /// Create from an existing vocabulary, stemmer, and stop words.
    pub fn with_vocab_and_stop_words(
        vocab: Vocabulary,
        stemmer: Box<dyn Stemmer>,
        stop_words: &[&str],
    ) -> Self {
        Self {
            vocab,
            stemmer,
            stop_words: stop_words.iter().map(|s| s.to_string()).collect(),
            english_possessive_filter: false,
            config: AnalyzerConfig::default(),
        }
    }

    /// Enable English possessive filter (strip 's from tokens).
    /// This matches Lucene's EnglishPossessiveFilter.
    pub fn set_english_possessive_filter(&mut self, enabled: bool) {
        self.english_possessive_filter = enabled;
        self.config.english_possessive_filter = enabled;
    }

    /// Set the analyzer config (for serialization).
    pub fn set_config(&mut self, config: AnalyzerConfig) {
        self.config = config;
    }

    /// Get the analyzer config.
    pub fn config(&self) -> &AnalyzerConfig {
        &self.config
    }

    /// Tokenize text using UAX#29 word boundaries (matching Lucene's StandardTokenizer),
    /// then apply possessive filter, lowercase, and stop words.
    fn tokenize(&self, text: &str) -> Vec<String> {
        text.unicode_words()
            .map(|t| {
                let lowered = t.to_lowercase();
                if self.english_possessive_filter {
                    // Strip trailing 's with straight or curly apostrophe
                    // Matches Lucene's EnglishPossessiveFilter which handles
                    // U+0027 ('), U+2019 ('), and U+FF07 (')
                    if lowered.ends_with("'s")
                        || lowered.ends_with("\u{2019}s")
                        || lowered.ends_with("\u{ff07}s")
                    {
                        // Remove last 2 chars (apostrophe + s)
                        let end = lowered.len() - "'s".len();
                        // Curly apostrophes are multi-byte, find the right cut point
                        let cut = lowered
                            .rfind(|c: char| c == '\'' || c == '\u{2019}' || c == '\u{ff07}')
                            .unwrap_or(end);
                        return lowered[..cut].to_string();
                    }
                }
                lowered
            })
            .filter(|s| !s.is_empty() && !self.stop_words.contains(s))
            .collect()
    }

    /// Analyze document text: tokenize, stem, compute TF, grow vocabulary.
    ///
    /// Returns `(term_indices, tf_values)` suitable for indexing.
    pub fn analyze_doc(&mut self, text: &str) -> (Vec<TermIndex>, Vec<f32>) {
        let tokens = self.tokenize(text);

        // Count term frequencies
        let mut tf_map: HashMap<String, f32> = HashMap::new();
        for token in &tokens {
            let stemmed = self.stemmer.stem(token);
            *tf_map.entry(stemmed).or_insert(0.0) += 1.0;
        }

        // Convert to term indices (growing vocabulary)
        let mut term_indices = Vec::with_capacity(tf_map.len());
        let mut tf_values = Vec::with_capacity(tf_map.len());
        for (term, tf) in tf_map {
            let idx = self.vocab.get_or_insert(&term);
            term_indices.push(idx);
            tf_values.push(tf);
        }

        (term_indices, tf_values)
    }

    /// Analyze query text: tokenize, stem, lookup in vocabulary.
    ///
    /// Does NOT grow vocabulary — unknown terms are skipped.
    /// Returns a map from TermIndex to TF (for boosting).
    pub fn analyze_query(&self, text: &str) -> HashMap<TermIndex, f32> {
        let tokens = self.tokenize(text);
        let mut query: HashMap<TermIndex, f32> = HashMap::new();

        for token in &tokens {
            let stemmed = self.stemmer.stem(token);
            if let Some(idx) = self.vocab.get(&stemmed) {
                *query.entry(idx).or_insert(0.0) += 1.0;
            }
        }

        query
    }

    /// Tokenize and stem text without vocabulary insertion (thread-safe).
    ///
    /// Returns a list of (stemmed_token, tf) pairs. This can be called
    /// from multiple threads since it only reads the stemmer and stop words.
    pub fn tokenize_and_stem(&self, text: &str) -> Vec<(String, f32)> {
        let tokens = self.tokenize(text);
        let mut tf_map: std::collections::HashMap<String, f32> = std::collections::HashMap::new();
        for token in &tokens {
            let stemmed = self.stemmer.stem(token);
            *tf_map.entry(stemmed).or_insert(0.0) += 1.0;
        }
        tf_map.into_iter().collect()
    }

    /// Get a reference to the vocabulary.
    pub fn vocab(&self) -> &Vocabulary {
        &self.vocab
    }

    /// Get a mutable reference to the vocabulary.
    pub fn vocab_mut(&mut self) -> &mut Vocabulary {
        &mut self.vocab
    }

    /// Save the vocabulary and analyzer config to the given directory.
    pub fn save(&self, dir: &Path) -> std::io::Result<()> {
        self.vocab.save(&dir.join("vocab"))?;
        let config_file = std::fs::File::options()
            .write(true)
            .create(true)
            .truncate(true)
            .open(dir.join("analyzer.cbor"))?;
        ciborium::ser::into_writer(&self.config, config_file)
            .map_err(|e| std::io::Error::new(std::io::ErrorKind::Other, e.to_string()))
    }

    /// Save only the vocabulary (backward compatible).
    pub fn save_vocab(&self, dir: &Path) -> std::io::Result<()> {
        self.save(dir)
    }

    /// Check if analyzer files exist in a directory.
    pub fn exists(dir: &Path) -> bool {
        dir.join("vocab.fst").exists()
    }

    /// Copy analyzer files from source to destination directory.
    pub fn copy_files(src_dir: &Path, dst_dir: &Path) -> std::io::Result<()> {
        for filename in &["vocab.fst", "analyzer.cbor"] {
            let src = src_dir.join(filename);
            let dst = dst_dir.join(filename);
            if src.exists() {
                if std::fs::hard_link(&src, &dst).is_err() {
                    std::fs::copy(&src, &dst)?;
                }
            }
        }
        Ok(())
    }

    /// Load vocabulary from `vocab.cbor` in the given directory, with a stemmer.
    pub fn load(dir: &Path, stemmer: Box<dyn Stemmer>) -> std::io::Result<Self> {
        let vocab = Vocabulary::load(&dir.join("vocab"))?;
        let config = Self::load_config(dir);
        Ok(Self {
            vocab,
            stemmer,
            stop_words: HashSet::new(),
            english_possessive_filter: config.english_possessive_filter,
            config,
        })
    }

    /// Load vocabulary with stop words.
    pub fn load_with_stop_words(
        dir: &Path,
        stemmer: Box<dyn Stemmer>,
        stop_words: &[&str],
    ) -> std::io::Result<Self> {
        let vocab = Vocabulary::load(&dir.join("vocab"))?;
        let config = Self::load_config(dir);
        Ok(Self {
            vocab,
            stemmer,
            stop_words: stop_words.iter().map(|s| s.to_string()).collect(),
            english_possessive_filter: config.english_possessive_filter,
            config,
        })
    }

    /// Load analyzer config if available, otherwise return default.
    pub fn load_config(dir: &Path) -> AnalyzerConfig {
        let config_path = dir.join("analyzer.cbor");
        if config_path.exists() {
            if let Ok(file) = std::fs::File::open(&config_path) {
                if let Ok(config) = ciborium::de::from_reader(file) {
                    return config;
                }
            }
        }
        AnalyzerConfig::default()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::vocab::stemmer::NoStemmer;

    #[test]
    fn test_analyze_doc() {
        let mut analyzer = TextAnalyzer::new(Box::new(NoStemmer));
        let (terms, values) = analyzer.analyze_doc("the quick brown fox the quick");

        // Should have 4 unique terms
        assert_eq!(terms.len(), 4);
        assert_eq!(values.len(), 4);

        // "the" and "quick" should have tf=2
        let the_idx = analyzer.vocab().get("the").unwrap();
        let quick_idx = analyzer.vocab().get("quick").unwrap();
        let pos_the = terms.iter().position(|&t| t == the_idx).unwrap();
        let pos_quick = terms.iter().position(|&t| t == quick_idx).unwrap();
        assert_eq!(values[pos_the], 2.0);
        assert_eq!(values[pos_quick], 2.0);
    }

    #[test]
    fn test_possessive_filter() {
        let mut analyzer = TextAnalyzer::new(Box::new(NoStemmer));
        analyzer.set_english_possessive_filter(true);
        let tokens = analyzer.tokenize("king's castle children's books it's fine");
        assert!(tokens.contains(&"king".to_string()));
        assert!(tokens.contains(&"castle".to_string()));
        assert!(tokens.contains(&"children".to_string()));
        assert!(tokens.contains(&"it".to_string()));
        assert!(!tokens.iter().any(|t| t.contains("'s")));
    }

    #[test]
    fn test_possessive_filter_curly_apostrophe() {
        let mut analyzer = TextAnalyzer::new(Box::new(NoStemmer));
        analyzer.set_english_possessive_filter(true);
        // U+2019 RIGHT SINGLE QUOTATION MARK (curly apostrophe)
        let tokens = analyzer.tokenize("Canada\u{2019}s Tower children\u{2019}s books");
        assert!(
            tokens.contains(&"canada".to_string()),
            "Should strip curly apostrophe possessive: got {:?}",
            tokens
        );
        assert!(tokens.contains(&"children".to_string()));
        assert!(tokens.contains(&"tower".to_string()));
    }

    #[test]
    fn test_apostrophe_in_words() {
        let mut analyzer = TextAnalyzer::new(Box::new(NoStemmer));
        let tokens = analyzer.tokenize("don't worry O'Brien");
        // Apostrophes within words are kept
        assert!(tokens.contains(&"don't".to_string()));
        assert!(tokens.contains(&"o'brien".to_string()));
    }

    #[test]
    fn test_periods_in_numbers() {
        let analyzer = TextAnalyzer::new(Box::new(NoStemmer));
        let tokens = analyzer.tokenize("price is 3.14 dollars U.S.A.");
        assert!(tokens.contains(&"3.14".to_string()));
        assert!(tokens.contains(&"u.s.a".to_string()));
    }

    #[test]
    fn test_analyze_query_no_growth() {
        let mut analyzer = TextAnalyzer::new(Box::new(NoStemmer));
        let _ = analyzer.analyze_doc("hello world");
        let vocab_size_before = analyzer.vocab().len();

        let query = analyzer.analyze_query("hello unknown");
        // Vocabulary should not grow
        assert_eq!(analyzer.vocab().len(), vocab_size_before);
        // Only "hello" should be in the query
        assert_eq!(query.len(), 1);
        let hello_idx = analyzer.vocab().get("hello").unwrap();
        assert!(query.contains_key(&hello_idx));
    }
}
