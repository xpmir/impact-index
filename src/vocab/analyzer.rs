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

/// Stem each stop word so it can be matched against already-stemmed tokens.
/// See [`TextAnalyzer::stemmed_stop_words`] for why this exists.
fn stem_stop_words(stemmer: &dyn Stemmer, stop_words: &[&str]) -> HashSet<String> {
    stop_words
        .iter()
        .map(|w| stemmer.stem(&w.to_lowercase()))
        .collect()
}

/// Tokenizer variant governing word-splitting rules.
///
/// `Standard` splits on UAX#29 word boundaries only. `LuceneEnglish` does
/// the same splitting, then additionally strips a trailing English
/// possessive ('s) before lowercasing -- matching Lucene's `EnglishAnalyzer`
/// tokenizer chain (`StandardTokenizer` -> `EnglishPossessiveFilter`). This
/// is a tokenizer-stage concern, not a stemming one: Porter/Snowball are
/// defined on already-clean word forms and never see the apostrophe, so
/// `LuceneEnglish` can be paired with any stemmer (or none).
#[derive(Serialize, Deserialize, Clone, Copy, Debug, PartialEq, Eq, Default)]
#[serde(rename_all = "kebab-case")]
pub enum Tokenizer {
    #[default]
    Standard,
    LuceneEnglish,
}

/// Analyzer configuration, serialized with the index for reproducibility.
#[derive(Serialize, Deserialize, Clone, Debug)]
pub struct AnalyzerConfig {
    /// Stemmer type: "porter", "snowball", or "none"
    pub stemmer: String,
    /// Language (for snowball stemmer and stop words)
    pub language: String,
    /// Whether stop words are enabled. Kept for backward compatibility with
    /// indices built before `stop_words_list` existed; `stop_words_list`
    /// takes precedence whenever it's non-empty.
    pub stop_words: bool,
    /// The actual stop word list that was used, so query-time analysis
    /// (`PyTextAnalyzer::from_index`) can reconstruct the *exact* list a
    /// custom (non-default) `stop_words=[...]` build used, instead of
    /// silently substituting the language's built-in default list.
    /// `#[serde(default)]` so indices built before this field existed still
    /// deserialize (as an empty list, falling back to the old `stop_words`
    /// bool + built-in-list behavior).
    #[serde(default)]
    pub stop_words_list: Vec<String>,
    /// Deprecated, superseded by `tokenizer`. Still written (kept in sync
    /// with `tokenizer`) so indices built before `tokenizer` existed still
    /// deserialize correctly; read via [`Self::effective_tokenizer`], never
    /// directly.
    pub english_possessive_filter: bool,
    /// Tokenizer variant (word-splitting rules). `#[serde(default)]` so
    /// older indices (no `tokenizer` field) deserialize as `Standard` here
    /// and fall back to the legacy `english_possessive_filter` bool via
    /// [`Self::effective_tokenizer`].
    #[serde(default)]
    pub tokenizer: Tokenizer,
}

impl Default for AnalyzerConfig {
    fn default() -> Self {
        Self {
            stemmer: "none".to_string(),
            language: "english".to_string(),
            stop_words: false,
            stop_words_list: Vec::new(),
            english_possessive_filter: false,
            tokenizer: Tokenizer::Standard,
        }
    }
}

impl AnalyzerConfig {
    /// The tokenizer variant actually in effect, falling back to the
    /// legacy `english_possessive_filter` bool for indices built before
    /// `tokenizer` existed.
    pub fn effective_tokenizer(&self) -> Tokenizer {
        if self.tokenizer == Tokenizer::LuceneEnglish || self.english_possessive_filter {
            Tokenizer::LuceneEnglish
        } else {
            Tokenizer::Standard
        }
    }
}

/// Full text analysis pipeline matching Lucene's EnglishAnalyzer:
/// tokenize -> possessive filter -> lowercase -> stop words -> stem -> stop
/// words (stemmed) -> vocabulary lookup. See [`TextAnalyzer::stemmed_stop_words`]
/// for why stop words are checked twice.
pub struct TextAnalyzer {
    vocab: Vocabulary,
    stemmer: Box<dyn Stemmer>,
    stop_words: HashSet<String>,
    /// Stemmed form of each entry in `stop_words`, checked *after* stemming
    /// a surviving token. Stop word removal itself always happens before
    /// stemming (on the raw token) -- that's the correct order, since
    /// stemming a word first and then comparing against raw stop words risks
    /// coincidentally dropping legitimate content words. But an inflected or
    /// misspelled variant of a stop word (e.g. "whats" instead of "what's")
    /// won't exact-match the raw stop word list, survives that first filter,
    /// and can then stem right back down to the stop word itself, quietly
    /// re-admitting it into the vocabulary. This second, stem-aware check
    /// catches exactly (and only) that case.
    stemmed_stop_words: HashSet<String>,
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
            stemmed_stop_words: HashSet::new(),
            english_possessive_filter: false,
            config: AnalyzerConfig::default(),
        }
    }

    /// Create a new analyzer with the given stemmer and stop words.
    pub fn with_stop_words(stemmer: Box<dyn Stemmer>, stop_words: &[&str]) -> Self {
        let stemmed_stop_words = stem_stop_words(stemmer.as_ref(), stop_words);
        Self {
            vocab: Vocabulary::new(),
            stemmer,
            stop_words: stop_words.iter().map(|s| s.to_string()).collect(),
            stemmed_stop_words,
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
            stemmed_stop_words: HashSet::new(),
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
        let stemmed_stop_words = stem_stop_words(stemmer.as_ref(), stop_words);
        Self {
            vocab,
            stemmer,
            stop_words: stop_words.iter().map(|s| s.to_string()).collect(),
            stemmed_stop_words,
            english_possessive_filter: false,
            config: AnalyzerConfig::default(),
        }
    }

    /// Enable English possessive filter (strip 's from tokens).
    /// This matches Lucene's EnglishPossessiveFilter.
    pub fn set_english_possessive_filter(&mut self, enabled: bool) {
        self.set_tokenizer(if enabled {
            Tokenizer::LuceneEnglish
        } else {
            Tokenizer::Standard
        });
    }

    /// Set the tokenizer variant (word-splitting rules). See [`Tokenizer`].
    pub fn set_tokenizer(&mut self, tokenizer: Tokenizer) {
        self.english_possessive_filter = tokenizer == Tokenizer::LuceneEnglish;
        self.config.tokenizer = tokenizer;
        self.config.english_possessive_filter = self.english_possessive_filter;
    }

    /// Set the analyzer config (for serialization).
    pub fn set_config(&mut self, config: AnalyzerConfig) {
        self.config = config;
    }

    /// Get the analyzer config.
    pub fn config(&self) -> &AnalyzerConfig {
        &self.config
    }

    /// Lowercase a token and, if enabled, strip a trailing English
    /// possessive ('s with straight or curly apostrophe). Matches Lucene's
    /// EnglishPossessiveFilter, which handles U+0027 ('), U+2019 ('), and
    /// U+FF07 (').
    fn normalize_token(&self, t: &str) -> String {
        let lowered = t.to_lowercase();
        if self.english_possessive_filter {
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
    }

    /// Tokenize text using UAX#29 word boundaries (matching Lucene's
    /// StandardTokenizer), then apply possessive filter, lowercase, and
    /// stop words -- keeping each surviving token's pre-filtering index as
    /// its position. This creates position gaps across removed stopwords
    /// (Lucene's position-increment behavior), which is required for
    /// correct phrase semantics: `#1(new york)` must not match "new the
    /// york".
    fn tokenize_indexed(&self, text: &str) -> Vec<(String, u32)> {
        text.unicode_words()
            .enumerate()
            .map(|(i, t)| (self.normalize_token(t), i as u32))
            .filter(|(s, _)| !s.is_empty() && !self.stop_words.contains(s))
            .collect()
    }

    /// Tokenize text using UAX#29 word boundaries (matching Lucene's StandardTokenizer),
    /// then apply possessive filter, lowercase, and stop words.
    fn tokenize(&self, text: &str) -> Vec<String> {
        self.tokenize_indexed(text)
            .into_iter()
            .map(|(t, _)| t)
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
            if self.stemmed_stop_words.contains(&stemmed) {
                continue;
            }
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

    /// Analyze document text like [`Self::analyze_doc`], but also record
    /// each stemmed term's token positions instead of collapsing them to a
    /// count.
    ///
    /// Returns `(term_index, positions)` pairs, one per distinct term.
    /// `positions` is sorted ascending (tokens are processed in text
    /// order); tf for a term is `positions.len()`.
    pub fn analyze_doc_positional(&mut self, text: &str) -> Vec<(TermIndex, Vec<u32>)> {
        let tokens = self.tokenize_indexed(text);

        let mut positions_map: HashMap<String, Vec<u32>> = HashMap::new();
        for (token, pos) in tokens {
            let stemmed = self.stemmer.stem(&token);
            if self.stemmed_stop_words.contains(&stemmed) {
                continue;
            }
            positions_map.entry(stemmed).or_default().push(pos);
        }

        positions_map
            .into_iter()
            .map(|(term, positions)| (self.vocab.get_or_insert(&term), positions))
            .collect()
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
            if self.stemmed_stop_words.contains(&stemmed) {
                continue;
            }
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
            if self.stemmed_stop_words.contains(&stemmed) {
                continue;
            }
            *tf_map.entry(stemmed).or_insert(0.0) += 1.0;
        }
        tf_map.into_iter().collect()
    }

    /// Thread-safe, position-preserving variant of [`Self::tokenize_and_stem`]
    /// (no vocabulary insertion), for the parallel batch path
    /// ([`crate::bow::BOWIndexBuilder::add_texts_batch`]) when positions
    /// are enabled.
    ///
    /// Returns a list of `(stemmed_token, positions)` pairs.
    pub fn tokenize_and_stem_positional(&self, text: &str) -> Vec<(String, Vec<u32>)> {
        let tokens = self.tokenize_indexed(text);
        let mut positions_map: HashMap<String, Vec<u32>> = HashMap::new();
        for (token, pos) in tokens {
            let stemmed = self.stemmer.stem(&token);
            if self.stemmed_stop_words.contains(&stemmed) {
                continue;
            }
            positions_map.entry(stemmed).or_default().push(pos);
        }
        positions_map.into_iter().collect()
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
            stemmed_stop_words: HashSet::new(),
            english_possessive_filter: config.effective_tokenizer() == Tokenizer::LuceneEnglish,
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
        let stemmed_stop_words = stem_stop_words(stemmer.as_ref(), stop_words);
        Ok(Self {
            vocab,
            stemmer,
            stop_words: stop_words.iter().map(|s| s.to_string()).collect(),
            stemmed_stop_words,
            english_possessive_filter: config.effective_tokenizer() == Tokenizer::LuceneEnglish,
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
    fn test_stop_word_survives_inflection_then_stem() {
        // "whats" (no apostrophe -- common informal spelling) isn't an exact
        // match for the raw stop word "what", so it survives the pre-stem
        // filter -- but Snowball/Porter2 stems it right back down to "what",
        // which must then be caught by the post-stem check, not silently
        // re-admitted into the vocabulary.
        use crate::vocab::stemmer::SnowballStemmer;
        let stemmer = SnowballStemmer::new("english").unwrap();
        let mut analyzer = TextAnalyzer::with_stop_words(Box::new(stemmer), &["what", "is", "the"]);

        let (terms, _) = analyzer.analyze_doc("whats next for the show");
        assert!(
            analyzer.vocab().get("what").is_none(),
            "'whats' must not stem back into the stop word 'what': vocab = {:?}",
            terms
        );

        // Query-side must reject it too, even against an index that (before
        // this fix) already has "what" polluted into its vocabulary.
        let query = analyzer.analyze_query("what");
        assert!(
            query.is_empty(),
            "querying a stop word must return no terms"
        );
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

#[cfg(test)]
mod tokenizer_compat_tests {
    use super::*;

    #[test]
    fn old_config_without_tokenizer_field_falls_back_to_bool() {
        // Simulates an `analyzer.cbor` written before `tokenizer` existed:
        // no `tokenizer` key at all, only the legacy bool.
        #[derive(Serialize)]
        struct OldAnalyzerConfig {
            stemmer: String,
            language: String,
            stop_words: bool,
            english_possessive_filter: bool,
        }
        let old = OldAnalyzerConfig {
            stemmer: "porter".to_string(),
            language: "english".to_string(),
            stop_words: true,
            english_possessive_filter: true,
        };
        let mut bytes = Vec::new();
        ciborium::ser::into_writer(&old, &mut bytes).unwrap();
        let config: AnalyzerConfig = ciborium::de::from_reader(bytes.as_slice()).unwrap();

        assert!(config.stop_words_list.is_empty());
        assert_eq!(config.tokenizer, Tokenizer::Standard);
        assert_eq!(config.effective_tokenizer(), Tokenizer::LuceneEnglish);
    }
}
