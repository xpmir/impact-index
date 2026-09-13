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

/// Which pipeline stage(s) stop words are checked at.
///
/// impact-index is meant to match two different reference pipelines
/// depending on the requested stop-word family, and they disagree about
/// *when* stop words are removed relative to stemming:
///
/// - Real Lucene (`EnglishAnalyzer`, wrapped by Anserini/Pyserini):
///   `StopFilter` runs before `PorterStemFilter` -- stop words are checked
///   exactly once, pre-stem.
/// - Real PISA (`pisa-engine/pisa`, compared against for the
///   Terrier-aligned pipeline): `tools/app.cpp`'s `Analyzer::text_analyzer`
///   appends `StopWordRemover` *after* the stemmer, and compares the
///   already-stemmed token against the literal, never-separately-stemmed
///   word list -- stop words are checked exactly once, post-stem, against
///   the raw list.
///
/// Neither real system checks both stages. impact-index historically did
/// (see `dcc4b20`), which is only correct for an arbitrary custom list with
/// no declared family -- see [`Self::Both`].
#[derive(Serialize, Deserialize, Clone, Copy, Debug, PartialEq, Eq, Default)]
#[serde(rename_all = "kebab-case")]
pub enum StopWordFilterMode {
    /// Check the raw (pre-stem) token only. Matches real Lucene: exactly
    /// one `StopFilter` pass, before stemming.
    PreStem,
    /// Check the stemmed token, against the *raw* (unstemmed) stop word
    /// set. Matches real PISA: exactly one pass, after stemming, against
    /// the literal word list (not a separately pre-stemmed copy of it).
    PostStem,
    /// Check both: the raw token pre-stem, then the stemmed token against
    /// a pre-stemmed copy of the list post-stem. This is the historical
    /// (pre-family) impact-index behavior. Kept as the default for a
    /// custom `stop_words=[...]` list -- there's no single correct
    /// reference pipeline for an arbitrary list -- and for any
    /// `AnalyzerConfig` serialized before this enum existed, so on-disk
    /// indices keep reproducing the results they were actually built with.
    #[default]
    Both,
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
    /// Which stop word family (`"lucene"` or `"terrier"`) `stop_words_list`
    /// was resolved from, e.g. `stop_words="terrier"` on `BOWIndexBuilder`.
    /// `None` for a custom `stop_words=[...]` list, no stop words, or an
    /// index built before this field existed. Purely informational: reload
    /// always uses the exact words in `stop_words_list` (or, for the oldest
    /// configs predating that field, falls back to the Lucene list for
    /// `language` -- see `PyTextAnalyzer::from_index`), so a missing family
    /// never changes which words are actually applied.
    #[serde(default)]
    pub stop_words_family: Option<String>,
    /// Which pipeline stage(s) stop words are filtered at (see
    /// [`StopWordFilterMode`]). `#[serde(default)]` (-> [`StopWordFilterMode::Both`])
    /// so any config serialized before this field existed keeps reproducing
    /// the old "check both" behavior on reload -- this is a real behavior
    /// change for the `"lucene"`/`"terrier"` presets, and existing on-disk
    /// indices built with them must not silently change query results.
    #[serde(default)]
    pub stop_words_filter_mode: StopWordFilterMode,
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
            stop_words_family: None,
            stop_words_filter_mode: StopWordFilterMode::Both,
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

/// Text analysis pipeline: tokenize -> possessive filter -> lowercase ->
/// [pre-stem stop words] -> stem -> [post-stem stop words] -> vocabulary
/// lookup. Which of the two stop-word checks actually run is governed by
/// [`StopWordFilterMode`] (see [`Self::is_pre_stem_stop_word`] and
/// [`Self::is_post_stem_stop_word`]) -- matching Lucene requires only the
/// pre-stem check, matching PISA/Terrier only the post-stem one, and an
/// arbitrary custom list runs both, for backward compatibility.
pub struct TextAnalyzer {
    vocab: Vocabulary,
    stemmer: Box<dyn Stemmer>,
    stop_words: HashSet<String>,
    /// Stemmed form of each entry in `stop_words`, used by the post-stem
    /// check only in [`StopWordFilterMode::Both`] mode (custom lists / old
    /// configs) -- see [`Self::is_post_stem_stop_word`]. An inflected or
    /// misspelled variant of a stop word (e.g. "whats" instead of "what's")
    /// won't exact-match the raw stop word list, survives the pre-stem
    /// filter, and can then stem right back down to the stop word itself;
    /// this stem-aware check catches exactly (and only) that case.
    /// [`StopWordFilterMode::PostStem`] mode (Terrier/PISA) deliberately
    /// does *not* use this set -- it checks the stemmed token against the
    /// raw `stop_words` set instead, matching PISA's literal C++ behavior.
    stemmed_stop_words: HashSet<String>,
    /// Which pipeline stage(s) stop words are actually checked at. Defaults
    /// to [`StopWordFilterMode::Both`] (the historical behavior) in every
    /// constructor; callers that know their family (`BOWIndexBuilder`'s
    /// `"lucene"`/`"terrier"` presets) narrow it via
    /// [`Self::set_stop_words_filter_mode`].
    filter_mode: StopWordFilterMode,
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
            filter_mode: StopWordFilterMode::Both,
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
            filter_mode: StopWordFilterMode::Both,
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
            filter_mode: StopWordFilterMode::Both,
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
            filter_mode: StopWordFilterMode::Both,
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

    /// Set which pipeline stage(s) stop words are actually checked at. See
    /// [`StopWordFilterMode`]. Also mirrored into `self.config` so it's
    /// persisted with the index.
    pub fn set_stop_words_filter_mode(&mut self, mode: StopWordFilterMode) {
        self.filter_mode = mode;
        self.config.stop_words_filter_mode = mode;
    }

    /// Whether the raw (pre-stem) `token` should be dropped as a stop word.
    /// Applies for [`StopWordFilterMode::PreStem`] and
    /// [`StopWordFilterMode::Both`] -- not for
    /// [`StopWordFilterMode::PostStem`] (Terrier/PISA), which checks only
    /// after stemming, in [`Self::is_post_stem_stop_word`].
    fn is_pre_stem_stop_word(&self, token: &str) -> bool {
        matches!(
            self.filter_mode,
            StopWordFilterMode::PreStem | StopWordFilterMode::Both
        ) && self.stop_words.contains(token)
    }

    /// Whether an already-stemmed token should be dropped as a stop word.
    /// Mode-dependent (see [`StopWordFilterMode`]):
    /// - `PreStem` (Lucene): never -- Lucene checks stop words exactly once,
    ///   pre-stem, in [`Self::is_pre_stem_stop_word`].
    /// - `PostStem` (Terrier/PISA): the stemmed token against the *raw*
    ///   `stop_words` set -- PISA's `StopWordRemover` runs after its
    ///   stemmer and compares against the literal, never-separately-stemmed
    ///   word list.
    /// - `Both` (custom list, or a config predating this enum): the stemmed
    ///   token against `stemmed_stop_words`, a pre-stemmed copy of the list
    ///   -- the historical impact-index behavior, unchanged for callers
    ///   without a named family.
    fn is_post_stem_stop_word(&self, stemmed: &str) -> bool {
        match self.filter_mode {
            StopWordFilterMode::PreStem => false,
            StopWordFilterMode::PostStem => self.stop_words.contains(stemmed),
            StopWordFilterMode::Both => self.stemmed_stop_words.contains(stemmed),
        }
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
    /// StandardTokenizer), then apply possessive filter, lowercase, and the
    /// pre-stem stop word check (see [`Self::is_pre_stem_stop_word`]) --
    /// keeping each surviving token's pre-filtering index as its position.
    /// This creates position gaps across removed stopwords (Lucene's
    /// position-increment behavior), which is required for correct phrase
    /// semantics: `#1(new york)` must not match "new the york".
    fn tokenize_indexed(&self, text: &str) -> Vec<(String, u32)> {
        text.unicode_words()
            .enumerate()
            .map(|(i, t)| (self.normalize_token(t), i as u32))
            .filter(|(s, _)| !s.is_empty() && !self.is_pre_stem_stop_word(s))
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
            if self.is_post_stem_stop_word(&stemmed) {
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
            if self.is_post_stem_stop_word(&stemmed) {
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
            if self.is_post_stem_stop_word(&stemmed) {
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
            if self.is_post_stem_stop_word(&stemmed) {
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
            if self.is_post_stem_stop_word(&stemmed) {
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
            // `Both`, not `config.stop_words_filter_mode`: this constructor
            // takes no stop words at all, so the mode is moot here; kept
            // consistent with `with_stop_words`'s default below.
            filter_mode: StopWordFilterMode::Both,
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
            // `Both`, not `config.stop_words_filter_mode`: `stop_words` here
            // is an explicit caller-supplied list (this constructor has no
            // way to know whether it matches a named family), so it gets
            // the same safe, conservative default as any other custom list
            // -- see `StopWordFilterMode::Both`. Callers that DO have a
            // config-derived mode to apply (`PyTextAnalyzer::from_index`)
            // set it explicitly afterward via `set_stop_words_filter_mode`.
            filter_mode: StopWordFilterMode::Both,
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
mod stop_word_filter_mode_tests {
    use super::*;
    use crate::vocab::stemmer::{SnowballStemmer, Stemmer};

    /// A toy stemmer with a fully controlled, non-linguistic mapping, so
    /// [`terrier_mode_checks_raw_list_not_stemmed_copy`] doesn't depend on
    /// finding a real English word with the right accidental stemming
    /// quirk.
    struct ToyStemmer;
    impl Stemmer for ToyStemmer {
        fn stem(&self, word: &str) -> String {
            match word {
                // "wolf" (the stop word itself) does NOT stem to itself --
                // this is the "stopword's own stem differs from itself"
                // setup the test needs.
                "wolf" => "wolv".to_string(),
                // A distinct token that stems to the *raw* stop word text.
                "wolves" => "wolf".to_string(),
                other => other.to_string(),
            }
        }
    }

    /// Lucene family (`PreStem` mode): a token whose raw form is NOT a
    /// stop word but whose STEM collides with one must be KEPT. This is a
    /// real, intentional reversal of `dcc4b20`'s effect for this family --
    /// the post-stem check must not run at all for Lucene.
    #[test]
    fn lucene_mode_keeps_stem_collision() {
        let stemmer = SnowballStemmer::new("english").unwrap();
        assert_eq!(stemmer.stem("whats"), "what", "test assumption");

        let mut analyzer = TextAnalyzer::with_stop_words(Box::new(stemmer), &["what", "is", "the"]);
        analyzer.set_stop_words_filter_mode(StopWordFilterMode::PreStem);

        let _ = analyzer.analyze_doc("whats next for the show");
        assert!(
            analyzer.vocab().get("what").is_some(),
            "Lucene mode must keep 'whats' -> 'what': it isn't a raw stop \
             word, and the post-stem check must not run for this family"
        );

        // Query side must match: a stemmed collision is a real, queryable
        // vocabulary term under this family.
        let query = analyzer.analyze_query("whats");
        assert!(
            !query.is_empty(),
            "querying 'whats' must find the term kept under Lucene mode"
        );
    }

    /// Terrier family (`PostStem` mode): the identical case IS removed --
    /// the post-stem check (against the raw list) catches it, matching
    /// PISA.
    #[test]
    fn terrier_mode_removes_stem_collision() {
        let stemmer = SnowballStemmer::new("english").unwrap();
        assert_eq!(stemmer.stem("whats"), "what", "test assumption");

        let mut analyzer = TextAnalyzer::with_stop_words(Box::new(stemmer), &["what", "is", "the"]);
        analyzer.set_stop_words_filter_mode(StopWordFilterMode::PostStem);

        let (terms, _) = analyzer.analyze_doc("whats next for the show");
        assert!(
            analyzer.vocab().get("what").is_none(),
            "Terrier mode must drop 'whats' -> 'what': vocab = {:?}",
            terms
        );

        let query = analyzer.analyze_query("what");
        assert!(
            query.is_empty(),
            "querying a stop word must return no terms under Terrier mode"
        );
    }

    /// Terrier family's post-stem check must compare against the RAW stop
    /// word set, not a separately pre-stemmed copy (the old
    /// `stemmed_stop_words` behavior) -- this is what distinguishes the fix.
    /// Uses [`ToyStemmer`] so the stop word's own stem deliberately differs
    /// from itself: "wolf" (raw stop word) stems to "wolv", while the
    /// document token "wolves" stems to "wolf" (the literal raw stop word
    /// text, not its stemmed form "wolv").
    #[test]
    fn terrier_mode_checks_raw_list_not_stemmed_copy() {
        let mut analyzer = TextAnalyzer::with_stop_words(Box::new(ToyStemmer), &["wolf"]);
        analyzer.set_stop_words_filter_mode(StopWordFilterMode::PostStem);

        let (terms, _) = analyzer.analyze_doc("wolves howl at night");
        assert!(
            analyzer.vocab().get("wolf").is_none(),
            "'wolves' stems to the raw stop word 'wolf' and must be \
             dropped under Terrier mode: vocab = {:?}",
            terms
        );
        // Sanity check this isn't accidentally passing some other way: the
        // stemmed value "wolv" (which the old `stemmed_stop_words` set
        // would have held) is not itself queryable as a stop word, proving
        // the check ran against "wolf" (raw), not "wolv" (stemmed copy).
        assert!(
            analyzer.vocab().get("wolv").is_none(),
            "no document token stems to 'wolv', so it should never enter \
             the vocabulary either way: vocab = {:?}",
            terms
        );
    }

    /// A custom (non-family) stop word list must keep today's existing
    /// behavior: both the pre-stem and post-stem checks apply. Regression
    /// test for the default `StopWordFilterMode::Both` (see also
    /// `test_stop_word_survives_inflection_then_stem` above, which covers
    /// the same guarantee via the plain, mode-agnostic constructor).
    #[test]
    fn custom_list_keeps_both_checks_by_default() {
        let stemmer = SnowballStemmer::new("english").unwrap();
        let analyzer = TextAnalyzer::with_stop_words(Box::new(stemmer), &["what", "is", "the"]);
        assert_eq!(
            analyzer.filter_mode,
            StopWordFilterMode::Both,
            "a custom list with no declared family must default to Both"
        );

        let mut analyzer = analyzer;
        let (terms, _) = analyzer.analyze_doc("whats next for the show");
        assert!(
            analyzer.vocab().get("what").is_none(),
            "custom list (Both mode) must still catch the stemmed \
             collision, unchanged from before this fix: vocab = {:?}",
            terms
        );
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

#[cfg(test)]
mod stop_words_family_compat_tests {
    use super::*;

    #[test]
    fn old_config_without_family_field_deserializes_as_none() {
        // Simulates an `analyzer.cbor` written before `stop_words_family`
        // existed (but after `stop_words_list` did): no `stop_words_family`
        // key at all. Must still deserialize -- reload falls back to
        // Lucene only via `stop_words`, which this pre-family format could
        // only ever have set from the Lucene list (see
        // `PyTextAnalyzer::from_index` in src/py/mod.rs).
        #[derive(Serialize)]
        struct PreFamilyAnalyzerConfig {
            stemmer: String,
            language: String,
            stop_words: bool,
            stop_words_list: Vec<String>,
            english_possessive_filter: bool,
            tokenizer: Tokenizer,
        }
        let old = PreFamilyAnalyzerConfig {
            stemmer: "porter".to_string(),
            language: "english".to_string(),
            stop_words: true,
            stop_words_list: vec!["the".to_string(), "is".to_string()],
            english_possessive_filter: false,
            tokenizer: Tokenizer::Standard,
        };
        let mut bytes = Vec::new();
        ciborium::ser::into_writer(&old, &mut bytes).unwrap();
        let config: AnalyzerConfig = ciborium::de::from_reader(bytes.as_slice()).unwrap();

        assert_eq!(config.stop_words_family, None);
        assert_eq!(config.stop_words_list, vec!["the", "is"]);
        // Also predates `stop_words_filter_mode` -- must default to `Both`,
        // the pre-existing "check both" behavior.
        assert_eq!(config.stop_words_filter_mode, StopWordFilterMode::Both);
    }

    #[test]
    fn old_config_without_filter_mode_field_deserializes_as_both() {
        // Simulates an `analyzer.cbor` written after `stop_words_family`
        // existed but before `stop_words_filter_mode` did -- the exact
        // format on-disk indices built with the "lucene"/"terrier" presets
        // have today. It must deserialize with `stop_words_filter_mode ==
        // Both`, so those existing indices keep reproducing the results
        // they were actually built with instead of silently switching to
        // the new pre-stem-only / post-stem-only behavior.
        #[derive(Serialize)]
        struct PreFilterModeAnalyzerConfig {
            stemmer: String,
            language: String,
            stop_words: bool,
            stop_words_list: Vec<String>,
            stop_words_family: Option<String>,
            english_possessive_filter: bool,
            tokenizer: Tokenizer,
        }
        let old = PreFilterModeAnalyzerConfig {
            stemmer: "snowball".to_string(),
            language: "english".to_string(),
            stop_words: true,
            stop_words_list: vec!["what".to_string(), "is".to_string(), "the".to_string()],
            stop_words_family: Some("lucene".to_string()),
            english_possessive_filter: true,
            tokenizer: Tokenizer::LuceneEnglish,
        };
        let mut bytes = Vec::new();
        ciborium::ser::into_writer(&old, &mut bytes).unwrap();
        let config: AnalyzerConfig = ciborium::de::from_reader(bytes.as_slice()).unwrap();

        assert_eq!(config.stop_words_family.as_deref(), Some("lucene"));
        assert_eq!(config.stop_words_filter_mode, StopWordFilterMode::Both);
    }

    #[test]
    fn family_round_trips_through_serialization() {
        let mut config = AnalyzerConfig::default();
        config.stop_words = true;
        config.stop_words_list = vec!["a".to_string()];
        config.stop_words_family = Some("terrier".to_string());

        let mut bytes = Vec::new();
        ciborium::ser::into_writer(&config, &mut bytes).unwrap();
        let reloaded: AnalyzerConfig = ciborium::de::from_reader(bytes.as_slice()).unwrap();

        assert_eq!(reloaded.stop_words_family.as_deref(), Some("terrier"));
    }
}
