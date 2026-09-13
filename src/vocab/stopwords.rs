//! Embedded stop word lists, in two families:
//!
//! - **Lucene/Snowball** ([`StopWordFamily::Lucene`]): short, per-language
//!   lists matching Lucene's language analyzers (English = the 33-word
//!   `EnglishAnalyzer` default). Covers 17 languages.
//! - **Terrier** ([`StopWordFamily::Terrier`]): Terrier's own, much longer
//!   list (`org.terrier.terms.Stopwords`, 733 words), sourced from
//!   `share/stopword-list.txt` in <https://github.com/terrier-org/terrier-core>.
//!   Terrier ships exactly one stop word list in its own repo -- English --
//!   so this family is English-only; every other language simply has no
//!   Terrier list (see [`get_stop_words_for_family`]).
//!
//! Lists are compiled into the binary via `include_str!` -- no runtime I/O.
//! Use [`get_stop_words`] for the historical Lucene-only API, or
//! [`get_stop_words_for_family`] to pick a family explicitly.

use std::collections::HashSet;
use std::fmt;
use std::str::FromStr;

/// Which stop word family to use. See the module docs for provenance.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum StopWordFamily {
    /// Lucene/Snowball lists (17 languages).
    Lucene,
    /// Terrier's own list (English only).
    Terrier,
}

impl StopWordFamily {
    /// Lowercase name used in the Python API and in serialized configs.
    pub fn as_str(&self) -> &'static str {
        match self {
            StopWordFamily::Lucene => "lucene",
            StopWordFamily::Terrier => "terrier",
        }
    }
}

impl fmt::Display for StopWordFamily {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.as_str())
    }
}

impl FromStr for StopWordFamily {
    type Err = String;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s.to_lowercase().as_str() {
            "lucene" => Ok(StopWordFamily::Lucene),
            "terrier" => Ok(StopWordFamily::Terrier),
            other => Err(format!(
                "Unknown stop word family '{}', expected 'lucene' or 'terrier'",
                other
            )),
        }
    }
}

/// Returns the Lucene/Snowball stop word list for the given language, or
/// None if not available.
///
/// The `"english"` list is the Lucene EnglishAnalyzer default (33 words).
/// Other languages use the Snowball/Lucene stop word lists.
///
/// Supported languages: arabic, danish, dutch, english, finnish, french,
/// german, greek, hungarian, italian, norwegian, portuguese, romanian,
/// russian, spanish, swedish, turkish.
///
/// This is a thin, unchanged wrapper around
/// [`get_stop_words_for_family`]`(language, `[`StopWordFamily::Lucene`]`)`,
/// kept as its own function so existing callers (Rust and the Python
/// `impact_index.get_stop_words(language)` binding) keep returning exactly
/// what they always have. Use [`get_stop_words_for_family`] to request the
/// Terrier family instead.
pub fn get_stop_words(language: &str) -> Option<Vec<&'static str>> {
    get_stop_words_for_family(language, StopWordFamily::Lucene)
}

/// Returns the stop word list for the given language and family, or None if
/// that language has no list in that family.
///
/// Note: [`StopWordFamily::Terrier`] only has an English list -- Terrier
/// itself doesn't ship stop word lists for other languages -- so this
/// returns `None` for every other language under that family, even ones
/// [`StopWordFamily::Lucene`] supports.
pub fn get_stop_words_for_family(
    language: &str,
    family: StopWordFamily,
) -> Option<Vec<&'static str>> {
    let text = match (family, language.to_lowercase().as_str()) {
        // Lucene/Snowball lists
        // English: use the Lucene EnglishAnalyzer default (33 words)
        (StopWordFamily::Lucene, "english") => {
            include_str!("stopwords/lucene/english_analyzer.txt")
        }
        (StopWordFamily::Lucene, "arabic") => include_str!("stopwords/lucene/arabic.txt"),
        (StopWordFamily::Lucene, "danish") => include_str!("stopwords/lucene/danish.txt"),
        (StopWordFamily::Lucene, "dutch") => include_str!("stopwords/lucene/dutch.txt"),
        (StopWordFamily::Lucene, "finnish") => include_str!("stopwords/lucene/finnish.txt"),
        (StopWordFamily::Lucene, "french") => include_str!("stopwords/lucene/french.txt"),
        (StopWordFamily::Lucene, "german") => include_str!("stopwords/lucene/german.txt"),
        (StopWordFamily::Lucene, "greek") => include_str!("stopwords/lucene/greek.txt"),
        (StopWordFamily::Lucene, "hungarian") => include_str!("stopwords/lucene/hungarian.txt"),
        (StopWordFamily::Lucene, "italian") => include_str!("stopwords/lucene/italian.txt"),
        (StopWordFamily::Lucene, "norwegian") => include_str!("stopwords/lucene/norwegian.txt"),
        (StopWordFamily::Lucene, "portuguese") => include_str!("stopwords/lucene/portuguese.txt"),
        (StopWordFamily::Lucene, "romanian") => include_str!("stopwords/lucene/romanian.txt"),
        (StopWordFamily::Lucene, "russian") => include_str!("stopwords/lucene/russian.txt"),
        (StopWordFamily::Lucene, "spanish") => include_str!("stopwords/lucene/spanish.txt"),
        (StopWordFamily::Lucene, "swedish") => include_str!("stopwords/lucene/swedish.txt"),
        (StopWordFamily::Lucene, "turkish") => include_str!("stopwords/lucene/turkish.txt"),

        // Terrier list: English only (see module docs)
        (StopWordFamily::Terrier, "english") => include_str!("stopwords/terrier/english.txt"),

        _ => return None,
    };

    Some(
        text.lines()
            .map(|l| l.trim())
            .filter(|l| !l.is_empty() && !l.starts_with('#'))
            .collect(),
    )
}

/// Returns the stop words as a HashSet for efficient lookup.
pub fn get_stop_words_set(language: &str) -> Option<HashSet<String>> {
    get_stop_words(language).map(|words| words.into_iter().map(|w| w.to_string()).collect())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_english_stop_words() {
        let words = get_stop_words("english").unwrap();
        assert_eq!(words.len(), 33);
        assert!(words.contains(&"the"));
        assert!(words.contains(&"is"));
        assert!(!words.contains(&"hello"));
    }

    #[test]
    fn test_french_stop_words() {
        let words = get_stop_words("french").unwrap();
        assert!(words.len() > 100);
        assert!(words.contains(&"le"));
    }

    #[test]
    fn test_unknown_language() {
        assert!(get_stop_words("klingon").is_none());
    }

    #[test]
    fn test_lucene_family_matches_get_stop_words() {
        // `get_stop_words` must remain an exact, unchanged alias for the
        // Lucene family -- this is the "legacy=True implies lucene"
        // guarantee the Python binding and BOWIndexBuilder both rely on.
        let via_default = get_stop_words("english").unwrap();
        let via_family = get_stop_words_for_family("english", StopWordFamily::Lucene).unwrap();
        assert_eq!(via_default, via_family);
    }

    #[test]
    fn test_terrier_english_stop_words() {
        let words = get_stop_words_for_family("english", StopWordFamily::Terrier).unwrap();
        assert_eq!(words.len(), 733);
        assert!(words.contains(&"the"));
        assert!(words.contains(&"whatsoever")); // Terrier-only, not in the Lucene list
        assert!(!words.contains(&"hello"));

        let lucene_words = get_stop_words("english").unwrap();
        assert!(
            words.len() > lucene_words.len(),
            "Terrier's English list should be much larger than Lucene's"
        );
    }

    #[test]
    fn test_terrier_family_is_english_only() {
        // Terrier doesn't ship stop word lists for other languages, unlike
        // Lucene -- confirm we don't silently invent one.
        for lang in ["french", "german", "spanish", "arabic"] {
            assert!(
                get_stop_words_for_family(lang, StopWordFamily::Terrier).is_none(),
                "expected no Terrier list for '{}'",
                lang
            );
            assert!(
                get_stop_words_for_family(lang, StopWordFamily::Lucene).is_some(),
                "expected a Lucene list for '{}'",
                lang
            );
        }
    }

    #[test]
    fn test_stop_word_family_from_str() {
        assert_eq!(
            "Lucene".parse::<StopWordFamily>().unwrap(),
            StopWordFamily::Lucene
        );
        assert_eq!(
            "TERRIER".parse::<StopWordFamily>().unwrap(),
            StopWordFamily::Terrier
        );
        assert!("klingon".parse::<StopWordFamily>().is_err());
    }
}
