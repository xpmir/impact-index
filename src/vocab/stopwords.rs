//! Embedded stop word lists from Lucene/Snowball.
//!
//! Lists are compiled into the binary via `include_str!` — no runtime I/O.
//! Use [`get_stop_words`] to retrieve the list for a language.

use std::collections::HashSet;

/// Returns the stop word list for the given language, or None if not available.
///
/// The `"english"` list is the Lucene EnglishAnalyzer default (33 words).
/// Other languages use the Snowball/Lucene stop word lists.
///
/// Supported languages: arabic, danish, dutch, english, finnish, french,
/// german, greek, hungarian, italian, norwegian, portuguese, romanian,
/// russian, spanish, swedish, turkish.
pub fn get_stop_words(language: &str) -> Option<Vec<&'static str>> {
    let text = match language.to_lowercase().as_str() {
        // English: use the Lucene EnglishAnalyzer default (33 words)
        "english" => include_str!("stopwords/english_analyzer.txt"),
        // Snowball/Lucene lists for other languages
        "arabic" => include_str!("stopwords/arabic.txt"),
        "danish" => include_str!("stopwords/danish.txt"),
        "dutch" => include_str!("stopwords/dutch.txt"),
        "finnish" => include_str!("stopwords/finnish.txt"),
        "french" => include_str!("stopwords/french.txt"),
        "german" => include_str!("stopwords/german.txt"),
        "greek" => include_str!("stopwords/greek.txt"),
        "hungarian" => include_str!("stopwords/hungarian.txt"),
        "italian" => include_str!("stopwords/italian.txt"),
        "norwegian" => include_str!("stopwords/norwegian.txt"),
        "portuguese" => include_str!("stopwords/portuguese.txt"),
        "romanian" => include_str!("stopwords/romanian.txt"),
        "russian" => include_str!("stopwords/russian.txt"),
        "spanish" => include_str!("stopwords/spanish.txt"),
        "swedish" => include_str!("stopwords/swedish.txt"),
        "turkish" => include_str!("stopwords/turkish.txt"),
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
}
