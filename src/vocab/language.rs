//! Language identifiers: ISO 639-1 codes are accepted wherever a language
//! name is (stemmers, stop word lists, default tokenizer).

/// Maps an ISO 639-1 code (e.g. `"fr"`) to the lowercase English language
/// name used throughout impact-index (e.g. `"french"`); any other value is
/// returned lowercased (so `"French"` and `"french"` both give `"french"`).
pub fn normalize_language(language: &str) -> String {
    let lang = language.to_lowercase();
    let name = match lang.as_str() {
        "ar" => "arabic",
        "bn" => "bengali",
        "da" => "danish",
        "de" => "german",
        "el" => "greek",
        "en" => "english",
        "es" => "spanish",
        "fa" => "persian",
        "fi" => "finnish",
        "fr" => "french",
        "hi" => "hindi",
        "hu" => "hungarian",
        "id" => "indonesian",
        "it" => "italian",
        "ja" => "japanese",
        "km" => "khmer",
        "ko" => "korean",
        "lo" => "lao",
        "my" => "burmese",
        "nb" | "nn" | "no" => "norwegian",
        "nl" => "dutch",
        "pt" => "portuguese",
        "ro" => "romanian",
        "ru" => "russian",
        "sv" => "swedish",
        "sw" => "swahili",
        "ta" => "tamil",
        "te" => "telugu",
        "th" => "thai",
        "tr" => "turkish",
        "yo" => "yoruba",
        "zh" => "chinese",
        _ => return lang,
    };
    name.to_string()
}

/// Whether text in this language is written without spaces between words
/// (Chinese, Japanese, Thai, Lao, Khmer, Burmese), and thus needs
/// dictionary-based word segmentation ([`super::analyzer::Tokenizer::Icu`]).
pub fn needs_word_segmentation(language: &str) -> bool {
    matches!(
        normalize_language(language).as_str(),
        "chinese" | "japanese" | "thai" | "lao" | "khmer" | "burmese"
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_normalize_language() {
        assert_eq!(normalize_language("fr"), "french");
        assert_eq!(normalize_language("French"), "french");
        assert_eq!(normalize_language("ZH"), "chinese");
        assert_eq!(normalize_language("klingon"), "klingon");
    }

    #[test]
    fn test_needs_word_segmentation() {
        assert!(needs_word_segmentation("zh"));
        assert!(needs_word_segmentation("thai"));
        assert!(!needs_word_segmentation("ko"));
        assert!(!needs_word_segmentation("english"));
    }
}
