//! Lucene 10.5.0 language analyzers, as selected by Anserini/Pyserini's
//! `-language` option (`AnalyzerMap`): `ArabicAnalyzer`, `BengaliAnalyzer`,
//! `GermanAnalyzer`, `SpanishAnalyzer`, `PersianAnalyzer`, `FinnishAnalyzer`,
//! `FrenchAnalyzer`, `HindiAnalyzer`, `IndonesianAnalyzer`,
//! `JapaneseAnalyzer`, `KoreanAnalyzer`, `RussianAnalyzer`, `TeluguAnalyzer`,
//! `ThaiAnalyzer`, `CJKAnalyzer` (Chinese), and `WhitespaceAnalyzer` for
//! languages Anserini has no analyzer for (e.g. Swahili, Yoruba).
//!
//! Each analyzer is split around its `StopFilter`: [`LuceneChain::tokens`]
//! runs the tokenizer and the filters before it, [`LuceneStemmer`] the
//! filters after it, so that [`super::analyzer::TextAnalyzer`] applies the
//! language's (Lucene) stop words at the same point as Lucene does.

mod indic_rules;
pub mod morph;
mod stemmers;
mod util;

use std::sync::Arc;

use rust_stemmers::{Algorithm, Stemmer as RustStemmer};
use unicode_segmentation::UnicodeSegmentation;

use self::morph::{MorphDictionary, Morphology};
use self::stemmers::*;
use self::util::{c, from_term, len16, lowercase, to_term};
use super::stemmer::Stemmer;

/// A Lucene language analyzer
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum LuceneLanguage {
    Arabic,
    Bengali,
    German,
    Spanish,
    Persian,
    Finnish,
    French,
    Hindi,
    Indonesian,
    Japanese,
    Korean,
    Russian,
    Telugu,
    Thai,
    /// `CJKAnalyzer`, used for Chinese
    Cjk,
    /// `WhitespaceAnalyzer`, for languages without a Lucene analyzer
    Whitespace,
}

impl LuceneLanguage {
    /// Analyzer used by Anserini for a language (name or ISO 639-1 code);
    /// `None` for English, which Anserini handles with its own analyzer
    /// (see the `pyserini` pipeline), and for the languages Anserini maps to
    /// a Lucene analyzer that is not ported here (see [`Self::resolve`])
    pub fn for_language(language: &str) -> Option<Self> {
        Self::resolve(language).ok().flatten()
    }

    /// Like [`Self::for_language`], but an error for unported analyzers
    pub fn resolve(language: &str) -> Result<Option<Self>, String> {
        use LuceneLanguage::*;
        let name = super::language::normalize_language(language);
        Ok(Some(match name.as_str() {
            "english" => return Ok(None),
            // Anserini's other analyzers (`AnalyzerMap`)
            "danish" | "hungarian" | "italian" | "dutch" | "norwegian" | "polish"
            | "portuguese" | "swedish" | "turkish" | "ukrainian" | "da" | "hu" | "it" | "nl"
            | "no" | "pl" | "pt" | "sv" | "tr" | "uk" => {
                return Err(format!(
                    "the Lucene analyzer used by Anserini for {} is not ported",
                    language
                ))
            }
            "arabic" => Arabic,
            "bengali" => Bengali,
            "german" => German,
            "spanish" => Spanish,
            "persian" => Persian,
            "finnish" => Finnish,
            "french" => French,
            "hindi" => Hindi,
            "indonesian" => Indonesian,
            "japanese" => Japanese,
            "korean" => Korean,
            "russian" => Russian,
            "telugu" => Telugu,
            "thai" => Thai,
            "chinese" => Cjk,
            _ => Whitespace,
        }))
    }

    /// Morphological dictionary needed by this analyzer
    pub fn dictionary(&self) -> Option<MorphDictionary> {
        match self {
            LuceneLanguage::Japanese => Some(MorphDictionary::Ipadic),
            LuceneLanguage::Korean => Some(MorphDictionary::KoDic),
            _ => None,
        }
    }

    /// The analyzer's default stop words (empty if it has none)
    pub fn default_stop_words(&self, language: &str) -> Vec<&'static str> {
        use super::stopwords::{get_stop_words_for_family, StopWordFamily};
        match self {
            LuceneLanguage::Korean | LuceneLanguage::Whitespace => Vec::new(),
            _ => get_stop_words_for_family(language, StopWordFamily::Lucene).unwrap_or_default(),
        }
    }
}

/// Lucene's `StandardTokenizer` token types that matter here
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum TokenType {
    Ideographic,
    Hiragana,
    Katakana,
    Hangul,
    Other,
}

fn is_han(ch: u32) -> bool {
    matches!(ch,
        0x2E80..=0x2E99 | 0x2E9B..=0x2EF3 | 0x2F00..=0x2FD5 | 0x3021..=0x3029 | 0x3038..=0x303B | 0x3400..=0x4DBF | 0x4E00..=0x9FFF
        | 0xF900..=0xFA6D | 0xFA70..=0xFAD9 | 0x20000..=0x2A6DF | 0x2A700..=0x2EBEF
        | 0x2F800..=0x2FA1F | 0x30000..=0x323AF)
}

fn is_hiragana(ch: u32) -> bool {
    matches!(ch, 0x3041..=0x3096 | 0x309D..=0x309F | 0x1B001..=0x1B11F)
}

fn is_katakana(ch: u32) -> bool {
    // Word_Break=Katakana
    matches!(ch,
        0x3031..=0x3035 | 0x309B..=0x309C | 0x30A0..=0x30FA | 0x30FC..=0x30FF
        | 0x31F0..=0x31FF | 0x32D0..=0x32FE | 0x3300..=0x3357 | 0xFF66..=0xFF9D
        | 0x1B000)
}

fn is_hangul(ch: u32) -> bool {
    matches!(ch,
        0x1100..=0x11FF | 0x3131..=0x318E | 0xA960..=0xA97C | 0xAC00..=0xD7A3
        | 0xD7B0..=0xD7FB | 0xFFA0..=0xFFDC)
}

fn is_mark(ch: char) -> bool {
    !ch.is_alphanumeric() && !ch.is_whitespace() && {
        let u = ch as u32;
        matches!(u, 0x0300..=0x036F | 0x0483..=0x0489 | 0x0591..=0x05BD | 0x0610..=0x061A
            | 0x064B..=0x065F | 0x0900..=0x0DFF | 0x0E31..=0x0E4E | 0x1AB0..=0x1AFF
            | 0x1DC0..=0x1DFF | 0x200C..=0x200D | 0x20D0..=0x20FF | 0x3099..=0x309A
            | 0xFE00..=0xFE0F | 0xFE20..=0xFE2F)
    }
}

/// Southeast Asian scripts (Line_Break=SA), kept as whole runs by Lucene
fn is_southeast_asian(ch: u32) -> bool {
    matches!(ch,
        0x0E00..=0x0EFF | 0x1000..=0x109F | 0x1780..=0x17FF | 0x1950..=0x19DF
        | 0x1A20..=0x1AAF | 0xA9E0..=0xA9FF | 0xAA60..=0xAADF)
}

fn is_arabic_script(ch: char) -> bool {
    matches!(ch as u32,
        0x0600..=0x06FF | 0x0750..=0x077F | 0x08A0..=0x08FF | 0xFB50..=0xFDFF | 0xFE70..=0xFEFF)
}

/// Whether a UAX#29 segment is a token: it contains a letter or a decimal
/// digit (other numbers like "²" are not tokens), or is katakana
fn is_word(word: &str) -> bool {
    word.chars()
        .any(|ch| ch.is_alphabetic() || ch.is_ascii_digit() || util::is_decimal_digit(ch))
        || (!word.is_empty() && word.chars().all(|ch| is_katakana(ch as u32)))
}

fn token_type(token: &str) -> TokenType {
    let mut chars = token.chars().filter(|ch| !is_mark(*ch));
    let Some(first) = chars.next() else {
        return TokenType::Other;
    };
    let first = first as u32;
    let all = |f: fn(u32) -> bool| {
        token
            .chars()
            .filter(|ch| !is_mark(*ch))
            .all(|ch| f(ch as u32))
    };
    if is_han(first) {
        TokenType::Ideographic
    } else if is_hiragana(first) {
        TokenType::Hiragana
    } else if all(is_katakana) {
        TokenType::Katakana
    } else if all(is_hangul) {
        TokenType::Hangul
    } else {
        TokenType::Other
    }
}

/// Lucene's `StandardAnalyzer.DEFAULT_MAX_TOKEN_LENGTH`
const MAX_TOKEN_LENGTH: usize = 255;

/// A token with its byte offsets and type
struct Tok {
    text: String,
    start: usize,
    end: usize,
    kind: TokenType,
}

/// `StandardTokenizer` (UAX#29 words). Too long tokens are skipped (`None`,
/// keeping a position gap), like Lucene
fn standard_tokens(text: &str) -> Vec<Option<Tok>> {
    let mut out: Vec<Option<Tok>> = Vec::new();
    // Lucene's tokenizer also breaks Arabic-script words at ZWNJ
    let words = text.split_word_bound_indices().flat_map(|(start, word)| {
        let mut pieces = Vec::new();
        let mut piece_start = 0;
        let mut previous: Option<char> = None;
        let mut chars = word.char_indices().peekable();
        while let Some((i, ch)) = chars.next() {
            if ch == '\u{200C}' {
                let next = chars.peek().map(|(_, c)| *c);
                if previous.is_some_and(is_arabic_script) && next.is_some_and(is_arabic_script) {
                    pieces.push((start + piece_start, &word[piece_start..i]));
                    piece_start = i + ch.len_utf8();
                }
            }
            previous = Some(ch);
        }
        pieces.push((start + piece_start, &word[piece_start..]));
        pieces
    });
    for (start, word) in words {
        if !is_word(word) {
            continue;
        }
        let end = start + word.len();
        // Runs of Southeast Asian characters form a single token
        if word
            .chars()
            .all(|ch| is_southeast_asian(ch as u32) || is_mark(ch))
        {
            if let Some(Some(prev)) = out.last_mut() {
                if prev.end == start
                    && prev
                        .text
                        .chars()
                        .all(|ch| is_southeast_asian(ch as u32) || is_mark(ch))
                {
                    prev.text.push_str(word);
                    prev.end = end;
                    continue;
                }
            }
        }
        out.push(Some(Tok {
            text: word.to_string(),
            start,
            end,
            kind: token_type(word),
        }));
    }
    for t in out.iter_mut() {
        if t.as_ref()
            .is_some_and(|t| len16(&t.text) > MAX_TOKEN_LENGTH)
        {
            *t = None;
        }
    }
    out
}

/// Java's `Character.isWhitespace`
fn is_java_whitespace(ch: char) -> bool {
    matches!(ch,
        '\t' | '\n' | '\u{B}' | '\u{C}' | '\r' | '\u{1C}'..='\u{1F}' | ' ' | '\u{1680}'
        | '\u{2000}'..='\u{2006}' | '\u{2008}'..='\u{200A}' | '\u{2028}' | '\u{2029}'
        | '\u{205F}' | '\u{3000}')
}

/// `WhitespaceTokenizer` (tokens longer than 255 chars are split)
fn whitespace_tokens(text: &str) -> Vec<String> {
    let mut out = Vec::new();
    for word in text.split(is_java_whitespace).filter(|w| !w.is_empty()) {
        let mut current = String::new();
        let mut n = 0;
        for ch in word.chars() {
            let l = ch.len_utf16();
            if n + l > MAX_TOKEN_LENGTH {
                out.push(std::mem::take(&mut current));
                n = 0;
            }
            current.push(ch);
            n += l;
        }
        out.push(current);
    }
    out
}

/// `ThaiTokenizer`: dictionary-based word segmentation (ICU4X's Thai
/// dictionary here, the JDK's in Lucene -- so segmentation may differ),
/// keeping words that start with a letter or digit
fn thai_tokens(text: &str) -> Vec<String> {
    use icu_segmenter::{options::WordBreakInvariantOptions, WordSegmenter};
    // The dictionary model is the closest to the JDK's (the LSTM one is
    // further off on MIRACL Thai)
    let segmenter = WordSegmenter::new_dictionary(WordBreakInvariantOptions::default());
    let mut words = Vec::new();
    let mut last = 0;
    for boundary in segmenter.segment_str(text) {
        if boundary > last {
            // The JDK's word iterator also breaks between letters/digits and
            // other characters (e.g. "__word__")
            let segment = &text[last..boundary];
            let mut run = String::new();
            for ch in segment.chars() {
                let alnum = ch.is_alphanumeric() || is_mark(ch);
                if !alnum && !run.is_empty() {
                    words.push(std::mem::take(&mut run));
                } else if alnum && (run.is_empty() && is_mark(ch)) {
                    continue;
                }
                if alnum {
                    run.push(ch);
                }
            }
            if !run.is_empty() {
                words.push(run);
            }
            last = boundary;
        }
    }
    words
}

/// Halfwidth katakana (0xFF65-0xFF9F) to fullwidth, from `CJKWidthFilter`
const KANA_NORM: [u16; 59] = [
    0x30fb, 0x30f2, 0x30a1, 0x30a3, 0x30a5, 0x30a7, 0x30a9, 0x30e3, 0x30e5, 0x30e7, 0x30c3, 0x30fc,
    0x30a2, 0x30a4, 0x30a6, 0x30a8, 0x30aa, 0x30ab, 0x30ad, 0x30af, 0x30b1, 0x30b3, 0x30b5, 0x30b7,
    0x30b9, 0x30bb, 0x30bd, 0x30bf, 0x30c1, 0x30c4, 0x30c6, 0x30c8, 0x30ca, 0x30cb, 0x30cc, 0x30cd,
    0x30ce, 0x30cf, 0x30d2, 0x30d5, 0x30d8, 0x30db, 0x30de, 0x30df, 0x30e0, 0x30e1, 0x30e2, 0x30e4,
    0x30e6, 0x30e8, 0x30e9, 0x30ea, 0x30eb, 0x30ec, 0x30ed, 0x30ef, 0x30f3, 0x3099, 0x309A,
];

const KANA_COMBINE_VOICED: [u8; 88] = [
    78, 0, 0, 0, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 0, 1,
    0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 8, 8, 8, 8, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1,
];

const KANA_COMBINE_HALF_VOICED: [u8; 88] = [
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 0, 0, 2, 0, 0, 2, 0, 0, 2, 0, 0, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
];

/// `CJKWidthFilter`
fn cjk_width(token: &str) -> String {
    let mut s = to_term(token);
    let mut i = 0;
    while i < s.len() {
        let ch = s[i];
        if (0xFF01..=0xFF5E).contains(&ch) {
            s[i] = ch - 0xFEE0;
        } else if (0xFF65..=0xFF9F).contains(&ch) {
            let combined = (ch == 0xFF9E || ch == 0xFF9F) && i > 0 && {
                let prev = s[i - 1];
                if (0x30A6..=0x30FD).contains(&prev) {
                    let table = if ch == 0xFF9F {
                        &KANA_COMBINE_HALF_VOICED
                    } else {
                        &KANA_COMBINE_VOICED
                    };
                    s[i - 1] = prev + table[(prev - 0x30A6) as usize] as u16;
                    s[i - 1] != prev
                } else {
                    false
                }
            };
            if combined {
                s.remove(i);
                continue;
            }
            s[i] = KANA_NORM[(ch - 0xFF65) as usize];
        }
        i += 1;
    }
    from_term(&s)
}

/// `CJKBigramFilter` (all four scripts, no unigrams): adjacent CJK tokens
/// are joined and split into overlapping bigrams; a lone CJK character is
/// kept as a unigram
fn cjk_bigrams(tokens: Vec<Option<Tok>>) -> Vec<String> {
    let mut out = Vec::new();
    let mut run: Vec<char> = Vec::new();
    let mut last_end = usize::MAX;
    let flush = |run: &mut Vec<char>, out: &mut Vec<String>| {
        match run.len() {
            0 => {}
            1 => out.push(run[0].to_string()),
            n => {
                for i in 0..n - 1 {
                    out.push([run[i], run[i + 1]].iter().collect());
                }
            }
        }
        run.clear();
    };
    for tok in tokens {
        let Some(tok) = tok else {
            flush(&mut run, &mut out);
            out.push(String::new());
            continue;
        };
        if tok.kind == TokenType::Other {
            flush(&mut run, &mut out);
            out.push(tok.text);
        } else {
            if tok.start != last_end {
                flush(&mut run, &mut out);
            }
            run.extend(tok.text.chars());
            last_end = tok.end;
        }
    }
    flush(&mut run, &mut out);
    out
}

/// A Lucene analyzer: tokenizer and filters up to the stop filter
#[derive(Clone)]
pub struct LuceneChain {
    language: LuceneLanguage,
    morphology: Option<Arc<Morphology>>,
}

impl LuceneChain {
    /// `morphology` must be given for Japanese and Korean
    pub fn new(language: LuceneLanguage, morphology: Option<Arc<Morphology>>) -> Self {
        assert!(
            language.dictionary().is_none() || morphology.is_some(),
            "a morphological dictionary is required for {:?}",
            language
        );
        Self {
            language,
            morphology,
        }
    }

    pub fn language(&self) -> LuceneLanguage {
        self.language
    }

    /// Lucene analyzer of an analyzer config ([`Tokenizer::Lucene`] only),
    /// loading its morphological dictionary if needed
    ///
    /// [`Tokenizer::Lucene`]: super::analyzer::Tokenizer::Lucene
    pub fn from_config(config: &super::analyzer::AnalyzerConfig) -> std::io::Result<Option<Self>> {
        if config.effective_tokenizer() != super::analyzer::Tokenizer::Lucene {
            return Ok(None);
        }
        let language = LuceneLanguage::resolve(&config.language)
            .ok()
            .flatten()
            .ok_or_else(|| {
                std::io::Error::new(
                    std::io::ErrorKind::InvalidData,
                    format!("no Lucene analyzer for language {}", config.language),
                )
            })?;
        Self::load(language, config.dictionary.as_deref()).map(Some)
    }

    /// Creates the analyzer, loading its morphological dictionary (if any)
    /// from the path registered with [`morph::set_dictionary_path`], or else
    /// from `dictionary` (if it exists), or else from the cache (downloading
    /// it if needed, see [`morph::download_dictionary`])
    pub fn load(language: LuceneLanguage, dictionary: Option<&str>) -> std::io::Result<Self> {
        let morphology = match language.dictionary() {
            None => None,
            Some(dict) => {
                let path = match morph::dictionary_path(dict).or_else(|| {
                    dictionary
                        .map(std::path::PathBuf::from)
                        .filter(|p| p.join("metadata.json").exists())
                }) {
                    Some(path) => path,
                    None => morph::download_dictionary(dict)?,
                };
                Some(Morphology::load(dict, &path)?)
            }
        };
        Ok(Self::new(language, morphology))
    }

    /// Path of the loaded morphological dictionary, if any
    pub fn dictionary_path(&self) -> Option<String> {
        self.morphology.as_ref().map(|m| m.path().to_string())
    }

    /// Tokens before the stop filter, in order; removed tokens are empty
    /// strings (so that positions keep gaps)
    pub fn tokens(&self, text: &str) -> Vec<String> {
        use LuceneLanguage::*;
        let standard = |f: &dyn Fn(&str) -> String| -> Vec<String> {
            standard_tokens(text)
                .into_iter()
                .map(|t| t.map(|t| f(&t.text)).unwrap_or_default())
                .collect()
        };
        let mut tokens = match self.language {
            Arabic => standard(&|t| util::decimal_digits(&lowercase(t))),
            Bengali => standard(&|t| {
                let mut s = to_term(&util::decimal_digits(&lowercase(t)));
                indic_normalize(&mut s);
                bengali_normalize(&mut s);
                from_term(&s)
            }),
            Hindi => standard(&|t| {
                let mut s = to_term(&util::decimal_digits(&lowercase(t)));
                indic_normalize(&mut s);
                hindi_normalize(&mut s);
                from_term(&s)
            }),
            Telugu => standard(&|t| {
                let mut s = to_term(&util::decimal_digits(t));
                indic_normalize(&mut s);
                telugu_normalize(&mut s);
                from_term(&s)
            }),
            Persian => standard(&|t| {
                let mut s = to_term(&util::decimal_digits(&lowercase(t)));
                arabic_normalize(&mut s);
                persian_normalize(&mut s);
                from_term(&s)
            }),
            German | Spanish | Finnish | Indonesian | Russian => standard(&lowercase),
            French => standard(&|t| lowercase(&french_elision(t))),
            Thai => thai_tokens(text)
                .iter()
                .map(|t| util::decimal_digits(&lowercase(t)))
                .collect(),
            Cjk => {
                let tokens = standard_tokens(text)
                    .into_iter()
                    .map(|t| {
                        t.map(|mut t| {
                            t.text = lowercase(&cjk_width(&t.text));
                            t
                        })
                    })
                    .collect();
                cjk_bigrams(tokens)
            }
            Japanese | Korean => self.morphology.as_ref().unwrap().tokens(text),
            Whitespace => whitespace_tokens(text),
        };
        // Lucene's filters drop tokens that become empty
        if self.language == Korean {
            for t in tokens.iter_mut() {
                *t = lowercase(t);
            }
        }
        tokens
    }
}

/// The filters after the stop filter (stemming, and normalization when
/// Lucene applies it after stop words), as a [`Stemmer`]
pub struct LuceneStemmer {
    language: LuceneLanguage,
    snowball: Option<RustStemmer>,
}

impl LuceneStemmer {
    pub fn new(language: LuceneLanguage) -> Self {
        let snowball = match language {
            LuceneLanguage::Finnish => Some(RustStemmer::create(Algorithm::Finnish)),
            LuceneLanguage::Russian => Some(RustStemmer::create(Algorithm::Russian)),
            _ => None,
        };
        Self { language, snowball }
    }
}

impl Stemmer for LuceneStemmer {
    fn stem(&self, word: &str) -> String {
        use LuceneLanguage::*;
        let apply = |fs: &[fn(&mut util::Term)]| {
            let mut s = to_term(word);
            for f in fs {
                f(&mut s);
            }
            from_term(&s)
        };
        match self.language {
            Arabic => apply(&[arabic_normalize, arabic_stem]),
            Bengali => apply(&[bengali_stem]),
            Hindi => apply(&[hindi_stem]),
            Telugu => apply(&[telugu_stem]),
            Persian => apply(&[persian_stem]),
            German => apply(&[german_normalize, german_light_stem]),
            Spanish => apply(&[spanish_light_stem]),
            French => apply(&[french_light_stem]),
            Indonesian => apply(&[indonesian_stem]),
            // Lucene's (newer) Snowball stemmers leave numbers unchanged
            Finnish | Russian if !word.chars().any(char::is_alphabetic) => word.to_string(),
            Finnish | Russian => self.snowball.as_ref().unwrap().stem(word).to_string(),
            Japanese => {
                // `JapaneseKatakanaStemFilter`, then `LowerCaseFilter`
                let mut s = to_term(word);
                if s.len() >= 4
                    && s.iter().all(|&ch| (0x30A0..=0x30FF).contains(&ch))
                    && *s.last().unwrap() == c('\u{30FC}')
                {
                    s.pop();
                }
                lowercase(&from_term(&s))
            }
            Thai | Cjk | Korean | Whitespace => word.to_string(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn analyze(lang: LuceneLanguage, text: &str) -> Vec<String> {
        let stemmer = LuceneStemmer::new(lang);
        LuceneChain::new(lang, None)
            .tokens(text)
            .iter()
            .filter(|t| !t.is_empty())
            .map(|t| stemmer.stem(t))
            .collect()
    }

    #[test]
    fn test_cjk_bigrams() {
        assert_eq!(
            analyze(LuceneLanguage::Cjk, "北京是中国的首都"),
            ["北京", "京是", "是中", "中国", "国的", "的首", "首都"]
        );
        // Lone character, mixed scripts, full width
        assert_eq!(
            analyze(LuceneLanguage::Cjk, "中 ＡＢＣ 国家"),
            ["中", "abc", "国家"]
        );
    }

    #[test]
    fn test_whitespace() {
        assert_eq!(
            analyze(LuceneLanguage::Whitespace, "Watoto walicheza, shuleni."),
            ["Watoto", "walicheza,", "shuleni."]
        );
    }

    #[test]
    fn test_hindi() {
        // Reference: Lucene 10.5.0 HindiAnalyzer
        assert_eq!(analyze(LuceneLanguage::Hindi, "स्कूल हूँ"), ["सकुल", "हुं"]);
    }

    #[test]
    fn test_french() {
        assert_eq!(analyze(LuceneLanguage::French, "L'école"), ["ecol"]);
    }
}
