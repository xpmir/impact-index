//! Japanese and Korean morphological analysis, emulating Lucene's
//! `JapaneseAnalyzer` (Kuromoji, search mode) and `KoreanAnalyzer` (Nori,
//! `DecompoundMode.DISCARD`) with Lindera and the same dictionaries
//! (mecab-ipadic / mecab-ko-dic), loaded at runtime from a directory holding a
//! Lindera dictionary (e.g. the unzipped `lindera-ipadic-<version>.zip` release
//! asset).

use std::collections::{HashMap, HashSet};
use std::path::{Path, PathBuf};
use std::sync::{Arc, LazyLock, Mutex};

use lindera::dictionary::load_fs_dictionary;
use lindera::mode::{Mode, Penalty};
use lindera::segmenter::Segmenter;

/// Morphological dictionaries
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum MorphDictionary {
    /// mecab-ipadic (Kuromoji's dictionary), for Japanese
    Ipadic,
    /// mecab-ko-dic (Nori's dictionary), for Korean
    KoDic,
}

impl MorphDictionary {
    pub fn from_name(name: &str) -> Option<Self> {
        match name.to_lowercase().as_str() {
            "ipadic" => Some(MorphDictionary::Ipadic),
            "ko-dic" | "kodic" | "ko_dic" => Some(MorphDictionary::KoDic),
            _ => None,
        }
    }

    pub fn name(&self) -> &'static str {
        match self {
            MorphDictionary::Ipadic => "ipadic",
            MorphDictionary::KoDic => "ko-dic",
        }
    }
}

/// Kuromoji's default `stoptags.txt` (Lucene 10.5.0)
const JA_STOP_TAGS: &[&str] = &[
    "接続詞",
    "助詞",
    "助詞-格助詞",
    "助詞-格助詞-一般",
    "助詞-格助詞-引用",
    "助詞-格助詞-連語",
    "助詞-接続助詞",
    "助詞-係助詞",
    "助詞-副助詞",
    "助詞-間投助詞",
    "助詞-並立助詞",
    "助詞-終助詞",
    "助詞-副助詞／並立助詞／終助詞",
    "助詞-連体化",
    "助詞-副詞化",
    "助詞-特殊",
    "助動詞",
    "記号",
    "記号-一般",
    "記号-読点",
    "記号-句点",
    "記号-空白",
    "記号-括弧開",
    "記号-括弧閉",
    "その他-間投",
    "フィラー",
    "非言語音",
];

/// `KoreanPartOfSpeechStopFilter.DEFAULT_STOP_TAGS` (Lucene 10.5.0)
const KO_STOP_TAGS: &[&str] = &[
    "EP", "EF", "EC", "ETN", "ETM", "IC", "JKS", "JKC", "JKG", "JKO", "JKB", "JKV", "JKQ", "JX",
    "JC", "MAG", "MAJ", "MM", "SP", "SSC", "SSO", "SC", "SE", "XPN", "XSA", "XSN", "XSV", "UNA",
    "NA", "VSV",
];

/// Kuromoji/Nori `isPunctuation`: Java general categories Z*, Cc, Cf, P*, S*
/// -- i.e. not a letter, mark or number (approximated without category
/// tables: alphanumeric characters and combining marks are kept)
fn is_punctuation(ch: char) -> bool {
    !(ch.is_alphanumeric()
        || matches!(ch as u32,
            0x0300..=0x036F | 0x1AB0..=0x1AFF | 0x1DC0..=0x1DFF | 0x20D0..=0x20FF
            | 0x3099..=0x309A | 0xFE20..=0xFE2F))
}

/// Lindera version of the downloaded dictionaries (their format follows
/// Lindera's)
pub const LINDERA_VERSION: &str = "6.2.0";

/// Cache directory of downloaded dictionaries: `$IMPACT_INDEX_CACHE`, or
/// `impact-index` in the OS cache directory (e.g. `~/.cache` on Linux,
/// `~/Library/Caches` on macOS)
pub fn cache_dir() -> PathBuf {
    match std::env::var_os("IMPACT_INDEX_CACHE") {
        Some(dir) => PathBuf::from(dir),
        None => dirs::cache_dir()
            .unwrap_or_else(std::env::temp_dir)
            .join("impact-index"),
    }
}

/// Returns the directory of a dictionary in the cache, downloading it (the
/// prebuilt Lindera release asset) if needed
pub fn download_dictionary(dictionary: MorphDictionary) -> std::io::Result<PathBuf> {
    use std::io::{Error, ErrorKind};
    let root = cache_dir().join(format!("lindera-{}", LINDERA_VERSION));
    let name = format!("lindera-{}", dictionary.name());
    let target = root.join(&name);
    if target.join("metadata.json").exists() {
        return Ok(target);
    }
    std::fs::create_dir_all(&root)?;
    let url = format!(
        "https://github.com/lindera/lindera/releases/download/v{v}/{name}-{v}.zip",
        v = LINDERA_VERSION
    );
    log::info!(
        "Downloading the {} dictionary from {}",
        dictionary.name(),
        url
    );
    let response = ureq::get(&url)
        .call()
        .map_err(|e| Error::new(ErrorKind::Other, format!("cannot download {}: {}", url, e)))?;
    // Download, then extract in a temporary directory, renamed when complete
    let tmp = tempfile_dir(&root)?;
    let zip_path = tmp.join("dictionary.zip");
    {
        let mut file = std::fs::File::create(&zip_path)?;
        std::io::copy(&mut response.into_reader(), &mut file)?;
    }
    let mut archive = zip::ZipArchive::new(std::fs::File::open(&zip_path)?)
        .map_err(|e| Error::new(ErrorKind::InvalidData, e))?;
    archive
        .extract(&tmp)
        .map_err(|e| Error::new(ErrorKind::InvalidData, e))?;
    let extracted = tmp.join(&name);
    if !extracted.join("metadata.json").exists() {
        return Err(Error::new(
            ErrorKind::InvalidData,
            format!("{} does not contain {}/metadata.json", url, name),
        ));
    }
    match std::fs::rename(&extracted, &target) {
        Ok(()) => {}
        // Downloaded concurrently by another process
        Err(_) if target.join("metadata.json").exists() => {}
        Err(e) => return Err(e),
    }
    let _ = std::fs::remove_dir_all(&tmp);
    Ok(target)
}

fn tempfile_dir(root: &Path) -> std::io::Result<PathBuf> {
    let nanos = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_nanos())
        .unwrap_or(0);
    let dir = root.join(format!(".download-{}-{}", std::process::id(), nanos));
    std::fs::create_dir_all(&dir)?;
    Ok(dir)
}

/// Dictionary paths registered with [`set_dictionary_path`]
static PATHS: LazyLock<Mutex<HashMap<MorphDictionary, PathBuf>>> =
    LazyLock::new(|| Mutex::new(HashMap::new()));

/// Sets where a dictionary is found (overrides the path recorded in an
/// index, e.g. when the index was built on another machine)
pub fn set_dictionary_path(dictionary: MorphDictionary, path: PathBuf) {
    PATHS.lock().unwrap().insert(dictionary, path);
}

/// Path registered with [`set_dictionary_path`]
pub fn dictionary_path(dictionary: MorphDictionary) -> Option<PathBuf> {
    PATHS.lock().unwrap().get(&dictionary).cloned()
}

/// Splits `s` where characters change from punctuation to non-punctuation
fn split_punctuation(s: &str) -> Vec<&str> {
    let mut pieces = Vec::new();
    let mut start = 0;
    let mut previous: Option<bool> = None;
    for (i, ch) in s.char_indices() {
        let p = is_punctuation(ch);
        if previous.is_some_and(|q| q != p) {
            pieces.push(&s[start..i]);
            start = i;
        }
        previous = Some(p);
    }
    if start < s.len() {
        pieces.push(&s[start..]);
    }
    pieces
}

pub struct Morphology {
    dictionary: MorphDictionary,
    path: String,
    segmenter: Segmenter,
    stop_tags: HashSet<&'static str>,
}

/// Loaded dictionaries, shared by all analyzers (keyed by path)
static LOADED: LazyLock<Mutex<Vec<(MorphDictionary, String, Arc<Morphology>)>>> =
    LazyLock::new(|| Mutex::new(Vec::new()));

impl Morphology {
    /// Loads (or reuses) the dictionary at `path`
    pub fn load(dictionary: MorphDictionary, path: &Path) -> std::io::Result<Arc<Morphology>> {
        let key = path.to_string_lossy().to_string();
        let mut loaded = LOADED.lock().unwrap();
        if let Some((_, _, m)) = loaded
            .iter()
            .find(|(d, p, _)| *d == dictionary && *p == key)
        {
            return Ok(m.clone());
        }
        let dict = load_fs_dictionary(path).map_err(|e| {
            std::io::Error::new(
                std::io::ErrorKind::InvalidData,
                format!(
                    "cannot load {} dictionary from {}: {}",
                    dictionary.name(),
                    key,
                    e
                ),
            )
        })?;
        let mode = match dictionary {
            // Kuromoji's default search mode (same penalties)
            MorphDictionary::Ipadic => Mode::Decompose(Penalty::default()),
            MorphDictionary::KoDic => Mode::Normal,
        };
        let stop_tags = match dictionary {
            MorphDictionary::Ipadic => JA_STOP_TAGS,
            MorphDictionary::KoDic => KO_STOP_TAGS,
        }
        .iter()
        .copied()
        .collect();
        let m = Arc::new(Morphology {
            dictionary,
            path: key.clone(),
            segmenter: Segmenter::new(mode, dict, None),
            stop_tags,
        });
        loaded.push((dictionary, key, m.clone()));
        Ok(m)
    }

    pub fn dictionary(&self) -> MorphDictionary {
        self.dictionary
    }

    pub fn path(&self) -> &str {
        &self.path
    }

    /// Tokens up to (and excluding) the stop word filter; POS-filtered
    /// tokens are returned as empty strings (keeping their position)
    pub fn tokens(&self, text: &str) -> Vec<String> {
        let mut tokens = match self.segmenter.segment(text.into()) {
            Ok(tokens) => tokens,
            Err(e) => {
                log::warn!("morphological analysis failed: {}", e);
                return Vec::new();
            }
        };
        let mut out = Vec::with_capacity(tokens.len());
        for token in tokens.iter_mut() {
            let surface = token.surface.to_string();
            let details: Vec<String> = token.details().iter().map(|s| s.to_string()).collect();
            // Kuromoji/Nori never group punctuation and other characters in
            // an unknown word: split at changes, and discard punctuation
            let pieces = split_punctuation(&surface);
            if pieces.len() > 1 {
                for piece in pieces {
                    if !piece.starts_with(is_punctuation) {
                        out.push(piece.to_string());
                    }
                }
                continue;
            }
            match surface.chars().next() {
                None => continue,
                Some(ch) if is_punctuation(ch) => continue, // discardPunctuation
                _ => {}
            }
            match self.dictionary {
                MorphDictionary::Ipadic => out.push(self.japanese(surface, &details)),
                MorphDictionary::KoDic => self.korean(surface, &details, &mut out),
            }
        }
        out
    }

    /// `JapaneseBaseFormFilter`, then `JapanesePartOfSpeechStopFilter`
    fn japanese(&self, surface: String, details: &[String]) -> String {
        let pos: Vec<&str> = details
            .iter()
            .take(4)
            .map(|s| s.as_str())
            .filter(|s| *s != "*")
            .collect();
        if self.stop_tags.contains(pos.join("-").as_str()) {
            return String::new();
        }
        match details.get(6) {
            Some(base) if base != "*" => base.clone(),
            _ => surface,
        }
    }

    /// Nori decompounding (DISCARD), `KoreanPartOfSpeechStopFilter`, then
    /// `KoreanReadingFormFilter`
    fn korean(&self, surface: String, details: &[String], out: &mut Vec<String>) {
        let get = |i: usize| details.get(i).map(|s| s.as_str()).unwrap_or("*");
        let pos_type = get(4);
        let expression = get(7);
        if pos_type != "*" && expression != "*" {
            // Inflect / Compound / Preanalysis: output the morphemes
            for morpheme in expression.split('+') {
                let mut parts = morpheme.split('/');
                let (Some(form), Some(tag)) = (parts.next(), parts.next()) else {
                    continue;
                };
                if form.is_empty() {
                    continue;
                }
                out.push(if self.stop_tags.contains(tag) {
                    String::new()
                } else {
                    form.to_string()
                });
            }
            return;
        }
        // Left POS: first tag
        let left_pos = match get(5) {
            "*" => get(0).split('+').next().unwrap_or("*"),
            first => first,
        };
        if self.stop_tags.contains(left_pos) {
            out.push(String::new());
            return;
        }
        // Reading form (hangul reading of hanja), for dictionary words
        match get(3) {
            "*" => out.push(surface),
            reading => out.push(reading.to_string()),
        }
    }
}
