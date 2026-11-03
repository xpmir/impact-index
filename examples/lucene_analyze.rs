//! Prints the terms produced by the Lucene language analyzers (see
//! `vocab::lucene`), for comparing with Lucene itself.
//!
//! Reads `language<TAB>text` lines from stdin and writes the space-separated
//! terms of each line (stop words removed). Japanese and Korean need the
//! dictionaries: `--ipadic <dir>` and `--ko-dic <dir>`.
//!
//! ```sh
//! cargo run --release --example lucene_analyze -- --ipadic DIR --ko-dic DIR < input.tsv
//! ```

use std::collections::{HashMap, HashSet};
use std::io::{BufRead, Write};
use std::path::PathBuf;

use impact_index::vocab::lucene::morph::{set_dictionary_path, MorphDictionary};
use impact_index::vocab::lucene::{LuceneChain, LuceneLanguage, LuceneStemmer};
use impact_index::vocab::stemmer::Stemmer;

struct Analyzer {
    chain: LuceneChain,
    stemmer: LuceneStemmer,
    stop_words: HashSet<String>,
}

fn main() {
    let mut args = std::env::args().skip(1);
    while let Some(arg) = args.next() {
        let dict = match arg.as_str() {
            "--ipadic" => MorphDictionary::Ipadic,
            "--ko-dic" => MorphDictionary::KoDic,
            other => panic!("unknown argument {}", other),
        };
        set_dictionary_path(dict, PathBuf::from(args.next().expect("missing path")));
    }

    let mut analyzers: HashMap<String, Analyzer> = HashMap::new();
    let stdout = std::io::stdout();
    let mut out = std::io::BufWriter::new(stdout.lock());
    for line in std::io::stdin().lock().lines() {
        let line = line.unwrap();
        let (lang, text) = line.split_once('\t').expect("expected language<TAB>text");
        let analyzer = analyzers.entry(lang.to_string()).or_insert_with(|| {
            let language = LuceneLanguage::for_language(lang).expect("English is not supported");
            Analyzer {
                chain: LuceneChain::load(language, None).unwrap(),
                stemmer: LuceneStemmer::new(language),
                stop_words: language
                    .default_stop_words(lang)
                    .into_iter()
                    .map(String::from)
                    .collect(),
            }
        });
        let terms: Vec<String> = analyzer
            .chain
            .tokens(text)
            .into_iter()
            .filter(|t| !t.is_empty() && !analyzer.stop_words.contains(t))
            .map(|t| analyzer.stemmer.stem(&t))
            .filter(|t| !t.is_empty())
            .collect();
        writeln!(out, "{}", terms.join(" ")).unwrap();
    }
}
