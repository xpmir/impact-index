#!/usr/bin/env python3
# /// script
# requires-python = ">=3.9"
# dependencies = [
#     "ir-datasets",
#     "pyserini",
#     "cbor2",
#     "snowballstemmer",
# ]
# ///
"""
Diagnose score differences between impact-index and Pyserini BM25 on MS MARCO.

Picks single-word and multi-word queries, then compares:
- Analyzed query terms (stemmed forms + term IDs)
- Top-5 results and scores from both systems
- Score ratios for matching documents
- Document frequency (df) comparison for single-word queries

Usage:
    uv run --with . examples/diagnose_queries.py
"""

import re
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cbor2
import ir_datasets
import snowballstemmer

import impact_index


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

OUTPUT_DIR = Path("/tmp/bm25_bench")
IMPACT_DIR = OUTPUT_DIR / "impact_index"
LUCENE_DIR = OUTPUT_DIR / "pyserini" / "lucene"
DATASET_NAME = "msmarco-passage/dev/small"
K1, B = 0.9, 0.4
TOP_K = 5
N_SINGLE = 5
N_MULTI = 5


# ---------------------------------------------------------------------------
# Query analyser (mirrors benchmark_bm25.py)
# ---------------------------------------------------------------------------


class QueryAnalyzer:
    """Analyze queries using saved vocabulary and snowball stemmer."""

    def __init__(
        self,
        index_dir: Path,
        language: str = "english",
        stop_words: Optional[List[str]] = None,
    ):
        vocab_path = index_dir / "vocab.cbor"
        with open(vocab_path, "rb") as f:
            vocab_data = cbor2.load(f)

        self.term_to_id: Dict[str, int] = vocab_data["term_to_id"]
        self.id_to_term: Dict[int, str] = {v: k for k, v in self.term_to_id.items()}
        self.stemmer = snowballstemmer.stemmer(language)
        self.stop_words: set = set(stop_words) if stop_words else set()

    def analyze_query(self, text: str) -> Dict[int, float]:
        tf: Dict[int, float] = defaultdict(float)
        for token in re.findall(r"\w+", text.lower(), re.UNICODE):
            if token in self.stop_words:
                continue
            stemmed = self.stemmer.stemWord(token)
            term_id = self.term_to_id.get(stemmed)
            if term_id is not None:
                tf[term_id] += 1.0
        return dict(tf)

    def analyze_query_debug(self, text: str):
        """Return detailed per-token analysis."""
        tokens = re.findall(r"\w+", text.lower(), re.UNICODE)
        details = []
        for token in tokens:
            is_stop = token in self.stop_words
            stemmed = self.stemmer.stemWord(token)
            term_id = self.term_to_id.get(stemmed)
            details.append(
                {
                    "raw": token,
                    "is_stop": is_stop,
                    "stemmed": stemmed,
                    "term_id": term_id,
                }
            )
        return details


# ---------------------------------------------------------------------------
# Lucene term stats via Pyserini internals
# ---------------------------------------------------------------------------


def get_lucene_df(searcher, term: str) -> int:
    """Get document frequency for a term in the Lucene index."""
    # Use the Java IndexReader directly
    try:
        from java.lang import String as JString
        from org.apache.lucene.index import Term
        reader = searcher.object.reader
        lucene_term = Term("contents", JString(term))
        return reader.docFreq(lucene_term)
    except Exception:
        # Fallback: try via searcher.doc_freq if available
        pass

    # Alternative: use Pyserini's built-in method
    try:
        return searcher.object.reader.docFreq(
            searcher.object.reader.terms("contents")
        )
    except Exception:
        return -1


def get_lucene_total_docs(searcher) -> int:
    """Get total number of documents in the Lucene index."""
    try:
        return searcher.object.reader.numDocs()
    except Exception:
        return -1


def get_lucene_term_stats(searcher, term: str):
    """Get df and total term frequency from Lucene."""
    try:
        from java.lang import String as JString
        from org.apache.lucene.index import Term

        reader = searcher.object.reader
        lucene_term = Term("contents", JString(term))
        df = reader.docFreq(lucene_term)
        ttf = reader.totalTermFreq(lucene_term)
        return df, ttf
    except Exception as e:
        return -1, -1


# ---------------------------------------------------------------------------
# Select queries
# ---------------------------------------------------------------------------


def select_queries(
    dataset, n_single: int, n_multi: int, analyzer: QueryAnalyzer
) -> Tuple[List[Tuple[str, str]], List[Tuple[str, str]]]:
    """Pick single-word and multi-word queries that have known terms."""
    single_word = []
    multi_word = []

    for query in dataset.queries_iter():
        qid = query.query_id
        text = query.text

        analyzed = analyzer.analyze_query(text)
        if not analyzed:
            continue

        # Count non-stop-word tokens
        tokens = [
            t
            for t in re.findall(r"\w+", text.lower(), re.UNICODE)
            if t not in analyzer.stop_words
        ]

        if len(tokens) == 1 and len(single_word) < n_single:
            single_word.append((qid, text))
        elif len(tokens) >= 3 and len(multi_word) < n_multi:
            multi_word.append((qid, text))

        if len(single_word) >= n_single and len(multi_word) >= n_multi:
            break

    return single_word, multi_word


# ---------------------------------------------------------------------------
# Diagnose a single query
# ---------------------------------------------------------------------------


def diagnose_query(
    qid: str,
    text: str,
    analyzer: QueryAnalyzer,
    scored_index,
    raw_index: impact_index.Index,
    searcher,
    doc_meta: impact_index.DocMetadata,
    is_single_word: bool,
):
    print(f"\n{'='*70}")
    print(f"Query: '{text}'  (qid={qid})  [{'single' if is_single_word else 'multi'}-word]")
    print(f"{'='*70}")

    # --- (a) Analyzed query terms ---
    details = analyzer.analyze_query_debug(text)
    query = analyzer.analyze_query(text)

    print("\n  Token analysis:")
    for d in details:
        status = "STOP" if d["is_stop"] else ("OOV" if d["term_id"] is None else f"id={d['term_id']}")
        print(f"    '{d['raw']}' -> stem='{d['stemmed']}' [{status}]")

    # --- (b) Top-5 from both systems ---
    # impact-index
    ii_hits = scored_index.search_maxscore(query, TOP_K)
    ii_results = [(h.docid, h.score) for h in ii_hits]

    # Pyserini
    ps_hits = searcher.search(text, k=TOP_K)
    ps_results = [(hit.docid, hit.score) for hit in ps_hits]

    print(f"\n  Top-{TOP_K} results:")
    print(f"    {'Rank':<6} {'impact-index':>30}  {'Pyserini':>30}")
    print(f"    {'':6} {'docid':>15} {'score':>14}  {'docid':>15} {'score':>14}")
    for rank in range(TOP_K):
        ii_str = f"{ii_results[rank][0]:>15} {ii_results[rank][1]:>14.6f}" if rank < len(ii_results) else f"{'---':>15} {'---':>14}"
        ps_str = f"{ps_results[rank][0]:>15} {ps_results[rank][1]:>14.6f}" if rank < len(ps_results) else f"{'---':>15} {'---':>14}"
        print(f"    {rank+1:<6} {ii_str}  {ps_str}")

    # --- (c) Score ratios for matching docs ---
    ii_score_map = {str(docid): score for docid, score in ii_results}
    ps_score_map = {str(docid): score for docid, score in ps_results}

    common_docs = set(ii_score_map.keys()) & set(ps_score_map.keys())
    if common_docs:
        print(f"\n  Score ratios (impact-index / pyserini) for {len(common_docs)} common doc(s):")
        ratios = []
        for docid in sorted(common_docs):
            ratio = ii_score_map[docid] / ps_score_map[docid] if ps_score_map[docid] != 0 else float("inf")
            ratios.append(ratio)
            print(f"    doc {docid}: {ii_score_map[docid]:.6f} / {ps_score_map[docid]:.6f} = {ratio:.6f}")
        if ratios:
            avg_ratio = sum(ratios) / len(ratios)
            print(f"    Average ratio: {avg_ratio:.6f}")
    else:
        print("\n  No common documents in top-5!")

    # --- (d) DF comparison for single-word queries ---
    if is_single_word:
        print("\n  Document frequency comparison (single-word query):")
        num_docs_ii = doc_meta.num_docs()
        avg_dl_ii = doc_meta.avg_dl()
        num_docs_lucene = get_lucene_total_docs(searcher)

        print(f"    Collection: impact-index N={num_docs_ii}, avg_dl={avg_dl_ii:.2f}")
        print(f"    Collection: Pyserini     N={num_docs_lucene}")

        for term_id, tf in query.items():
            stem = analyzer.id_to_term.get(term_id, "???")
            # impact-index df: count postings for this term
            postings_iter = raw_index.postings(term_id)
            ii_df = postings_iter.length()

            # Lucene: Lucene applies its own analyzer (porter stemmer by default)
            # We query with the *original* token, not the snowball stem,
            # because Pyserini's searcher handles its own analysis.
            # But for DF comparison, we need the Lucene-analyzed term.
            # Lucene's default English analyzer uses PorterStemmer.
            lucene_df, lucene_ttf = get_lucene_term_stats(searcher, stem)

            print(f"    term '{stem}' (id={term_id}):  impact-index df={ii_df}  Lucene df={lucene_df} ttf={lucene_ttf}")

            # Also check with the Lucene porter stem
            try:
                # Try to get what Lucene's analyzer produces
                from org.apache.lucene.analysis.en import EnglishAnalyzer
                from java.io import StringReader
                from org.apache.lucene.analysis.tokenattributes import CharTermAttribute

                lucene_analyzer = EnglishAnalyzer()
                # Analyze the raw query token
                raw_token = [d["raw"] for d in details if d["term_id"] == term_id][0]
                ts = lucene_analyzer.tokenStream("contents", StringReader(raw_token))
                char_attr = ts.addAttribute(CharTermAttribute.class_)
                ts.reset()
                lucene_terms = []
                while ts.incrementToken():
                    lucene_terms.append(char_attr.toString())
                ts.end()
                ts.close()

                if lucene_terms:
                    lucene_stem = lucene_terms[0]
                    if lucene_stem != stem:
                        print(f"      NOTE: Lucene stems '{raw_token}' -> '{lucene_stem}' vs snowball -> '{stem}'")
                        lucene_df2, lucene_ttf2 = get_lucene_term_stats(searcher, lucene_stem)
                        print(f"      Lucene df for '{lucene_stem}': {lucene_df2} ttf={lucene_ttf2}")
                    else:
                        print(f"      Stemmers agree: '{lucene_stem}'")
            except Exception as e:
                print(f"      (Could not compare Lucene stemmer: {e})")


# ---------------------------------------------------------------------------
# BM25 formula comparison
# ---------------------------------------------------------------------------


def compare_bm25_formula(
    analyzer: QueryAnalyzer,
    raw_index: impact_index.Index,
    doc_meta: impact_index.DocMetadata,
    searcher,
):
    """Manually compute BM25 for a single term/doc to see where formulas diverge."""
    print(f"\n{'='*70}")
    print("BM25 FORMULA COMPARISON")
    print(f"{'='*70}")

    num_docs = doc_meta.num_docs()
    avg_dl = doc_meta.avg_dl()
    num_docs_lucene = get_lucene_total_docs(searcher)

    print(f"\n  Collection stats:")
    print(f"    impact-index: N={num_docs}, avg_dl={avg_dl:.4f}")
    print(f"    Lucene:       N={num_docs_lucene}")

    print(f"\n  BM25 params: k1={K1}, b={B}")

    print(f"\n  Note: Lucene BM25 uses:")
    print(f"    idf = ln(1 + (N - df + 0.5) / (df + 0.5))")
    print(f"    tf_norm = tf / (tf + k1 * (1 - b + b * dl/avgdl))")
    print(f"    score = idf * tf_norm")
    print(f"  Traditional BM25 uses:")
    print(f"    idf = ln((N - df + 0.5) / (df + 0.5))")
    print(f"    tf_norm = (k1 + 1) * tf / (tf + k1 * (1 - b + b * dl/avgdl))")
    print(f"    score = idf * tf_norm")

    import math

    # Pick an example: first term in vocabulary with reasonable df
    # Use a term from a single-word query
    print(f"\n  Example: computing BM25 for a sample term...")
    sample_term_id = None
    sample_stem = None
    for stem, tid in list(analyzer.term_to_id.items())[:1000]:
        postings_iter = raw_index.postings(tid)
        df = postings_iter.length()
        if 1000 < df < 50000:
            sample_term_id = tid
            sample_stem = stem
            break

    if sample_term_id is None:
        print("    Could not find suitable sample term.")
        return

    postings_iter = raw_index.postings(sample_term_id)
    ii_df = postings_iter.length()

    # Get Lucene df
    lucene_df, lucene_ttf = get_lucene_term_stats(searcher, sample_stem)

    print(f"    Term: '{sample_stem}' (id={sample_term_id})")
    print(f"    impact-index df={ii_df}, Lucene df={lucene_df}")

    # Lucene IDF
    if lucene_df > 0:
        lucene_idf = math.log(1 + (num_docs_lucene - lucene_df + 0.5) / (lucene_df + 0.5))
        print(f"    Lucene IDF (BM25Similarity): {lucene_idf:.6f}")

    # Traditional IDF
    if ii_df > 0:
        trad_idf = math.log((num_docs - ii_df + 0.5) / (ii_df + 0.5))
        trad_idf_plus1 = math.log(1 + (num_docs - ii_df + 0.5) / (ii_df + 0.5))
        print(f"    Traditional IDF ln((N-df+0.5)/(df+0.5)): {trad_idf:.6f}")
        print(f"    Lucene-style IDF ln(1+(N-df+0.5)/(df+0.5)): {trad_idf_plus1:.6f}")

    # For tf=1, dl=avg_dl:
    tf = 1.0
    dl = avg_dl
    tf_norm_lucene = tf / (tf + K1 * (1 - B + B * dl / avg_dl))
    tf_norm_trad = (K1 + 1) * tf / (tf + K1 * (1 - B + B * dl / avg_dl))

    print(f"\n    For tf=1, dl=avg_dl:")
    print(f"      Lucene tf_norm:      {tf_norm_lucene:.6f}")
    print(f"      Traditional tf_norm: {tf_norm_trad:.6f}")
    print(f"      Ratio (trad/lucene): {tf_norm_trad / tf_norm_lucene:.6f}  (should be k1+1={K1+1})")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    print("Loading dataset...")
    dataset = ir_datasets.load(DATASET_NAME)

    # Load impact-index
    print(f"Loading impact-index from {IMPACT_DIR}...")
    raw_index = impact_index.Index.load(str(IMPACT_DIR), in_memory=True)
    doc_meta = impact_index.DocMetadata.load(str(IMPACT_DIR))
    scoring = impact_index.BM25Scoring(k1=K1, b=B)
    scored_index = raw_index.with_scoring(scoring, doc_meta)

    analyzer = QueryAnalyzer(
        IMPACT_DIR, stop_words=impact_index.get_stop_words("english")
    )

    # Load Pyserini
    print(f"Loading Pyserini from {LUCENE_DIR}...")
    from pyserini.search.lucene import LuceneSearcher

    searcher = LuceneSearcher(str(LUCENE_DIR))
    searcher.set_bm25(K1, B)

    # Select queries
    print("Selecting queries...")
    single_queries, multi_queries = select_queries(dataset, N_SINGLE, N_MULTI, analyzer)

    print(f"\nSelected {len(single_queries)} single-word queries:")
    for qid, text in single_queries:
        print(f"  qid={qid}: '{text}'")
    print(f"Selected {len(multi_queries)} multi-word queries:")
    for qid, text in multi_queries:
        print(f"  qid={qid}: '{text}'")

    # Diagnose each query
    for qid, text in single_queries:
        diagnose_query(
            qid, text, analyzer, scored_index, raw_index, searcher, doc_meta,
            is_single_word=True,
        )

    for qid, text in multi_queries:
        diagnose_query(
            qid, text, analyzer, scored_index, raw_index, searcher, doc_meta,
            is_single_word=False,
        )

    # BM25 formula comparison
    compare_bm25_formula(analyzer, raw_index, doc_meta, searcher)

    # --- Summary of findings ---
    print(f"\n{'='*70}")
    print("SUMMARY OF POTENTIAL DIFFERENCE SOURCES")
    print(f"{'='*70}")
    print("""
  1. STEMMER: impact-index uses Snowball stemmer; Lucene uses a modified
     Porter stemmer. These can produce different stems for some words.

  2. IDF FORMULA: Lucene BM25Similarity uses ln(1 + (N-df+0.5)/(df+0.5))
     while traditional BM25 uses ln((N-df+0.5)/(df+0.5)). Check which
     variant impact-index implements.

  3. TF NORMALIZATION: Lucene BM25 uses tf/(tf+k1*...) without the (k1+1)
     numerator factor. Traditional BM25 has (k1+1)*tf/(tf+k1*...).
     The (k1+1) factor is a constant multiplier that does not change ranking
     within a single term, but affects multi-term relative weighting if IDF
     also differs.

  4. STOP WORDS: Both use English stop words but the exact lists may differ.

  5. TOKENIZATION: impact-index uses \\w+ regex; Lucene uses StandardTokenizer
     which handles hyphens, apostrophes, etc. differently.

  6. DOCUMENT LENGTH: impact-index counts tokens after stemming (unique stems
     per doc or total?); Lucene counts tokens before stemming. This affects
     the dl/avgdl ratio in BM25.
""")


if __name__ == "__main__":
    main()
