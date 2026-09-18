# Impact Index for Information Retrieval

A Python/Rust library for efficient sparse retrieval. Built on Rust with PyO3 bindings for high performance.

Supports both neural IR models with floating-point impact scores and traditional BM25 bag-of-words retrieval with performance competitive with Lucene/Pyserini and Terrier.

## Features

- **BM25 bag-of-words indexing** with built-in tokenization, stemming (Snowball), and stop words: Lucene-family lists (17 languages) or Terrier's own, much longer list (English only) — see [Stop Words](#stop-words)
- **Block-Max MaxScore and BMW (Block-Max WAND)** search with early termination
- **SIMD bitpacking compression** (BitPacker4x) with quantized impacts and reusable block buffers
- **One-liner compression**: `index.compress("/path/to/output")`
- **Document reordering** by recursive graph bisection (`index.reorder(...)`) for smaller indices and stronger block-max pruning
- **Posting list splitting** by quantile for term impact decomposition
- **Index versioning**: per-index `manifest.json` with format version checks and one-step migration (`Index.update(path)`)
- **BMP (Block-Max Pruning)** for fast approximate search ([SIGIR 2024](https://github.com/pisa-engine/BMP))
- **Document store** with zstd compression and key-based retrieval
- **Async support** for non-blocking search and document retrieval
- **Parallel index compression** with rayon
- **Structured queries**: matchop-style `#combine`/`#syn`/`#band`/`#1` (phrase)/`#uwN` (window) operators, evaluated directly by WAND/MaxScore (`search_wand_query`/`search_maxscore_query`); the positional ones (`#1`, `#uwN`) need an index built with `positions=True`

## Performance

BM25 on MS MARCO passage (8.8M docs, 6,980 queries, top-100, single-threaded).
impact-index is built twice below, each time matching one reference system's
own tokenizer/stemmer/stopwords (see [BENCHMARKS.md](BENCHMARKS.md) for why,
and for a third build aligned with real Terrier 5 instead of PISA).
MaxScore is its headline algorithm.

**Lucene-aligned** (`pipeline="pyserini"`) — vs Pyserini:

| System | ARM q/s | x86 q/s | Index size | MRR@10 |
|--------|---------|---------|-----------|--------|
| **impact-index** (compressed + reordered, MaxScore) | **295** | **102 ± 0** | 0.65 GB | 0.1859 |
| Pyserini (Lucene) | 213 | 99 ± 1 | 0.59 GB | 0.1855 |

**PISA-aligned** (`pipeline="terrier-pisa"`) — vs PISA:

| System | x86 q/s | Index size | MRR@10 |
|--------|---------|-----------|--------|
| **impact-index** (compressed, MaxScore) | **235 ± 2** | 0.64 GB | 0.1866 |
| PISA (Block-Max WAND) | 215 ± 1 | 0.60 GB | 0.1854 |

- Result overlap: @10=0.985/@100=0.989 vs Pyserini, @10=0.976/@100=0.979 vs
  PISA.
- Compressed index is lossless (same results as raw) in both configurations.
- q/s is mean ± std over 5 search-only repeats, warm resident index. ARM numbers are from an earlier session (no ARM host this run).

See **[BENCHMARKS.md](BENCHMARKS.md)** for WAND/BMW numbers, full methodology, and a settings ablation (stemmer, tokenizer, stopwords, positions).

## Installation

```bash
pip install impact-index
```

Or build from source:

```bash
pip install maturin
maturin develop --release
```

## Quick Start: BM25 Search

```python
import impact_index

# Build a BM25 index with stemming and stop words
builder = impact_index.BOWIndexBuilder(
    "/path/to/index",
    stemmer="porter",  # matches Lucene/Pyserini
    stop_words=True,  # Lucene-compatible English stop words
)

# Index documents
builder.add_text(0, "the quick brown fox jumps over the lazy dog")
builder.add_text(1, "a quick brown cat jumps high")
builder.add_text(2, "the lazy dog sleeps all day")

# Build index (doc metadata and analyzer saved automatically)
index = builder.build(in_memory=True)

# BM25 scoring (doc lengths loaded automatically from index)
scored = index.with_scoring(impact_index.BM25Scoring(k1=0.9, b=0.4))

# Query analysis (analyzer loaded automatically from index)
query = index.analyzer().analyze_query("quick fox")
results = scored.search_maxscore(query, top_k=10)
for doc in results:
    print(f"Document {doc.docid}: {doc.score:.4f}")
```

## Structured Queries

`search_wand_query`/`search_maxscore_query` also accept Terrier-matchop-style
structured queries, on top of the flat `{term_id: weight}` form above:

- Each operator runs as a "virtual" posting list under the same WAND/MaxScore pruning as flat queries — no separate exhaustive path.
- `#1` (phrase) and `#uwN` (window) need positions: `BOWIndexBuilder(..., positions=True)`. Other operators and flat queries pay nothing for it.

| Syntax | Meaning | Needs positions? |
|--------|---------|-------------------|
| `#combine(...)` / `#combine:0=W0:1=W1(...)` | Weighted sum of children's scores | No |
| `#syn(t1 t2 ...)` | Synonym/OR: term frequencies summed, one virtual term | No |
| `#band(n1 n2 ...)` | Boolean AND: matches all children, score = sum | No |
| `#1(t1 t2 ...)` | Exact phrase: adjacent positions | Yes |
| `#uwN(t1 t2 ...)` | Unordered window of width `N` tokens | Yes |

A query is either a matchop string (needs an index built with
`BOWIndexBuilder`) or an equivalent nested Python structure with term ids:
`{"term": ix}`/`{"term": [ix, weight]}`, `{"combine": [[w, node], ...]}`,
`{"syn": [ix, ...]}`, `{"band": [node, ...]}`, `{"phrase": [ix, ...]}`,
`{"window": {"terms": [ix, ...], "width": N}}`.

```python
builder = impact_index.BOWIndexBuilder(
    "/path/to/index", stemmer="porter", stop_words=True, positions=True,
)
builder.add_text(0, "the quick brown fox jumps over the lazy dog")
index = builder.build(in_memory=True)
scored = index.with_scoring(impact_index.BM25Scoring())

results = scored.search_wand_query(
    "#combine(quick #1(brown fox) #band(lazy dog))", top_k=10
)
for doc in results:
    print(f"Document {doc.docid}: {doc.score:.4f}")
```

Scoring follows Terrier 5. Every operator is scored as one virtual term:
- `#syn`: tf = sum of the children's tfs, df = sum of their dfs.
- `#band`: tf = 1, df = sum of the children's dfs.
- `#1`/`#uwN`: tf = number of matches, df = N/100 (Terrier's fixed heuristic).

Nested `#combine` weights multiply. With `BM25Scoring(k3=8)` and
`pipeline="terrier", stemmer="porter"`, rankings are identical to
Terrier's. See the guide's "How structured queries are scored" section.

## Compression

Compress for smaller index size and block-max pruning:

```python
# Compress (standalone — includes vocab, docmeta, analyzer)
compressed = index.compress("/path/to/compressed")

# Search the compressed index (same API)
scored = compressed.with_scoring(impact_index.BM25Scoring())
results = scored.search_maxscore(query, top_k=10)
```

The default settings (`block_size=128`, `nbits=0`) are optimized:
- **block_size=128** aligns with SIMD registers and enables block-max pruning
- **nbits=0** lossless integer bitpacking for TF counts (~2-3 bits/value). Use `nbits=8` for neural IR with float impacts

## Document Reordering

Renumber documents by recursive graph bisection (BP) so similar documents
get nearby ids, for a smaller index and stronger block-max pruning:

```python
# From a raw index: reorder + compress in one step
reordered = index.reorder("/path/to/reordered")

# Fully transparent: search results carry the ORIGINAL document ids
scored = reordered.with_scoring(impact_index.BM25Scoring())
results = scored.search_maxscore(query, top_k=10)
for doc in results:
    print(f"Document {doc.docid}: {doc.score:.4f}")
```

The internal renumbering is invisible to callers; `reorder_map()` exposes
the raw permutation for advanced uses.

## Index Versioning & Migration

Every index directory carries a `manifest.json` with its format version.
Loading an index built by an older version raises an actionable error;
migrate with:

```python
impact_index.Index.update("/path/to/index")            # in place
impact_index.Index.update("/path/to/index", "/dest")   # or to a copy
```

Indices without a manifest (built before versioning existed) load
normally and are stamped on first load.

## Neural IR (Impact Scores)

```python
import numpy as np
import impact_index

# Build an index from pre-computed impact scores
builder = impact_index.IndexBuilder("/path/to/index")
builder.add(0, np.array([1, 5, 10], dtype=np.uintp),
            np.array([0.5, 1.2, 0.8], dtype=np.float32))
index = builder.build(in_memory=True)

# Search
results = index.search_maxscore({5: 1.0, 10: 0.5}, top_k=10)
```

## Stop Words

Two built-in stop word *families*, selectable independently of stemmer/language:

- **`"lucene"`** (default): short, per-language lists matching Lucene's language analyzers. 17 languages: arabic, danish, dutch, english, finnish, french, german, greek, hungarian, italian, norwegian, portuguese, romanian, russian, spanish, swedish, turkish.
- **`"terrier"`**: Terrier's own, much longer list (`org.terrier.terms.Stopwords`, 733 words for English) — what PISA and Terrier 5 use by default. **English only** — other languages raise an error rather than silently substituting something else.

```python
# Get stop words for any supported language/family
words = impact_index.get_stop_words("english")              # 33 words (Lucene, default)
words = impact_index.get_stop_words("french")                # 154 words (Lucene)
words = impact_index.get_stop_words("german")                 # 231 words (Lucene)
words = impact_index.get_stop_words("english", "terrier")     # 733 words (Terrier)
```

`BOWIndexBuilder`'s `stop_words` argument accepts the same families by name:

```python
builder = impact_index.BOWIndexBuilder(
    "/path/to/index", stemmer="snowball", language="english",
    stop_words="terrier",   # or "lucene", True (alias for "lucene"), a list, or None
)
```

- `stop_words=True` is a permanent alias for `stop_words="lucene"` — unaffected by the `"terrier"` addition.
- Whichever family (or custom list) was used is saved with the index and restored on reload. Indices built before the family selector existed reload as Lucene, matching what `stop_words=True` meant at the time.

## Documentation

Full documentation including guides on compression, BMP search, and the document store:

**https://experimaestro-ir-rust.readthedocs.io/en/latest/index.html**
