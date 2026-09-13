# Impact Index for Information Retrieval

A Python/Rust library for efficient sparse retrieval. Built on Rust with PyO3 bindings for high performance.

Supports both neural IR models with floating-point impact scores and traditional BM25 bag-of-words retrieval with performance competitive with Lucene/Pyserini and Terrier.

## Features

- **BM25 bag-of-words indexing** with built-in tokenization, stemming (Snowball), and stop words (17 languages, matching Lucene)
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

BM25 on MS MARCO passage (8.8M docs, 6,980 queries, top-100, single-threaded;
ARM measured 2026-08 on Apple M-series, x86 measured 2026-09 on x86-64/AVX2,
all x86 numbers from one session so they're directly comparable):

| System | ARM q/s | x86 q/s | Index size | MRR@10 |
|--------|---------|---------|-----------|--------|
| **impact-index** (compressed, MaxScore) | **278** | **99** | 0.64 GB | 0.1858 |
| **impact-index** (compressed + reordered, MaxScore) | **295** | **106** | 0.66 GB | 0.1858 |
| impact-index (compressed, WAND/BMW) | — | 71 | 0.64 GB | 0.1858 |
| Pyserini (Lucene) | 213 | 109 | 0.6 GB | 0.1855 |
| Terrier 5 (PyTerrier) | 68 | 27 | 1.3 GB | 0.1876 |
| PISA (Block-Max WAND) | — | 172 | 0.60 GB | 0.1854 |
| PISA (MaxScore) | — | 149 | 0.60 GB | 0.1854 |

MaxScore is impact-index's headline algorithm — its own WAND/BMW is included
above for transparency but is markedly slower at this top-k (see below), so
comparisons elsewhere on this page use MaxScore. PISA is included with both
of its own algorithms the same way. impact-index's WAND/BMW throughput was
raised 50% on x86 (and 5% on ARM) by maintaining cursor order incrementally
(a single bubble-down swap after each cursor advance) instead of a full
`sort_by` every iteration, mirroring PISA's own `block_max_wand_query` —
see the WAND source for details.

**Every comparison above uses impact-index configured to match that row's
own analysis pipeline, not one fixed config compared against everyone.**
Reference systems disagree on tokenizer/stemmer/stopwords, so a single
impact-index build compared against all of them would be an apples-to-oranges
result for whichever ones it doesn't match. `examples/benchmark_bm25.py`
therefore builds impact-index twice and reports two aligned comparisons:

- **Lucene-aligned** (Porter stemmer, Lucene's ~33-word stopword list) —
  matches Pyserini's own defaults. Result overlap vs Pyserini: @10=0.978,
  @100=0.985 (identical for impact-index's MaxScore and WAND — both are
  exact top-k algorithms, not approximate, so they must and do agree).
- **Terrier-aligned** (Snowball/Porter2 stemmer, Terrier's own ~730-word
  stopword list) — matches PISA's and Terrier 5's defaults. Result overlap
  vs PISA (Block-Max WAND): @10=0.883, @100=0.908; Terrier 5 itself only
  reaches @10=0.878 against PISA (its stemmer is classic Porter, not
  Porter2 — a smaller residual mismatch than stopwords, which PISA and
  Terrier 5 do share).

Compressed index is lossless (same results as raw) in both configurations.

Getting the Terrier-aligned overlap in that range required fixing two real
bugs, both in impact-index's stop-word handling (surfaced by this
alignment work, since Terrier's much larger stopword list exercises
stemmer interaction far more than Lucene's small one does):
1. An inflected or misspelled variant of a stop word (e.g. "whats" instead
   of "what's"/"what is" — common in web text) doesn't exact-match the raw
   stopword list, survives that filter, and then stems right back down to
   the stop word itself (Snowball: "whats" → "what"), quietly re-admitting
   it into the vocabulary as a real, scored term. Stop words are now also
   checked *after* stemming, against a pre-stemmed copy of the stopword
   list.
2. Query-time analysis (`index.analyzer()`) always reconstructed stop words
   from the language's *built-in default* list, discarding whatever custom
   `stop_words=[...]` list the index was actually built with — so a custom
   list like Terrier's never actually governed query-time filtering. The
   index format now persists the real list it was built with.
Before these fixes, the Terrier-aligned overlap vs PISA was only 0.78@10 —
not a config mismatch, but impact-index silently scoring stray stopword
matches that PISA correctly ignored.

Reproduce with `examples/benchmark_bm25.py`. Terrier 5.11 runs through
PyTerrier (single-pass index, one query at a time via
`pt.terrier.Retriever`, which adds some Python overhead per query).
Terrier uses its default exhaustive DAAT matching (`daat.Full`) — stock
Terrier 5.11 ships no WAND/block-max dynamic pruning, unlike
impact-index (MaxScore/BMW) and Lucene (Block-Max WAND). Java: Terrier 5
needs Java 11+ and Pyserini needs Java 21 — both measured with OpenJDK
(Temurin) 21.

PISA runs through [`pyterrier-pisa`](https://github.com/terrierteam/pyterrier_pisa),
which needs no JVM (pure C++/pybind11 bindings) but ships wheels for Linux
x86_64 only, so there's no ARM number. Its 0.60 GB excludes the raw
forward/inverted-index files PISA keeps on disk alongside the final index
(same convention as impact-index's own raw-vs-compressed split).

impact-index's own WAND/BMW trailing its MaxScore this much is a known,
measured effect of top_k=100: WAND-family pruning relies on the top-k
threshold θ rising fast enough to let the block-max bound reject candidates
outright, but at top_k=100, θ stays low for a long time — instrumented
counters over the full query set show 92% of the core WAND loop's
iterations are single-document cursor catch-ups with no pruning benefit at
all (only ~18,500 of ~234,600 loop iterations per query actually score,
reject-on-tightening, or block-skip a candidate). This is a real property
of the algorithm at this top-k, not a broken implementation — though PISA's
own WAND still beating its MaxScore under the same top-k suggests there's
implementation headroom here beyond what the algorithmic effect alone
explains.

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

Beyond the flat `{term_id: weight}` queries above, `search_wand_query`/
`search_maxscore_query` accept Terrier-matchop-style structured queries.
Each operator (`#combine`, `#syn`, `#band`, `#1`, `#uwN`) is evaluated as
a "virtual" posting list on top of the same WAND/MaxScore dynamic pruning
used for flat queries — no separate exhaustive path. `#1` (phrase) and
`#uwN` (window) need positional information, which is opt-in:
`BOWIndexBuilder(..., positions=True)`. Positions cost extra disk and are
read lazily, so queries without positional operators (`#combine`, `#syn`,
`#band`, or flat queries) pay nothing for it.

| Syntax | Meaning | Needs positions? |
|--------|---------|-------------------|
| `#combine(...)` / `#combine:0=W0:1=W1(...)` | Weighted sum of children's scores | No |
| `#syn(t1 t2 ...)` | Synonym/OR: term frequencies summed, scored as one virtual term | No |
| `#band(n1 n2 ...)` | Boolean AND: matches docs containing every child, score = sum | No |
| `#1(t1 t2 ...)` | Exact phrase: adjacent positions | Yes |
| `#uwN(t1 t2 ...)` | Unordered window of width `N` tokens | Yes |

A query is either a matchop string (terms resolved via the index's own
analyzer/vocabulary — requires an index built with `BOWIndexBuilder`) or
a nested Python structure of the same shape, with term ids in place of
words: `{"term": ix}` (or `{"term": [ix, weight]}`), `{"combine": [[w,
node], ...]}`, `{"syn": [ix, ...]}`, `{"band": [node, ...]}`, `{"phrase":
[ix, ...]}`, `{"window": {"terms": [ix, ...], "width": N}}`.

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

Scoring: `#1`/`#uwN` are scored as a single virtual term by the query-time
model (sum-of-idfs for BM25, same convention as `#syn`); `#band` is a
match-all filter whose score is the sum of its children's scores; `#syn`
sums term frequencies across its children and scores the merge once (not
once per child).

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
get nearby ids — the index gets smaller and block-max pruning gets stronger:

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
Loading an index built by an older library version raises an actionable
error; migrate with:

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

Built-in Lucene/Snowball stop word lists for 17 languages:

```python
# Get stop words for any supported language
words = impact_index.get_stop_words("english")   # 33 words
words = impact_index.get_stop_words("french")     # 154 words
words = impact_index.get_stop_words("german")     # 231 words
```

Supported: arabic, danish, dutch, english, finnish, french, german, greek,
hungarian, italian, norwegian, portuguese, romanian, russian, spanish,
swedish, turkish.

## Documentation

Full documentation including guides on compression, BMP search, and the document store:

**https://experimaestro-ir-rust.readthedocs.io/en/latest/index.html**
