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

## Performance

BM25 on MS MARCO passage (8.8M docs, 6,980 queries, top-100, single-threaded,
compressed index with MaxScore; measured 2026-08 on Apple M-series and
x86-64/AVX2):

| System | ARM q/s | x86 q/s | Index size | MRR@10 |
|--------|---------|---------|-----------|--------|
| **impact-index** (compressed) | **278** | **101** | 0.69 GB | 0.1858 |
| **impact-index** (compressed + reordered) | **295** | **106** | 0.66 GB | 0.1858 |
| Pyserini (Lucene) | 213 | 90 | 0.6 GB | 0.1855 |
| Terrier 5 (PyTerrier) | 68 | — | 1.3 GB | 0.1879 |

Result overlap with Pyserini: @10=0.985, @100=0.989. Compressed index
is lossless (same results as raw). Analysis pipeline matches Lucene's
EnglishAnalyzer: UAX#29 tokenizer, Porter stemmer, English possessive
filter, and stop words. Per-step measurements live in `optimizations.md`.

Reproduce with `examples/benchmark_bm25.py`. Terrier 5.11 runs through
PyTerrier (single-pass index, one query at a time via
`pt.terrier.Retriever`, which adds some Python overhead per query); its
analysis pipeline (Terrier's own stop word list and stemming) differs
from the Lucene-matched systems, hence the different MRR@10 and a
Pyserini overlap of only 0.70@10. Java: Terrier 5 needs Java 11+ and
Pyserini needs Java 21 — both measured with OpenJDK (Temurin) 21.

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
