# Impact Index for Information Retrieval

A Python/Rust library for efficient sparse retrieval. Built on Rust with PyO3 bindings for high performance.

Supports both neural IR models with floating-point impact scores and traditional BM25 bag-of-words retrieval with performance competitive with Lucene/Pyserini.

## Features

- **BM25 bag-of-words indexing** with built-in tokenization, stemming (Snowball), and stop words (17 languages, matching Lucene)
- **Block-Max MaxScore and BMW (Block-Max WAND)** search with early termination
- **SIMD bitpacking compression** (BitPacker4x) with quantized impacts and reusable block buffers
- **One-liner compression**: `index.compress("/path/to/output")`
- **Posting list splitting** by quantile for term impact decomposition
- **BMP (Block-Max Pruning)** for fast approximate search ([SIGIR 2024](https://github.com/pisa-engine/BMP))
- **Document store** with zstd compression and key-based retrieval
- **Async support** for non-blocking search and document retrieval
- **Parallel index compression** with rayon

## Performance

BM25 on MS MARCO passage (8.8M docs, 6,980 queries, top-100, single-threaded):

| System | q/s | Index size | MRR@10 |
|--------|-----|-----------|--------|
| **impact-index MaxScore** | **193** | 3.2 GB | 0.1858 |
| **impact-index Compressed** | **182** | 0.6 GB | 0.1858 |
| Pyserini (Lucene) | 221 | 0.6 GB | 0.1855 |

Result overlap with Pyserini: @10=0.985, @100=0.989. Compressed index
is lossless (same results as raw). Analysis pipeline matches Lucene's
EnglishAnalyzer: UAX#29 tokenizer, Porter stemmer, English possessive
filter, and stop words.

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

# Build and score with BM25
query = builder.analyze_query("quick fox")
index, doc_meta = builder.build(in_memory=True)
scored = index.with_scoring(impact_index.BM25Scoring(k1=0.9, b=0.4), doc_meta)

# Search with MaxScore (fastest)
results = scored.search_maxscore(query, top_k=10)
for doc in results:
    print(f"Document {doc.docid}: {doc.score:.4f}")
```

## Compression

Compress for smaller index size and block-max pruning:

```python
# One-liner: SIMD bitpacking + 8-bit quantization, block_size=128
compressed = index.compress("/path/to/compressed")

# Search the compressed index (same API)
scored = compressed.with_scoring(impact_index.BM25Scoring(), doc_meta)
results = scored.search_maxscore(query, top_k=10)
```

The default settings (`block_size=128`, `nbits=8`) are optimized:
- **block_size=128** aligns with SIMD registers (BitPacker4x) and enables effective block-max pruning
- **nbits=8** uses a fast path with direct byte reads (no bit-level overhead). `nbits=16` also has a fast path via direct `u16` reads

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
