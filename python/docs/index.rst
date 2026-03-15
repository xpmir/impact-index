impact-index
=============

A Python/Rust library for efficient sparse retrieval. Built on Rust with
PyO3 bindings for high performance.

**impact-index** supports both neural IR models with pre-computed floating-point
impact scores and traditional BM25 bag-of-words retrieval with performance
competitive with Lucene/Pyserini.

Features
--------

- **BM25 bag-of-words indexing** with stemming (Snowball) and stop words (17 languages)
- **Block-Max MaxScore and BMW WAND** search with early termination
- **SIMD bitpacking compression** with quantized impacts and block-max pruning
- **One-liner compression**: ``index.compress("/path/to/output")``
- **Posting list splitting** by quantile for term impact decomposition
- **BMP (Block-Max Pruning)** for fast approximate search
- **Document store** with zstd compression and key-based retrieval
- **Async support** for non-blocking search and document retrieval

Installation
------------

From PyPI::

    pip install impact-index

From source (requires Rust toolchain)::

    pip install maturin
    maturin develop --release

.. toctree::
   :maxdepth: 2
   :caption: Contents

   guide
   api
