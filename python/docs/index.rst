impact-index
=============

A Python/Rust library for efficient sparse retrieval. Built on Rust with
PyO3 bindings for high performance.

**impact-index** supports both neural IR models with pre-computed floating-point
impact scores and traditional BM25 bag-of-words retrieval with performance
competitive with Lucene/Pyserini.

Features
--------

- **BM25 bag-of-words indexing** with built-in tokenization, stemming
  (Snowball, Porter) and stop words: Lucene-family lists (17 languages) or
  Terrier's own, much longer list (English only)
- **Block-Max MaxScore and BMW WAND** search with early termination
- **Structured queries**: Terrier-style ``#combine``/``#syn``/``#band``/``#1``
  (phrase)/``#uwN`` (window) operators, evaluated by WAND/MaxScore
- **SIMD bitpacking compression** with quantized impacts and block-max pruning
- **One-liner compression**: ``index.compress("/path/to/output")``
- **Document reordering** by recursive graph bisection
  (``index.reorder(...)``) for smaller indices and stronger pruning
- **Posting list splitting** by quantile for term impact decomposition
- **Index versioning**: per-index ``manifest.json`` with format checks and
  one-step migration (``Index.update(path)``)
- **Approximate search** over learned impacts with
  :ref:`BMP (Block-Max Pruning) <bmp>` and :ref:`Seismic <seismic>`
- **Document store** with zstd compression and key-based retrieval
- **Async support** for non-blocking search and document retrieval

Performance
-----------

On MS MARCO passage, BM25 search is on par with or faster than Pyserini
(Lucene) and PISA at the same effectiveness. For SPLADE-v3, exact MaxScore
takes a few hundred ms per query, BMP around 15-30 ms and Seismic around
1 ms for the top-10, at nearly the same effectiveness. See the `README
<https://github.com/xpmir/impact-index#performance>`__ for summary tables
and `BENCHMARKS.md
<https://github.com/xpmir/impact-index/blob/master/BENCHMARKS.md>`__ for the
full methodology.

Installation
------------

From PyPI::

    pip install impact-index

From source (requires Rust toolchain)::

    pip install maturin
    maturin develop --release

Documentation
-------------

- :doc:`first-index`: build an index from learned sparse vectors
  (e.g. SPLADE) and search it with WAND or MaxScore
- :doc:`bow`: BM25 indexing from text or term frequencies, text analysis,
  positions and structured queries
- :doc:`compression`: compressed, split and reordered indices, and
  approximate search with BMP and Seismic
- :doc:`docstore`: compressed storage for the documents themselves
- :doc:`versioning`: on-disk format versions and migrating old indices

Each page ends with the API reference of the classes it covers.

.. toctree::
   :maxdepth: 2
   :caption: User guide

   first-index
   bow
   compression
   docstore
   versioning
