.. _compression:

Compression and approximate search
==================================

An index built with :class:`~impact_index.IndexBuilder` or
:class:`~impact_index.BOWIndexBuilder` stores raw postings. This page
covers how to turn it into a smaller, faster index: compression, posting
list splitting and document reordering keep search **exact**, while
:ref:`BMP <bmp>` and :ref:`Seismic <seismic>` build separate indices for
fast **approximate** search over learned impacts.

Compression and transforms
--------------------------

Compressed indices use PFOR-delta for doc IDs and adaptive bitpacking
for values, with 128-posting blocks that enable block-max pruning.

Quick compression
~~~~~~~~~~~~~~~~~

The simplest way to compress an index:

.. code-block:: python

    import impact_index

    index = impact_index.Index.load("/path/to/raw_index", in_memory=True)

    # Compress with defaults (PFOR doc IDs, lossless integer TF, block_size=128)
    compressed = index.compress("/path/to/compressed")

    # The compressed index is standalone — includes vocab, docmeta, analyzer
    scored = compressed.with_scoring(impact_index.BM25Scoring())

    # For neural IR with float impacts, use quantization:
    compressed = index.compress("/path/to/compressed", nbits=8)

The compressed index is fully standalone: auxiliary files (vocabulary,
analyzer config, document metadata) are automatically copied from the
source index.

The default settings are optimized for BM25:

- **block_size=128**: aligns with SIMD registers and enables effective
  block-max pruning during search.
- **nbits=0** (default): lossless integer bitpacking for TF counts
  (~2-3 bits per value). Use ``nbits=8`` or ``nbits=16`` for quantized
  float compression (neural IR like SPLADE).

Advanced compression
~~~~~~~~~~~~~~~~~~~~

For full control over compressors, use
:class:`~impact_index.CompressionTransform`:

.. code-block:: python

    import impact_index

    index = impact_index.Index.load("/path/to/raw_index", in_memory=True)

    # Choose compressors
    docid_compressor = impact_index.BitPackingCompressor()   # SIMD (recommended)
    # docid_compressor = impact_index.EliasFanoCompressor()  # alternative

    # Fixed-range quantization (if you know the value range)
    impact_compressor = impact_index.ImpactQuantizer(nbits=8, min=0.0, max=10.0)

    # Or auto-ranging quantization (determines range from the index)
    impact_compressor = impact_index.GlobalImpactQuantizer(nbits=8)

    # Apply compression
    transform = impact_index.CompressionTransform(
        max_block_size=128,
        doc_ids_compressor=docid_compressor,
        impacts_compressor=impact_compressor,
    )
    transform.process("/path/to/compressed", index)

    # Load the compressed index
    compressed = impact_index.Index.load("/path/to/compressed", in_memory=True)

Splitting by quantiles
~~~~~~~~~~~~~~~~~~~~~~

:class:`~impact_index.SplitIndexTransform` partitions each term's postings
into sub-lists by value ranges, enabling more aggressive pruning with
MaxScore:

.. code-block:: python

    base_transform = impact_index.CompressionTransform(
        max_block_size=128,
        doc_ids_compressor=impact_index.EliasFanoCompressor(),
        impacts_compressor=impact_index.GlobalImpactQuantizer(nbits=8),
    )

    split_transform = impact_index.SplitIndexTransform(
        quantiles=[0.5, 0.9],   # split at 50th and 90th percentile
        sink=base_transform,
    )
    split_transform.process("/path/to/split_index", index)

Document reordering (graph bisection)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

:meth:`~impact_index.Index.reorder` renumbers documents by recursive graph
bisection (BP): documents sharing many terms receive nearby ids, which
shrinks posting-list gaps (smaller, faster-to-decode index) and makes
per-block maxima and minimum document lengths informative, sharpening
block-max pruning during search.

.. code-block:: python

    index = impact_index.Index.load("/path/to/raw_index", in_memory=True)

    # Reorder + compress in one step (same knobs as compress())
    reordered = index.reorder("/path/to/reordered", block_size=128, nbits=0)

    # Fully transparent: results carry the ORIGINAL document ids
    scored = reordered.with_scoring(impact_index.BM25Scoring())
    results = scored.search_maxscore(query, top_k=10)

The internal renumbering never surfaces: search results are translated
back to the original document ids automatically. ``reorder_map()``
exposes the raw permutation for advanced uses (e.g. interpreting raw
posting iterators). The permutation is deterministic (same input always
yields the same ordering). For composition with other transforms, use
:class:`~impact_index.ReorderTransform` with a ``sink`` transform, like
:class:`~impact_index.SplitIndexTransform` above.


.. _bmp:

BMP (Block-Max Pruning)
-----------------------

BMP implements "Faster Learned Sparse Retrieval with Block-Max Pruning"
(SIGIR 2024) for fast approximate search.

Converting to BMP format
~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    import impact_index

    index = impact_index.Index.load("/path/to/index", in_memory=True)

    # Streaming conversion (recommended — memory-efficient)
    index.to_bmp_streaming("/path/to/bmp_index.bin", bsize=64, compress_range=True)

    # Or legacy method (loads all postings into memory)
    # index.to_bmp("/path/to/bmp_index.bin", bsize=64, compress_range=True)

Searching with BMP
~~~~~~~~~~~~~~~~~~

Load a BMP index with :class:`~impact_index.BmpSearcher` and search:

.. code-block:: python

    searcher = impact_index.BmpSearcher("/path/to/bmp_index.bin")
    print(f"Documents: {searcher.num_documents()}")

    # Query uses string term IDs
    query = {"term1": 1.0, "term2": 0.5}
    doc_ids, scores = searcher.search(query, k=10, alpha=1.0, beta=1.0)
    for docid, score in zip(doc_ids, scores):
        print(f"{docid}: {score}")

BMP search parameters:

- ``k`` — number of results to return
- ``alpha`` — controls early termination aggressiveness (default: 1.0)
- ``beta`` — controls block skipping (default: 1.0)


.. _seismic:

Seismic (approximate search)
----------------------------

`Seismic <https://github.com/TusKANNy/seismic>`__ (Bruch et al., SIGIR 2024)
gives very fast **approximate** top-k retrieval over learned impacts. Like
BMP, a Seismic index is a separate artefact built from an existing impact
index. It scores by dot product over the stored impacts only, so it
supports neither query-time scoring models (BM25, LM) nor structured
queries.

Seismic is included in the Python package from PyPI. When building from
source, it is the ``seismic`` cargo feature (enabled by ``maturin
develop``), which needs the nightly toolchain pinned in
``rust-toolchain.toml``. You can check whether your build has it with
``hasattr(impact_index, "SeismicSearcher")``.

Converting to Seismic format
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

:meth:`~impact_index.Index.to_seismic` writes a Seismic index
directory. The defaults follow Seismic's recommendations for SPLADE on
MS MARCO:

.. code-block:: python

    import impact_index

    index = impact_index.Index.load("/path/to/index", in_memory=True)
    index.to_seismic("/path/to/seismic")

Build parameters:

- ``n_postings`` -- average number of postings kept per term (default:
  6000). Postings beyond this global budget are pruned *at build time*, so
  this sets a ceiling on recall at large depths.
- ``max_fraction`` -- maximum posting list length, as a multiple of
  ``n_postings`` (default: 1.5)
- ``centroid_fraction`` -- number of blocks (k-means centroids) per
  posting list, as a fraction of its length (default: 0.1)
- ``summary_energy`` -- fraction of the L1 mass kept in each block
  summary (default: 0.4)
- ``knn`` -- number of nearest neighbours stored per document; 0 (the
  default) builds no kNN graph

Searching with Seismic
~~~~~~~~~~~~~~~~~~~~~~

Open the directory with :class:`~impact_index.SeismicSearcher`. Queries use
the same ``{term_index: weight}`` dictionaries as exact search, and results
are the same scored documents:

.. code-block:: python

    searcher = impact_index.SeismicSearcher("/path/to/seismic")
    print(f"Documents: {searcher.num_documents()}")

    results = searcher.search({5: 1.0, 10: 0.5}, top_k=10,
                              query_cut=10, heap_factor=0.7)
    for r in results:
        print(r.docid, r.score)

Search parameters:

- ``top_k`` -- number of results to return
- ``query_cut`` -- only the ``query_cut`` highest-weighted query terms are
  traversed (default: 10)
- ``heap_factor`` -- blocks whose summary score is below ``heap_factor``
  times the current k-th score are skipped; lower is faster and less
  accurate, 1.0 disables this approximation (default: 0.7)
- ``n_knn`` -- number of kNN neighbours used to refine the results; needs
  an index built with ``knn > 0`` (default: 0)

Raising ``query_cut`` and ``heap_factor`` trades speed for accuracy. On
SPLADE-v3 / MS MARCO, Seismic is near-exact on the top-10 but drifts from
exact search deeper in the ranking (mostly because of ``n_postings``
pruning); see the `benchmarks
<https://github.com/xpmir/impact-index/blob/master/BENCHMARKS.md>`_.

A Seismic directory is tied to the Seismic version it was built with and
cannot be migrated: after an upgrade that changes the format, loading it
fails and you need to rebuild it with ``to_seismic``.


API reference
-------------

.. autoapiclass:: impact_index.Transform
   :members:
   :undoc-members:
   :show-inheritance:

.. autoapiclass:: impact_index.CompressionTransform
   :members:
   :undoc-members:
   :show-inheritance:

.. autoapiclass:: impact_index.SplitIndexTransform
   :members:
   :undoc-members:
   :show-inheritance:

.. autoapiclass:: impact_index.ReorderTransform
   :members:
   :undoc-members:
   :show-inheritance:

.. autoapiclass:: impact_index.DocIdCompressor
   :members:
   :undoc-members:
   :show-inheritance:

.. autoapiclass:: impact_index.BitPackingCompressor
   :members:
   :undoc-members:
   :show-inheritance:

.. autoapiclass:: impact_index.EliasFanoCompressor
   :members:
   :undoc-members:
   :show-inheritance:

.. autoapiclass:: impact_index.PForCompressor
   :members:
   :undoc-members:
   :show-inheritance:

.. autoapiclass:: impact_index.ImpactCompressor
   :members:
   :undoc-members:
   :show-inheritance:

.. autoapiclass:: impact_index.ImpactQuantizer
   :members:
   :undoc-members:
   :show-inheritance:

.. autoapiclass:: impact_index.GlobalImpactQuantizer
   :members:
   :undoc-members:
   :show-inheritance:

.. autoapiclass:: impact_index.BitPackedIntCompressor
   :members:
   :undoc-members:
   :show-inheritance:

.. autoapiclass:: impact_index.QuantizedBitPackedCompressor
   :members:
   :undoc-members:
   :show-inheritance:

.. autoapiclass:: impact_index.BmpSearcher
   :members:
   :undoc-members:
   :show-inheritance:

.. autoapiclass:: impact_index.SeismicSearcher
   :members:
   :undoc-members:
   :show-inheritance:
