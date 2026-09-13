User Guide
==========

.. _building-an-index:

Building an Index
-----------------

Use :class:`~impact_index.IndexBuilder` to create a sparse index from
document impact vectors. Each document is represented as a set of term
indices with associated impact values.

.. code-block:: python

    import numpy as np
    import impact_index

    builder = impact_index.IndexBuilder("/path/to/index")

    # Add documents: docid, term_indices, impact_values
    terms = np.array([0, 5, 42], dtype=np.uintp)
    values = np.array([1.2, 0.5, 3.1], dtype=np.float32)
    builder.add(0, terms, values)

    # More documents...
    builder.add(1, np.array([2, 5, 8], dtype=np.uintp),
                np.array([0.3, 0.9, 1.1], dtype=np.float32))

    # Finalize and get a searchable index
    index = builder.build(in_memory=True)

Builder options
~~~~~~~~~~~~~~~

Use :class:`~impact_index.BuilderOptions` to control checkpointing
(for crash recovery) and memory usage:

.. code-block:: python

    options = impact_index.BuilderOptions()
    options.checkpoint_frequency = 100000   # checkpoint every N documents
    options.in_memory_threshold = 1000000   # max postings per term before flush

    builder = impact_index.IndexBuilder("/path/to/index", options=options)

    # Resume from a checkpoint (returns None if no checkpoint exists)
    last_docid = builder.get_checkpoint_doc_id()
    if last_docid is not None:
        print(f"Resuming from document {last_docid}")

Storage dtype
~~~~~~~~~~~~~

By default, impact values are stored as ``float32``. You can choose a
different on-disk type to trade precision for space:

.. code-block:: python

    # Use float16 for smaller indices
    builder = impact_index.IndexBuilder("/path/to/index", dtype="float16")

Supported dtypes: ``"float32"`` (default), ``"float16"``, ``"bfloat16"``,
``"float64"``, ``"int32"``, ``"int64"``.


.. _searching:

Searching
---------

Load an existing index and search it with WAND or MaxScore. Both return
a list of :class:`~impact_index.ScoredDocument`:

.. code-block:: python

    import impact_index

    index = impact_index.Index.load("/path/to/index", in_memory=True)

    # Query: {term_index: query_weight}
    query = {5: 1.0, 10: 0.5, 42: 1.5}

    # WAND algorithm
    results = index.search_wand(query, top_k=10)
    for doc in results:
        print(f"Document {doc.docid}: {doc.score}")

    # MaxScore algorithm (often faster on compressed/split indices)
    results = index.search_maxscore(query, top_k=10)

Async search
~~~~~~~~~~~~

For non-blocking retrieval (e.g., in a web server):

.. code-block:: python

    results = await index.aio_search_wand(query, top_k=10)
    results = await index.aio_search_maxscore(query, top_k=10)

Iterating over postings
~~~~~~~~~~~~~~~~~~~~~~~

You can inspect individual posting lists. Each element is a
:class:`~impact_index.TermImpact`:

.. code-block:: python

    iterator = index.postings(term_id)
    print(f"Length: {iterator.length()}")
    print(f"Max impact: {iterator.max_value()}")
    print(f"Max doc ID: {iterator.max_doc_id()}")

    for posting in iterator:
        print(f"Doc {posting.docid}: {posting.value}")


.. _bm25:

BM25 and Bag-of-Words Indexing
------------------------------

For traditional IR with BM25 scoring, use
:class:`~impact_index.BOWIndexBuilder` instead of
:class:`~impact_index.IndexBuilder`. It automatically tracks document
lengths and optionally integrates text analysis (tokenization + stemming).

.. note::

    :class:`~impact_index.BOWIndexBuilder` is a layer on top of the same
    storage engine as :class:`~impact_index.IndexBuilder`: a BOW index
    *is* a sparse impact index (postings, WAND/MaxScore search,
    compression, BMP conversion all apply unchanged), where the "impact
    value" happens to be a raw or analyzer-computed term frequency. What
    ``BOWIndexBuilder`` adds on top is bookkeeping BM25 needs but a raw
    impact index doesn't: per-document length tracking and, optionally,
    the text analysis pipeline (tokenizer, stemmer, stop words,
    vocabulary) described below.

Pre-tokenized input
~~~~~~~~~~~~~~~~~~~

If you already have term indices and term-frequency values:

.. code-block:: python

    import numpy as np
    import impact_index

    builder = impact_index.BOWIndexBuilder("/path/to/index", dtype="int32")

    # Add documents: docid, term_indices, tf_values
    terms = np.array([0, 5, 42], dtype=np.uintp)
    tf = np.array([3, 1, 2], dtype=np.int32)
    builder.add(0, terms, tf)

    builder.add(1, np.array([2, 5, 8], dtype=np.uintp),
                np.array([1, 4, 1], dtype=np.int32))

    # Build returns searchable Index (doc metadata stored automatically)
    index = builder.build(in_memory=True)

    # Create a BM25-scored index (doc lengths loaded automatically)
    scored = index.with_scoring(impact_index.BM25Scoring(k1=1.2, b=0.75))

    # Search with MaxScore (fastest algorithm)
    query = {0: 1.0, 5: 1.0}
    results = scored.search_maxscore(query, top_k=10)
    for doc in results:
        print(f"Document {doc.docid}: {doc.score}")

Raw text input with stemming
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

For direct text indexing with automatic tokenization, stemming, and
vocabulary management:

.. code-block:: python

    import impact_index

    builder = impact_index.BOWIndexBuilder(
        "/path/to/index",
        dtype="int32",
        stemmer="porter",  # Lucene-compatible Porter stemmer
        stop_words=True,   # Lucene default English stop words
    )

    builder.add_text(0, "the quick brown fox jumps over the lazy dog")
    builder.add_text(1, "a quick brown cat jumps high")
    builder.add_text(2, "the lazy dog sleeps all day")

    # Build index (doc metadata and analyzer config saved automatically)
    index = builder.build(in_memory=True)

    # BM25 scoring (doc lengths loaded automatically from index)
    scored = index.with_scoring(impact_index.BM25Scoring())

    # Query analysis (analyzer loaded automatically from index)
    query = index.analyzer().analyze_query("quick fox")
    results = scored.search_maxscore(query, top_k=10)

Choosing a stemmer
~~~~~~~~~~~~~~~~~~

Two stemmers are available via ``stemmer=``:

- ``"porter"`` — a direct port of Lucene's ``PorterStemFilter``. Pick this
  to match Lucene/Anserini/Pyserini defaults as closely as possible
  (useful when comparing results against, or reproducing, Pyserini runs).
- ``"snowball"`` — the classic Porter2/Snowball algorithm. Pick this to
  match PISA/Terrier defaults instead — both use Snowball-family stemmers.
- ``None`` (default) — no stemming.

The two disagree on some common words (e.g. "community", "day", "use"
stem differently), so pick based on which system you're comparing against
or reproducing rather than assuming they're interchangeable. See the
top-level README's Performance section for a benchmark of both
configurations against their respective reference systems.

Tokenizer variant
~~~~~~~~~~~~~~~~~~

Besides stemming, the analyzer also picks a tokenizer variant at the Rust
level (``Tokenizer::Standard`` vs ``Tokenizer::LuceneEnglish``,
``src/vocab/analyzer.rs``). ``LuceneEnglish`` additionally strips a
trailing English possessive (``'s``) before lowercasing, matching
Lucene's ``EnglishAnalyzer`` tokenizer chain (``StandardTokenizer`` ->
``EnglishPossessiveFilter``); this is a tokenization concern, not a
stemming one, so it composes with any stemmer choice (or none).

.. note::

    This isn't a separate ``BOWIndexBuilder`` argument: the possessive
    filter is enabled automatically whenever ``language="english"`` (the
    default), for *any* stemmer choice, and left off for every other
    language. There is currently no way to analyze English text without
    possessive-stripping through ``BOWIndexBuilder`` — use the lower-level
    Rust ``TextAnalyzer`` API directly if you need that.

Stop words
~~~~~~~~~~

Stop words (common words like "the", "is", "a") can be filtered during
indexing and querying to reduce index size and improve search speed.
Two built-in *families* are available, selectable independently of the
stemmer/language settings: ``"lucene"`` (default; short, per-language
lists matching Lucene's language analyzers, 17 languages) and
``"terrier"`` (Terrier's own, much longer list — the default PISA and
Terrier 5 themselves use; English only).

.. code-block:: python

    import impact_index

    # Use default stop words for the language (matches Lucene/Pyserini)
    builder = impact_index.BOWIndexBuilder(
        "/path/to/index",
        stemmer="snowball",
        language="english",
        stop_words=True,
    )

    # Or select the Terrier family (matches PISA/Terrier 5 defaults)
    builder = impact_index.BOWIndexBuilder(
        "/path/to/index",
        stemmer="snowball",
        language="english",
        stop_words="terrier",
    )

    # Or provide an explicit list
    builder = impact_index.BOWIndexBuilder(
        "/path/to/index",
        stemmer="snowball",
        stop_words=["the", "a", "is", "in"],
    )

    # Get the stop word list for any supported language/family
    words = impact_index.get_stop_words("english")              # 33 words (Lucene default)
    words = impact_index.get_stop_words("french")                # 154 words (Lucene)
    words = impact_index.get_stop_words("german")                 # 231 words (Lucene)
    words = impact_index.get_stop_words("english", "terrier")     # 733 words (Terrier)

Supported Lucene-family languages: arabic, danish, dutch, english,
finnish, french, german, greek, hungarian, italian, norwegian,
portuguese, romanian, russian, spanish, swedish, turkish. The Terrier
family covers English only; requesting it for another language raises
an error rather than silently substituting a Lucene list.

``stop_words=True`` is a permanent alias for ``stop_words="lucene"``, and
the family (or custom list) an index was built with is saved and
restored automatically on reload.

.. note::

    For fair comparison with Pyserini/Lucene, always enable stop words.
    Without them, high-frequency terms like "the" create very long
    posting lists that slow down search significantly.

.. note::

    Stop-word filtering is applied at a different pipeline stage per
    family, matching each family's reference system exactly: the
    ``"lucene"`` family filters the raw token *before* stemming (as
    Lucene's ``EnglishAnalyzer`` does), while the ``"terrier"`` family
    stems first and then filters the *stemmed* token against the raw
    (never-stemmed) stop-word list (as PISA's analyzer does) — so an
    inflected form like "however" (stemming to "howev") is *not* caught
    even under the Terrier family, matching PISA's own behavior rather
    than being over-aggressively filtered. A custom ``stop_words=[...]``
    list checks both stages, since there's no single reference pipeline
    to match.

Loading a saved index
~~~~~~~~~~~~~~~~~~~~~

The index automatically detects and loads auxiliary components
(doc metadata, analyzer config, vocabulary) from the directory:

.. code-block:: python

    import impact_index

    index = impact_index.Index.load("/path/to/index", in_memory=True)

    # Doc metadata and analyzer are loaded automatically
    scored = index.with_scoring(impact_index.BM25Scoring())
    analyzer = index.analyzer()
    query = analyzer.analyze_query("quick fox")
    results = scored.search_maxscore(query, top_k=10)

Token positions
~~~~~~~~~~~~~~~

By default, a BOW index stores only term frequencies — enough for BM25,
but not enough to know whether two terms were adjacent. Building with
``positions=True`` additionally records each term's token positions
within each document, which the phrase (``#1``) and window (``#uwN``)
structured query operators below need to evaluate.

.. code-block:: python

    builder = impact_index.BOWIndexBuilder(
        "/path/to/index",
        stemmer="porter",
        stop_words=True,
        positions=True,
    )
    builder.add_text(0, "the quick brown fox jumps over the lazy dog")
    builder.add_text(1, "a quick brown cat jumps high")
    index = builder.build(in_memory=True)

.. note::

    From Python, positional indexing only goes through text: with
    ``positions=True``, ``add_text``/``add_texts`` work as usual, but the
    pre-tokenized ``add(docid, terms, values)`` method raises an error
    instead (its message points to a Rust-only ``add_with_positions``
    method that isn't exposed to Python). Positions are stored per block
    and decoded lazily, so a query without ``#1``/``#uwN`` pays no extra
    cost at search time — the cost is the extra on-disk positions data
    written at build time.

Structured queries
~~~~~~~~~~~~~~~~~~

Beyond flat ``{term_id: weight}`` dicts,
:meth:`~impact_index.Index.search_wand_query` /
:meth:`~impact_index.Index.search_maxscore_query` (and the
:class:`~impact_index.ScoredIndex` equivalents) accept Terrier-matchop-style
structured queries, evaluated as "virtual" posting lists on top of the
same WAND/MaxScore dynamic pruning used for flat queries:

- ``#combine(...)`` — weighted sum of children (the default combinator
  when a query has multiple terms); ``#combine:0=2:1=1(quick fox)``
  weights the first child 2x and the second 1x.
- ``#syn(t1 t2 ...)`` — synonyms/OR: term frequencies are summed and the
  merged postings scored as a single virtual term.
- ``#band(n1 n2 ...)`` — boolean AND: only documents containing every
  child match; score is the sum of the children's scores.
- ``#1(t1 t2 ...)`` — exact phrase, adjacent positions only. **Requires
  an index built with** ``positions=True``.
- ``#uwN(t1 t2 ...)`` — unordered window: all terms within ``N`` tokens
  of each other, any order. **Requires positions**, same as ``#1``.

.. code-block:: python

    scored = index.with_scoring(impact_index.BM25Scoring())

    results = scored.search_wand_query(
        "#combine(#1(new york) #syn(city town) #band(guide budget))",
        top_k=10,
    )

A matchop string is resolved against the index's own analyzer (the same
tokenizer/stemmer/stop words used at indexing time), so it requires an
index built via ``BOWIndexBuilder``. You can also build the query tree
directly from term ids, with no analyzer involved, as nested dicts:
``{"term": ix}`` (or ``{"term": [ix, weight]}``), ``{"combine": [[w1,
node1], ...]}``, ``{"syn": [ix, ...]}``, ``{"band": [node, ...]}``,
``{"phrase": [ix, ...]}``, or ``{"window": {"terms": [ix, ...], "width":
N}}``.


.. _compression:

Compression and Transforms
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

Index Versioning and Migration
------------------------------

Every index directory contains a ``manifest.json`` recording its format
version. When a library upgrade changes the on-disk format, loading an
older index raises an error telling you to migrate:

.. code-block:: python

    # Error: "index format v1, this version requires v2 — run
    #         Index.update(path) ... to migrate"
    impact_index.Index.update("/path/to/index")            # migrate in place
    impact_index.Index.update("/path/to/index", "/dest")   # or to a copy

    index = impact_index.Index.load("/path/to/index", in_memory=True)

Migrations are streaming and fast (metadata-only where possible — e.g.
adding per-block statistics does not rewrite the postings files). Indices
that predate versioning (no ``manifest.json``) load normally and are
stamped with a manifest on first load.


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


.. _document-store:

Document Store
--------------

The document store provides compressed storage for document content and
metadata, using zstd block compression. Documents can be retrieved by
sequential number or by key fields.

Building a store
~~~~~~~~~~~~~~~~

Use :class:`~impact_index.DocumentStoreBuilder` to create a store:

.. code-block:: python

    import impact_index

    builder = impact_index.DocumentStoreBuilder(
        "/path/to/store",
        block_size=4096,    # documents per compressed block
        zstd_level=3,       # compression level
    )

    # Add documents with key-value metadata and binary content
    builder.add({"docno": "DOC001", "url": "http://example.com"}, b"document text here")
    builder.add({"docno": "DOC002", "url": "http://example.com/2"}, b"another document")

    # Finalize (can only be called once)
    builder.build()

Resumable builds (crash recovery)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The ``checkpoint_frequency`` argument controls both crash recovery and
automatic checkpointing:

- ``0`` (default) — checkpointing disabled. Output files are truncated
  on open and any existing checkpoint file is removed.
- ``N`` (positive int) — recover from any existing checkpoint, then
  automatically write a new checkpoint every ``N`` added documents.
- ``None`` — recover from any existing checkpoint, but never
  auto-checkpoint. Call ``builder.checkpoint()`` manually whenever you
  want a durable savepoint (e.g. before exiting cleanly).

When recovery happens, any documents added between the last checkpoint
and the crash are discarded and the output files are rewound to a
consistent state. ``builder.num_documents()`` returns how many documents
were restored.

``builder.add(...)`` returns ``True`` whenever the call ended in an
automatic checkpoint (only possible with a positive
``checkpoint_frequency``), which is convenient for surfacing progress in
your ingest loop.

.. code-block:: python

    # Auto-checkpoint mode
    builder = impact_index.DocumentStoreBuilder(
        "/path/to/store",
        checkpoint_frequency=10_000,
    )
    for doc in documents:
        if builder.add(doc.keys, doc.content):
            print(f"checkpointed at {builder.num_documents()} docs")
    builder.build()  # clears the checkpoint on success

.. code-block:: python

    # Manual mode: recover if a checkpoint exists, never auto-write one
    builder = impact_index.DocumentStoreBuilder(
        "/path/to/store",
        checkpoint_frequency=None,
    )
    print(f"resuming from {builder.num_documents()} docs")
    for batch in batches:
        for doc in batch:
            builder.add(doc.keys, doc.content)
        builder.checkpoint()  # one checkpoint per batch
    builder.build()

Retrieving documents
~~~~~~~~~~~~~~~~~~~~

Load a store with :meth:`~impact_index.DocumentStore.load` and retrieve
:class:`~impact_index.Document` objects by number or key. Each document
has :attr:`~impact_index.Document.keys` (metadata dict) and
:attr:`~impact_index.Document.content` (bytes):

.. code-block:: python

    store = impact_index.DocumentStore.load(
        "/path/to/store",
        content_access="memory",  # or "mmap" or "disk"
    )

    print(f"Total documents: {store.num_documents()}")
    print(f"Key fields: {store.key_names()}")

    # By sequential number (0-based)
    docs = store.get_by_number([0, 1, 2])
    for doc in docs:
        print(doc.keys, doc.content)

    # By key field value
    docs = store.get_by_key("docno", ["DOC001", "DOC002"])
    for doc in docs:
        if doc is not None:
            print(doc.keys, doc.content)

The ``content_access`` parameter controls how content data is accessed:

- ``"memory"`` — loads all content into RAM (fastest, highest memory)
- ``"mmap"`` — memory-mapped I/O (OS manages caching)
- ``"disk"`` — reads from disk on demand (lowest memory)

Async retrieval
~~~~~~~~~~~~~~~

.. code-block:: python

    docs = await store.aio_get_by_number([0, 1, 2])
    docs = await store.aio_get_by_key("docno", ["DOC001", "DOC002"])

Internal DocId vs external identifiers
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The index and the document store each keep their own, independent
numbering, and impact-index does not maintain a mapping between them:

- The index only ever knows the internal ``DocId`` (``u64``) you pass to
  ``add``/``add_text`` — a plain sequential counter with no notion of a
  corpus's own document identifier (e.g. a TREC ``docno``). Postings and
  search results are all expressed in terms of this id.
- :class:`~impact_index.DocumentStoreBuilder` assigns its own sequential
  ``internal_id`` purely from call order: ``builder.add(keys, content)``
  takes no doc id argument at all — the first call gets internal id 0,
  the second 1, and so on. Its *only* id-based lookup is
  :meth:`~impact_index.DocumentStore.get_by_key`, which maps a key field
  you chose (e.g. ``"docno"``) to that internal sequential number via an
  FST. There is no lookup from an index ``DocId`` to a store key, or vice
  versa, anywhere in the library.

In practice, the way to tie the two together is to build both structures
in lockstep — feeding them the same documents in the same order, with
contiguous ids starting at 0 — and to keep the corpus's own identifier as
a key field in the store:

.. code-block:: python

    index_builder = impact_index.BOWIndexBuilder("/path/to/index", stemmer="porter")
    store_builder = impact_index.DocumentStoreBuilder("/path/to/store")

    for docid, doc in enumerate(documents):        # docid: 0, 1, 2, ...
        index_builder.add_text(docid, doc.text)
        store_builder.add({"docno": doc.external_id}, doc.text.encode())

    index = index_builder.build(in_memory=True)
    store_builder.build()

Because both were fed the same documents in the same order, the store's
sequential number *is* the index's ``DocId`` — so after search,
``store.get_by_number(docid)`` retrieves the exact document that was
scored:

.. code-block:: python

    store = impact_index.DocumentStore.load("/path/to/store")
    scored = index.with_scoring(impact_index.BM25Scoring())
    results = scored.search_maxscore(query, top_k=10)

    for hit in results:
        doc = store.get_by_number([hit.docid])[0]
        print(doc.keys["docno"], hit.score, doc.content)

    # Going the other way -- external id to content, no search involved:
    doc = store.get_by_key("docno", ["W1234"])[0]

This convention breaks silently if the two are ever built out of lockstep
(e.g. documents filtered/skipped on one side but not the other, or
non-contiguous ``DocId`` values) — nothing validates the correspondence,
so it is entirely the caller's responsibility. It does survive
:meth:`~impact_index.Index.reorder`: reordering renumbers ids internally
for storage locality, but search results are always translated back to
the *original* ``DocId`` automatically, which is what the store was built
against.
