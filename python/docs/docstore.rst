.. _document-store:

Document store
==============

The document store provides compressed storage for document content and
metadata, using zstd block compression. Documents can be retrieved by
number or by key fields, and key values looked up from document numbers
(e.g. the ``docno`` of a search result). Key-only stores keep just these
mappings.

Building a store
----------------

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
---------------------------------

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
--------------------

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
---------------

.. code-block:: python

    docs = await store.aio_get_by_number([0, 1, 2])
    docs = await store.aio_get_by_key("docno", ["DOC001", "DOC002"])

Internal DocId vs external identifiers
--------------------------------------

The index only knows the integer ``DocId`` you pass to ``add`` /
``add_text``, not the corpus's own identifiers (e.g. a TREC ``docno``).
The document store maps between the two: give each document its index
``DocId`` with ``docid=``, and keep the external identifier as a key.

.. code-block:: python

    index_builder = impact_index.BOWIndexBuilder("/path/to/index", stemmer="porter")
    store_builder = impact_index.DocumentStoreBuilder("/path/to/store")

    for docid, doc in documents:                    # any increasing ids
        index_builder.add_text(docid, doc.text)
        store_builder.add({"docno": doc.external_id}, doc.text.encode(), docid=docid)

    index = index_builder.build(in_memory=True)
    store_builder.build()

Document numbers given with ``docid=`` must increase, and may have gaps
(numbers without document); without ``docid=``, documents get the next
number (0, 1, 2... when it is never given). Then, in both directions:

.. code-block:: python

    store = impact_index.DocumentStore.load("/path/to/store")
    results = index.with_scoring(impact_index.BM25Scoring()).search_maxscore(query, top_k=10)

    # DocId -> docno, without decompressing the documents
    docnos = store.get_keys("docno", [hit.docid for hit in results])
    # docno -> DocId
    store.get_numbers("docno", ["W1234"])          # [1234] (None if unknown)
    # DocId -> full document
    doc = store.get_by_number([results[0].docid])[0]

This survives :meth:`~impact_index.Index.reorder`: reordering renumbers
ids internally for storage locality, but search results are always
translated back to the *original* ``DocId``, which is what the store was
built against.

Key-only stores
---------------

To map identifiers without storing the documents, build a key-only store
with ``content=False``: the content passed to ``add`` (optional) is
ignored, no zstd-compressed blocks are written, and
:meth:`~impact_index.DocumentStore.get_by_number` returns documents with
their keys and an empty content.

.. code-block:: python

    builder = impact_index.DocumentStoreBuilder("/path/to/docnos", content=False)
    for docid, docno in docnos:
        builder.add({"docno": docno}, docid=docid)
    builder.build()

    store = impact_index.DocumentStore.load("/path/to/docnos")
    store.has_content()                            # False

Key-only stores can also be built and read in WebAssembly (see
:doc:`wasm`); with segments (:doc:`incremental`), use one store per
segment, numbered like the segment.

Format versions
---------------

Stores are written in format version 2, which adds the document number →
key tables (``key_<name>.idx`` / ``key_<name>.dat``), key-only stores and
document numbers with gaps. Stores written by earlier versions (format 1)
still load and work as before; on them,
:meth:`~impact_index.DocumentStore.get_keys` decompresses the documents
instead of reading a table, so it is slower — rebuild the store to get the
tables. Earlier versions of impact-index cannot read key-only stores nor
stores with gaps, and a store from a later format version fails to load
with an error asking to upgrade.

API reference
-------------

.. autoapiclass:: impact_index.DocumentStoreBuilder
   :members:
   :undoc-members:
   :show-inheritance:

.. autoapiclass:: impact_index.DocumentStore
   :members:
   :undoc-members:
   :show-inheritance:

.. autoapiclass:: impact_index.Document
   :members:
   :undoc-members:
   :show-inheritance:
