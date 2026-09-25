.. _document-store:

Document store
==============

The document store provides compressed storage for document content and
metadata, using zstd block compression. Documents can be retrieved by
sequential number or by key fields.

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
