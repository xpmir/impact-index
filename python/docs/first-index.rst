.. _building-an-index:

Building your first index
=========================

This page shows how to build a sparse index from pre-computed impact
vectors (e.g. the output of SPLADE or another learned sparse model) and
how to search it. For term-frequency indices scored with BM25, see
:doc:`bow`.

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
---------------

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
-------------

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

Batch and multi-threaded search
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

All search methods (``search_wand``, ``search_maxscore``, their
``_filtered`` variants, on both :class:`~impact_index.Index` and
:class:`~impact_index.ScoredIndex`) release the GIL while searching, so
calls from several Python threads run in parallel.

To search many queries at once, the ``_many`` variants take a list of
queries and run them on a Rust thread pool (one Python call, no thread
pool on the Python side). They return one result list per query, in order:

.. code-block:: python

    queries = [{5: 1.0}, {10: 0.5, 42: 1.5}]
    batch = index.search_maxscore_many(queries, top_k=10)
    for query, results in zip(queries, batch):
        ...

Structured queries have the same batch form,
``search_wand_query_many``/``search_maxscore_query_many`` (see
:doc:`bow`).

The thread pool is rayon's global pool; set ``RAYON_NUM_THREADS`` to bound
its size.

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


API reference
-------------

.. autoapiclass:: impact_index.BuilderOptions
   :members:
   :undoc-members:
   :show-inheritance:

.. autoapiclass:: impact_index.IndexBuilder
   :members:
   :undoc-members:
   :show-inheritance:

.. autoapiclass:: impact_index.IndexView
   :members:
   :undoc-members:
   :show-inheritance:

.. autoapiclass:: impact_index.Index
   :members:
   :undoc-members:
   :show-inheritance:

.. autoapiclass:: impact_index.ScoredDocument
   :members:
   :undoc-members:
   :show-inheritance:

.. autoapiclass:: impact_index.SparseIndexIterator
   :members:
   :undoc-members:
   :show-inheritance:

.. autoapiclass:: impact_index.TermImpact
   :members:
   :undoc-members:
   :show-inheritance:
