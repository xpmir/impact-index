.. _incremental:

Incremental indexing
====================

impact-index indices are built once and never modified. A collection that
changes continuously (documents added, modified, deleted) is kept instead as
a set of immutable **segments**. The library provides what that needs:

- **filtered search**, to skip deleted documents (tombstones) or restrict a
  search to a subset of the collection;
- **comparable quantisation**, so that scores from different segments can be
  merged;
- **multi-segment search**, returning one top-k over all segments.

The caller keeps the policy: when to flush a new segment, when to merge,
and where the segment list and tombstones are stored.

This works for learned impacts (e.g. SPLADE), whose scores do not depend on
collection statistics. BM25 segments would need global statistics
(document frequencies, average length) and are not supported yet.

Filtered search
---------------

:meth:`~impact_index.Index.search_maxscore_filtered` and
:meth:`~impact_index.Index.search_wand_filtered` (also on
:class:`~impact_index.ScoredIndex`) take two optional filters, each a numpy
boolean mask indexed by docid or a sequence of docids:

- ``deleted``: documents never returned;
- ``allowed``: if given, the only documents that may be returned.

.. code-block:: python

    import numpy as np

    deleted = np.zeros(index_size, dtype=bool)
    deleted[[3, 17, 42]] = True

    results = index.search_maxscore_filtered(query, top_k=10, deleted=deleted)
    results = index.search_wand_filtered(query, top_k=10, allowed=[1, 2, 5, 8])

Filters are checked once per candidate document, before it enters the
result heap. A rejected document never raises the pruning threshold, so the
result is exactly the top-k of the accepted documents (no over-fetching
needed). On a :ref:`reordered <compression>` index, filters use the original
docids, like search results. The unfiltered methods are unchanged and pay
nothing for this.

Comparable quantisation
-----------------------

``Index.compress(..., nbits=8)`` quantises impacts over the min/max of the
index being compressed. Two segments then decode the same weight to slightly
different values: at most one quantisation step, ``(max - min) / 2**nbits``,
per posting. That is negligible with ``nbits=16`` and visible with
``nbits=8``. Two options make scores exactly comparable:

.. code-block:: python

    # A fixed range, when the weight distribution is known
    seg = raw.compress("/path/seg", nbits=8, quantization=(0.0, 3.0))

    # Or the parameters of an existing index (typically the first segment)
    seg0 = raw0.compress("/path/seg0", nbits=8)
    seg1 = raw1.compress("/path/seg1", quantization=seg0.quantizer())

:meth:`~impact_index.Index.quantizer` returns the
:class:`~impact_index.QuantizerParams` of a compressed index (``None`` for a
lossless one). Passed as ``quantization``, they also set the number of bits.
The same option exists on
:class:`~impact_index.QuantizedBitPackedCompressor`, for use with
:class:`~impact_index.CompressionTransform`.

Values outside a fixed or reference range are clamped to it.
:attr:`~impact_index.Index.clamped_impacts` on the compressed index counts
them. When it grows, consider re-quantising: merge the segments with a new
reference.

Segmented search
----------------

:class:`~impact_index.SegmentedIndex` searches several indices as one. Each
segment is searched with its own top-k and filters, and the results are
merged into a global top-k of :class:`~impact_index.SegmentScoredDocument`
(``segment``, ``docid``, ``score``):

.. code-block:: python

    segments = impact_index.SegmentedIndex([seg0, seg1, seg2])
    hits = segments.search(
        query,
        top_k=10,
        deleted=[tombstones0, None, tombstones2],  # one entry per segment
        algorithm="maxscore",                      # or "wand"
    )
    for hit in hits:
        print(hit.segment, hit.docid, hit.score)

Segments are searched largest first, and each search starts from the global
k-th score found so far, so small segments prune harder
(``share_threshold=False`` disables this; results are the same up to ties at
the k-th score). Docids are those of each segment. Using the same global
docid in every segment, with the older versions tombstoned, keeps them
unambiguous.

The incremental pattern
-----------------------

Put together (a runnable version is in
`examples/incremental.py
<https://github.com/xpmir/impact-index/blob/master/examples/incremental.py>`__):

- Keep the raw sparse vectors as the source of truth, so any segment can be
  rebuilt without re-encoding documents.
- Put new and modified documents in a small buffer, searched exhaustively.
  When it fills, build it into a new immutable segment, quantised with the
  first segment's parameters.
- Tombstone deleted and modified documents in the segment holding them, and
  pass the tombstones as ``deleted`` at search time.
- Merge in the background (tiered, or when a segment is mostly deleted) by
  rebuilding from the stored vectors.
- List the live segments and their tombstones in a manifest, swapped
  atomically (write to a temporary file, then ``os.replace``), for crash
  safety.

API reference
-------------

.. autoapiclass:: impact_index.SegmentedIndex
   :members:
   :undoc-members:
   :show-inheritance:

.. autoapiclass:: impact_index.SegmentScoredDocument
   :members:
   :undoc-members:
   :show-inheritance:

.. autoapiclass:: impact_index.QuantizerParams
   :members:
   :undoc-members:
   :show-inheritance:
