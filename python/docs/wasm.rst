.. _wasm:

WebAssembly
===========

The ``impact-index-wasm`` crate (``wasm/`` in the repository) builds the
library for ``wasm32-unknown-unknown``, to index and search in a browser
page, a web worker or another JavaScript runtime (Node.js, privileged
Firefox ``ChromeWorker``\ s...). It has the same index format as the native
library: an index built natively loads in WASM, and an index built in WASM
loads natively (the files are byte-identical for the same documents and
options).

**Bytes in, bytes out.** The module does no file I/O: indexes are passed in
and returned as maps from file names to ``Uint8Array``\ s, which the
application stores where it likes (IndexedDB, the Origin Private File
System, a server, files...).

Building
--------

The crate is not published yet; build it from a checkout (with the nightly
toolchain pinned in ``rust-toolchain.toml``)::

    rustup target add wasm32-unknown-unknown
    wasm-pack build --release --target web wasm

``--target nodejs`` or ``--target bundler`` give packages for Node.js or for
bundlers (webpack, Vite...). Without ``wasm-pack``, build with ``cargo build
-p impact-index-wasm --target wasm32-unknown-unknown --release`` and run
``wasm-bindgen`` (the same version as the ``wasm-bindgen`` crate in
``Cargo.lock``) on ``target/wasm32-unknown-unknown/release/impact_index_wasm.wasm``.

Indexing and search
-------------------

.. code-block:: javascript

    import init, { Builder, Index } from "./pkg/impact_index_wasm.js";
    await init();

    // Build: document ids must be increasing; terms are vocabulary indices
    const builder = new Builder({ range: { min: 0, max: 3 } });
    builder.add(0, new Uint32Array([12, 40]), new Float32Array([0.8, 1.5]));
    builder.add(1, new Uint32Array([3, 12]), new Float32Array([2.1, 0.2]));
    const files = builder.finish(); // Map: file name -> Uint8Array

    // Load (from a Map or a plain object) and search
    const index = Index.load(files);
    const { docids, scores } = index.search(
        new Uint32Array([12, 40]), new Float32Array([1.2, 0.4]), 10);

``Builder`` options (all optional):

- ``blockSize`` (128): postings per block;
- ``docids`` (``"pfor"``): document id codec, ``"pfor"``, ``"ef"``
  (Elias-Fano) or ``"bitpacking"``;
- ``impacts`` (``"quantized"``): ``"quantized"``, or ``"float"`` to store
  impacts unquantised;
- ``nbits`` (8): bits per quantised impact;
- ``range``: quantisation range, ``{min, max}`` or the ``quantizer()`` of
  another index (see :ref:`incremental`); by default, the range of the
  indexed impacts.

``Index.load`` accepts any compressed or split index, and forward indexes,
with file names relative to the index directory (e.g. ``inner/docids.dat``
for a split index). Search uses MaxScore. Document ids are returned as a
``Uint32Array``, so they must be below 2\ :sup:`32`.

Incremental indexing
--------------------

The :ref:`incremental pattern <incremental>` carries over: ``DocSet`` holds
the deleted documents of a segment, and ``SegmentedIndex`` searches several
segments as one.

.. code-block:: javascript

    import { Builder, DocSet, Index, SegmentedIndex } from "./pkg/impact_index_wasm.js";

    const seg0 = Index.load(files0);
    // A new segment, quantised like the first one: exactly comparable scores
    const builder = new Builder({ range: seg0.quantizer() });
    // ... builder.add(...)
    const seg1 = Index.load(builder.finish());
    console.log(builder.clampedImpacts); // impacts outside seg0's range

    const segments = new SegmentedIndex();
    segments.push(seg0, DocSet.fromDocids(new Uint32Array([3, 17])));
    segments.push(seg1);
    const hits = segments.search(new Uint32Array([12]), new Float32Array([1]), 10);
    // hits.segments, hits.docids, hits.scores

    segments.setDeleted(1, DocSet.fromDocids(new Uint32Array([5])));

Errors (corrupt or incomplete index files, invalid documents...) are thrown
as JavaScript exceptions, and the module stays usable.

Limits
------

- Single-threaded: no ``rayon`` and no shared memory.
- The index files are copied into the WASM memory when loaded (no memory
  mapping), so memory grows with the index size: plan for the size of the
  index files plus some 30 MB. On 200k synthetic documents (45 MB index),
  a top-100 search takes 1-3 ms in Node.js.
- Not available in WASM: BM25 / text indexing, positions and structured
  queries, document stores, BMP and Seismic, and builder checkpoints.
- Licence: GPL-2.0-or-later, like the rest of the library; a WASM bundle
  that includes it must be distributed under the GPL.

Rust features
-------------

The WASM crate depends on ``impact-index`` without its default features.
They can also be disabled in other Rust projects:

``python``
    Python bindings
``mmap``
    memory-mapped index files (otherwise, files are read into memory)
``parallel``
    multi-threaded indexing and compression with ``rayon``
``progress``
    progress bars
``zstd``
    document store and builder checkpoints (zstd is a C library)
``bmp``
    conversion to BMP indexes and CIFF export
