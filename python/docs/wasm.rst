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

Installation
------------

The package is published on npm as ``impact-index`` (an ES module built
with ``wasm-pack --target web``, for browsers, workers and privileged
Firefox code such as Zotero plugins)::

    npm install impact-index

Its version follows the Python package. ``init()`` fetches
``impact_index_wasm_bg.wasm`` next to the module; when loading it
differently (Node.js, extensions), pass the bytes or a URL:
``await init({ module_or_path: bytes })``.

Building
--------

To build from a checkout (with the nightly toolchain pinned in
``rust-toolchain.toml``)::

    rustup target add wasm32-unknown-unknown
    wasm/build-npm.sh            # the npm package, in wasm/pkg
    wasm-pack build --release --target web wasm   # plain wasm-pack build

``--target nodejs`` or ``--target bundler`` give packages for Node.js or for
bundlers (webpack, Vite...). Without ``wasm-pack``, build with ``cargo build
-p impact-index-wasm --target wasm32-unknown-unknown --release`` and run
``wasm-bindgen`` (the same version as the ``wasm-bindgen`` crate in
``Cargo.lock``) on ``target/wasm32-unknown-unknown/release/impact_index_wasm.wasm``.

Indexing and search
-------------------

.. code-block:: javascript

    import init, { Builder, Index } from "impact-index";
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

Streamed output
~~~~~~~~~~~~~~~

``builder.finish(output)`` hands the files to callbacks instead of
returning them, in pieces of at most ``chunkSize`` bytes (1 MiB by
default), so they never sit whole in the WASM memory:

.. code-block:: javascript

    builder.finish({
        create(name) { /* optional: open the file */ },
        write(name, bytes) { queue.push([name, bytes]); },  // bytes: a copy
        close(name) { /* optional */ },
        chunkSize: 1 << 20,
    });
    await drain(queue);  // e.g. IOUtils writes, in order

The callbacks are synchronous and their return value is ignored: queue
asynchronous writes, and wait for the queue before loading the index or
writing anything that refers to it. An exception thrown by a callback aborts
``finish`` with that error; the builder can be finished again.

Spilling builder
~~~~~~~~~~~~~~~~

By default, ``Builder`` keeps every posting in the WASM memory until
``finish`` (about 8 bytes each). With the ``spill`` option, postings over a
memory budget go to a scratch file of the caller's, appended to through
``write``, and are read back by ``finish``:

.. code-block:: javascript

    const builder = new Builder({
        range: { min: 0, max: 3 },
        spill: {
            budget: 16 << 20,               // bytes of postings (64 MiB)
            write(bytes) { queue.push(bytes); },  // append, in order
        },
    });
    // ... builder.add(...) ...
    await drain(queue);  // every spill write must be done
    builder.finish(output, (offset, length) => readSync(scratch, offset, length));
    // then delete the scratch file

When the postings exceed ``budget``, the largest posting lists are appended
to the scratch file until half the budget remains. ``finish`` reads them
back term by term, through read-ahead buffers that also fit in ``budget``,
so a 16 MiB budget costs a few dozen reads for 10M postings. The WASM memory
still reaches about twice the budget (allocator slack, per-term data), plus
what ``finish`` needs for the term table. ``builder.memoryBytes`` and
``builder.spilledBytes`` give the current sizes.

An exception in ``write`` leaves the builder unusable (the scratch file is
incomplete): later calls fail, and the index must be built again.

``Index.load`` accepts any compressed or split index, and forward indexes,
with file names relative to the index directory (e.g. ``inner/docids.dat``
for a split index). Search uses MaxScore. Document ids are returned as a
``Uint32Array``, so they must be below 2\ :sup:`32`.

Lazy loading
~~~~~~~~~~~~

``Index.load`` copies every file into the WASM memory. ``Index.open`` reads
only the metadata (``index.bin``...), and each search then reads the posting
lists of its terms, through a synchronous ranged reader:

.. code-block:: javascript

    const index = Index.open({
        // size of a file, undefined if it does not exist
        size(name) { return sizes.get(name); },
        // exactly `length` bytes of `name` at `offset`
        read(name, offset, length) {
            return new Uint8Array(new FileReaderSync().readAsArrayBuffer(
                files.get(name).slice(offset, offset + length)));
        },
    }, { cacheBytes: 16 << 20 });
    const { docids, scores } = index.search(terms, weights, 10);

In a Firefox worker (e.g. Zotero), ``File`` + ``FileReaderSync`` as above
reads ranges without copying the file; in a web page, an OPFS
``createSyncAccessHandle`` or a ``File`` from an ``<input>`` works too.
Posting lists are cached, up to ``cacheBytes`` bytes (64 MiB by default,
least recently used out; the lists of the current query always stay).
``index.reads`` and ``index.cacheBytes`` give the number of reads and the
cache size.

Each search first reads the lists it needs, so a read error (e.g. a deleted
file) is thrown by ``search``, and the index can be searched again once the
file is back. Compressed and split indexes (including positions) are read
lazily; ``SegmentedIndex`` accepts lazy and loaded segments alike.

The block table (``index.bin``, a few percent of the index) stays as read:
the blocks of a term are decoded when a query uses it, and cached (and
evicted) with its posting lists. On 10M postings (12.8 MB of files, 1.5 MB
of ``index.bin``), ``open`` takes 2.9 MB of WASM memory against 21 MB for
``load``, and a 30-term query 0.35 ms instead of 0.22 ms (Node.js, files in
the OS cache); after 200 queries with an 8 MiB cache, the WASM memory is
8.3 MB.

Incremental indexing
--------------------

The :ref:`incremental pattern <incremental>` carries over: ``DocSet`` holds
the deleted documents of a segment, and ``SegmentedIndex`` searches several
segments as one.

.. code-block:: javascript

    import { Builder, DocSet, Index, SegmentedIndex } from "impact-index";

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

Document ids and keys
---------------------

Search results are document ids; a key-only :doc:`document store
<docstore>` maps them to keys such as a ``docno`` and back. Build it in
WASM with ``KeyStoreBuilder``, or natively (``content=False``):

.. code-block:: javascript

    import { KeyStoreBuilder, DocumentStore } from "impact-index";

    const keys = new KeyStoreBuilder();
    keys.add({ docno: "D-17" }, 0);       // document id 0 (as in Builder.add)
    keys.add({ docno: "D-42" });          // without an id: the next one (1)
    const store = DocumentStore.load(keys.finish());

    store.getKeys("docno", docids);       // docno of each hit (or undefined)
    store.getDocids("docno", ["D-42"]);   // [1]

Stores built natively with content also load: their keys can be looked up
(format version 2), but not their content, which needs zstd.

Errors (corrupt or incomplete index files, invalid documents...) are thrown
as JavaScript exceptions, and the module stays usable.

Limits
------

- Single-threaded: no ``rayon`` and no shared memory.
- ``Index.load`` copies the index files into the WASM memory (no memory
  mapping), so memory grows with the index size: plan for the size of the
  index files plus some 30 MB, or use ``Index.open``. On 200k synthetic documents (45 MB index),
  a top-100 search takes 1-3 ms in Node.js.
- ``Builder`` keeps the postings in memory until ``finish``, at about 8
  bytes per posting (10M postings: ~80 MB), plus the compressed files,
  unless the postings are spilled (``spill`` option) and the files streamed
  (``finish(output)``).
- Not available in WASM: BM25 / text indexing, positions and structured
  queries, document contents (key-only stores work), BMP and Seismic, and
  builder checkpoints.
- Licence: MIT OR Apache-2.0, like the rest of the library (``wasm-pack``
  copies ``LICENSE-MIT`` and ``LICENSE-APACHE`` into the package). The
  Lucene-derived analysis code, Apache-2.0 only, is not part of the WASM
  build.

Rust features
-------------

The WASM crate depends on ``impact-index`` without its default features,
and with ``lazy-pages``. They can also be disabled in other Rust projects:

``python``
    Python bindings
``mmap``
    memory-mapped index files (otherwise, files are read into memory)
``parallel``
    multi-threaded indexing and compression with ``rayon``
``progress``
    progress bars
``zstd``
    document contents in document stores, and builder checkpoints (zstd
    is a C library); key-only stores do not need it
``bmp``
    conversion to BMP indexes and CIFF export
``lazy-pages`` (off by default)
    indexes read through a ranged reader (``storage::ReaderFiles``, as
    ``Index.open``) decode their block table term by term, cached with the
    posting lists; on in the WASM build. Without it, such indexes decode
    the whole table when opened
``analysis``
    text analyzers (tokenizers, stemmers, stop words, Lucene analyzers,
    ICU4X word segmentation); pure Rust, it also builds for WASM, but the
    WASM bindings do not expose it. Without it, documents and queries are
    given as term ids
``analysis-download``
    on-demand download of the analysis data (ICU4X segmentation data,
    Japanese / Korean dictionaries) into the cache directory; native only.
    Without it, the paths of the data must be set explicitly
``analysis-cjk``
    Japanese and Korean morphological analysis (Lindera); implies
    ``analysis-download``
