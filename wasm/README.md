# impact-index

WebAssembly build of [impact-index](https://github.com/xpmir/impact-index),
a sparse retrieval library for learned sparse models (SPLADE and others):
build compressed impact indexes and search them with exact block-max
WAND / MaxScore, in a browser page, a web worker, Node.js or privileged
Firefox code (e.g. a Zotero plugin).

Indexes are byte-compatible with the native library (Rust and the
`impact-index` Python package): an index built natively loads here, and the
other way round. Index files are passed in and returned as maps from file
names to `Uint8Array`s, so the application stores them where it likes
(IndexedDB, the Origin Private File System, a server, files...).

```javascript
import init, { Builder, Index } from "impact-index";
await init();

// Build: document ids must be increasing; terms are vocabulary indices
const builder = new Builder({ range: { min: 0, max: 3 } });
builder.add(0, new Uint32Array([12, 40]), new Float32Array([0.8, 1.5]));
builder.add(1, new Uint32Array([3, 12]), new Float32Array([2.1, 0.2]));
const files = builder.finish(); // Map: file name -> Uint8Array
// or stream them out: builder.finish({ write(name, bytes) { ... } })

// Load (from a Map or a plain object) and search
const index = Index.load(files);
// or read posting lists lazily: Index.open({ size(name), read(name, offset, length) })
const { docids, scores } = index.search(
    new Uint32Array([12, 40]), new Float32Array([1.2, 0.4]), 10);
```

`init()` fetches `impact_index_wasm_bg.wasm` next to the module; pass the
bytes or a URL (`init({ module_or_path })`) when loading it differently.

Incremental indexing (segments with deleted documents, `SegmentedIndex`),
key-only document stores and the limits of the WASM build are described in
the [documentation](https://impact-index.readthedocs.io/en/latest/wasm.html).

Licensed under either of the Apache License, Version 2.0 or the MIT license,
at your option.
