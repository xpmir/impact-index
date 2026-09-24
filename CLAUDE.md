# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is **impact-index**, a Rust library with Python bindings for efficient sparse retrieval from neural information retrieval systems. Unlike traditional IR libraries, it specifically targets neural IR models with floating-point impact scores (no quantization assumed) and does not implement frequency-based algorithms.

The project includes **BMP (Block-Max Pruning)**, a sub-crate implementing the algorithm from "Faster Learned Sparse Retrieval with Block-Max Pruning" (SIGIR 2024).

## Build Commands

### Rust
```bash
# Build library (release)
cargo build --release

# Run tests
cargo test

# Run a specific test
cargo test test_search

# Run benchmarks
cargo bench
```

### Python Bindings (impact-index)
```bash
# Build and install (from root directory)
pip install maturin
maturin develop --release
```

### Seismic backend (`seismic` feature)
Pinned git deps (`seismic`, `vectorium`) that need a recent nightly, hence
the dated channel in `rust-toolchain.toml`. The feature is off for plain
`cargo build`/`cargo test` but enabled for the Python package
(`pyproject.toml`), so `maturin develop` includes it.
```bash
cargo test --features seismic --test seismic
```
When bumping the pinned `seismic` rev, also bump `SEISMIC_FORMAT` in
`src/seismic/mod.rs` and keep `vectorium` at the rev seismic's lockfile uses.
When bumping the nightly, update `rust-toolchain.toml` and the
`rust-toolchain`/`rustup` lines in `.github/workflows/*.yml` together.

### Python Bindings (BMP)
```bash
# Build and install (from BMP/python directory)
cd BMP/python
maturin develop --release
# or
maturin build --release && pip install target/wheels/*.whl
```

## Architecture

### Two Index Systems

1. **impact-index** (root crate): Full-featured sparse index with compression and transforms
   - Python module: `impact_index`
   - Entry point: `src/lib.rs` and `src/py/mod.rs`

2. **BMP** (BMP/ subdirectory): Block-Max Pruning implementation
   - Python module: `bmp`
   - Entry point: `BMP/src/lib.rs` and `BMP/python/src/lib.rs`

### Core Components (impact-index)

- **`src/builder.rs`**: Index construction with checkpointing support (`Indexer`, `BuilderOptions`)
- **`src/index.rs`**: `SparseIndex` trait and block-based iterator interfaces (`BlockTermImpactIterator`)
- **`src/search/`**: Search algorithms (WAND in `wand.rs`, MaxScore in `maxscore.rs`)
- **`src/compress/`**: Compression schemes for doc IDs (Elias-Fano) and impact values
- **`src/transforms/`**: Index transforms including split index (`split.rs`)
- **`src/seismic/`** (feature `seismic`, on in the Python package): conversion to / search over a Seismic approximate index (`convert_to_seismic`, `SeismicSearcher`)
- **`src/py/mod.rs`**: Python bindings exposing `IndexBuilder`, `Index`, compression classes

### Core Components (BMP)

- **`BMP/src/index/`**: Inverted and forward index structures
- **`BMP/src/query/`**: Query cursors and live block processing
- **`BMP/src/search.rs`**: BMP search algorithm implementation
- **`BMP/src/ciff/`**: CIFF (Common Index File Format) import

### Key Types

- `DocId` = `u64`: Document identifier
- `TermIndex` = `usize`: Term/token index
- `ImpactValue` = `f32`: Impact score (must be > 0)
- `TermImpact`: Struct containing `(docid, value)`

### Data Flow

1. **Indexing**: Documents added via `Indexer.add(docid, terms[], values[])` -> builds forward index with optional checkpointing
2. **Transform**: Optional compression/splitting via `IndexTransform` implementations
3. **Search**: Query as `HashMap<TermIndex, ImpactValue>` -> WAND or MaxScore algorithm -> `Vec<ScoredDocument>`

### Index Formats

- **Forward index**: Raw postings stored in `postings.dat` with metadata in `information.cbor`
- **Split index**: Separates high/low impact postings based on quantiles
- **BMP index**: Binary format with inverted + blocked forward index for efficient pruning

## Python Type Stubs

`python/impact_index.pyi` is **auto-generated** by `pyo3-stub-gen`. Regenerate with:

```bash
cargo run --bin stub_gen --no-default-features --features stub-gen,seismic
mv impact_index.pyi python/impact_index.pyi
```

Some `#[pymethods]` blocks are skipped due to upstream `pyo3-stub-gen` limitations (see comments in `src/py/mod.rs` and `python/README.md`).

## Testing

Tests are in `tests/` directory. The test helper library in `libs/helpers/` provides `TestIndex` for generating random test indices.

```bash
# Run all tests with logging
RUST_LOG=debug cargo test

# Run specific test
cargo test test_index
```
