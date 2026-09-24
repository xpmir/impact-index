# Python Documentation (Development)

This directory contains the Python type stubs and Sphinx documentation source for the `impact_index` module.

For the full user documentation, see: https://experimaestro-ir-rust.readthedocs.io/en/latest/index.html

## Files

- `impact_index.pyi` — Type stubs providing IDE autocompletion and type checking
- `docs/` — Sphinx documentation source

## Building the Documentation

```bash
cd python
make docs
```

Then open `docs/_build/index.html`.

## Updating `impact_index.pyi`

The `.pyi` stub file is **auto-generated** by `pyo3-stub-gen`. To regenerate:

```bash
cargo run --bin stub_gen --no-default-features --features stub-gen,seismic
mv impact_index.pyi python/impact_index.pyi
```

### When to regenerate

- A new `#[pyclass]` or `#[pymethods]` block is added
- A method signature changes (arguments, return type)
- A method or class is removed or renamed

### Known limitations

Some `#[pymethods]` blocks cannot use `#[gen_stub_pymethods]` due to
upstream issues in `pyo3-stub-gen`:

- **`PyArray1<usize>`**: `usize` doesn't implement `NumPyScalar`
  ([#97](https://github.com/Jij-Inc/pyo3-stub-gen/issues/97))
- **`&[u8]`**: byte slices don't implement `PyStubType`
  ([#97](https://github.com/Jij-Inc/pyo3-stub-gen/issues/97))
- **`(Self, Parent)` constructors**: `#[new]` returning a tuple for
  inheritance causes `Self` scope errors

These blocks are skipped (with comments in `src/py/mod.rs`) and their
stubs appear as empty `...` in the generated file. When the upstream
issues are fixed, add `#[gen_stub_pymethods]` back.
