# Performance Optimizations

Current status (vs Pyserini BM25 on MS MARCO passage, 1000 queries):
- macOS ARM: ~203 q/s vs 211 q/s (4% gap)
- x86_64: ~70 q/s vs 84 q/s (17% gap)

## Summary table

Measurement protocol: `examples/profile_search.py --output-dir ~/temporary/bm25_bench
--max-queries 0 --passes 3` — MS MARCO passage dev/small (6980 queries), BM25
k1=0.9 b=0.4, top-100, MaxScore on the PFOR-compressed index (`index_pfor_nb0_bs128`),
in-memory, median pass. Measured cumulatively in the suggested order at the bottom.

Baseline (2026-08-31, master e40772d): **mac 188.9 q/s**, **xian 67.6 q/s**
(Pyserini reference: mac 213.0 q/s, xian 89.6 q/s; MRR@10 0.1858 vs 0.1855 — matches).
Build for measurements: `maturin build --release` + install into `~/temporary/ii-venv`
(do NOT trust `uv run --with .` caching — it silently reuses stale wheels).

| # | Optimization | Kind | Est. gain | Effort | ARM/x86 notes | mac (q/s) | xian (q/s) |
|---|--------------|------|-----------|--------|---------------|-----------|------------|
| P8 | `lto=fat`, `codegen-units=1`, then PGO | build | 3–8% | trivial (TOML) / small (PGO) | PGO works on both; BOLT Linux-only | 195.0 (+3.2%) | 70.0 (+2.9%) |
| P1a | Store per-block (and per-term) **min doc length** at index time — model-agnostic statistic; any dl-monotone scorer (BM25, LM-Dirichlet) gets tight bounds instead of global `min_dl` | index stat | 5–15% (more pruning) | small | arch-neutral; format addition, scoring stays query-time | 278.2 (+0.9%)¹ | 100.5 (−3.8%)¹ |
| P1b | **Batch scoring per block**: when a block is entered, gather norms + score all/lazy-chunks of 128 postings into an `[f32;128]` buffer (vectorized divide, pipelined norm loads) instead of per-posting scalar score | scoring layer | 5–10% | medium | vectorizes on NEON+AVX2; amortizes the f16 lookups driving the x86 gap | measured with P3 ↑ | |
| P4 | Cursor API: `next()` = index+1, `next_geq` = galloping from current pos (replaces per-posting `partition_point`) | data structure | ~5% | medium | arch-neutral | 240.5 (+23.3%, with P5) | 83.6 (+19.4%, with P5) |
| P5 | u32 doc IDs inside decoded blocks (`[u32;128]`+`[f32;128]`) | data layout | 3–5% | small–medium | halves buffer footprint; feeds SIMD on both | measured with P4 ↑ | |
| P3 | Monomorphize search loop over **(cursor × scorer)** pairs — generic `ScoringCursor<C, S>`, one enum dispatch per query, zero vtables per posting | Rust-specific | 8–15% (two vtable layers removed) | medium | arch-neutral; Rust's answer to JVM devirtualization | 275.8 (+14.7%, with P1b) | 104.5 (+25.0%, with P1b) |
| P2 | Doc reordering by recursive graph bisection (BP) | algorithm (index-time) | 20–40% query + 10–30% smaller index | large (~200 lines + rebuild) | arch-neutral; also boosts BMP | 294.8 (+6.0%)² | 106.0 (+5.5%)² |
| P6 | Fuse MaxScore passes; f32 accumulation; `peek_mut` heap; incremental WAND sort | search loop | 2–4% | small | arch-neutral | | |
| P7 | SIMD block scoring (autovectorized fixed-size loops; `wide`/`std::simd` fallback) | SIMD | 2–5% | small–medium | NEON + AVX2 from one source; avoid raw intrinsics | | |
| P9 | Interleave docid+impact block bytes; zero-copy EF or drop EF default; prefetch next block | storage | 1–3% | small | prefetch intrinsic unstable on aarch64 → volatile-read trick | | |
| P10 | Route quantized top-k queries to BMP (already in repo) | engine choice | large (per SIGIR'24) | small (glue) | **deferred**: BMP bakes scores at conversion — conflicts with query-time model choice | | |
| P1-later | Precompute quantized impacts at index time (idf folded in, 8-bit, no scoring layer) | algorithm + format | 15–25% | medium | **deferred**: freezes the scoring model at index time — revisit as an *optional* transform per model | | |

¹ P1a is near-neutral on mac and −3.8% on xian with the current random doc
ordering: a 128-posting block almost always contains a short document, so
block min_dl ≈ global min_dl — pure bookkeeping overhead, no pruning gain.
On skewed/clustered lengths the same code gives +56% (synthetic bin, bimodal
lengths). Decision deferred to after P2 reordering (which clusters similar
documents into blocks): if P1a is still net-negative on reordered MS MARCO,
revert its query-time use and keep only the format/stat. Migration of the
620MB MS MARCO index via `Index.update`: 1.0s (mac), 3.3s (xian).

² P2 on MS MARCO: reorder+recompress of 8.8M docs in 416s (mac) / 781s
(xian); index 675MB vs 706MB (−4.4%); MRR@10 verified via reorder_map
(mac Δ0.0002 tie reordering, xian Δ0.0000). Includes the P1a bound working
on reordered blocks — on xian, P2's gain (+5.5% over the P1a state) mostly
recovers P1a's −3.8%; net over P3+P1b is +1.4% (mac: +6.9%). Isolating
"P2 without P1a bound" is an open follow-up. Below the 20–40% literature
figures (those are BMW-centric evals).

## Measured result (2026-08-31, branch perf/roadmap @ 5ecda42)

| | mac (M-series) | xian (x86-64) |
|---|---|---|
| Baseline (master e40772d) | 188.9 q/s | 68.0 q/s |
| + P8 fat LTO | 195.0 | 70.0 |
| + P4+P5 cursor/u32 | 240.5 | 83.6 |
| + P3+P1b monomorph/batch | 275.8 | 104.5 |
| + P1a min_dl bounds | 278.2 | 100.5 |
| + P2 BP reordering | **294.8** | **106.0** |
| **Cumulative** | **+56%** | **+56%** |
| Pyserini (Lucene) | 213.0 (−28% vs us) | 89.6 (−15% vs us) |

Gains are not additive (P1b/P3 shrink what P4 optimizes). `bitpacking` 0.9.3
verified to have NEON kernels for `BitPacker4x`, so the SIMD decode format is
already portable; `BitPacker8x` (AVX2, ~2× decode) would fork the on-disk
format and is scalar-only on ARM — keep 4x as the default.

The plan below supersedes the earlier priority list (kept at the bottom).
The earlier list was about closing the gap with Lucene by imitating it;
this one is about beating it by *not* imitating it. Ordered by estimated
impact. Everything here is portable ARM (NEON) / x86 (AVX2) — verified
that `bitpacking` 0.9.3 has real NEON kernels for `BitPacker4x`, so the
SIMD decode path is already portable.

## Constraint: scoring model chosen at query time

The scoring model (BM25, LM, …) must stay a query-time choice, so nothing
model-specific may be baked into the postings. Consequences:

- **P1-later** (precompute quantized impacts) and **P10** (BMP conversion,
  which quantizes scores) are deferred. Both remain the biggest known wins
  and can come back later as *optional per-model transforms* of the same
  raw-TF index.
- The `ScoringBlockIterator` layer stays, so removing its *per-posting*
  overhead (P1b, P3) matters more, and pruning must be improved through
  model-agnostic statistics (P1a).

## P0-fmt. Index manifest, versioning, and migration (prerequisite for P1a/P2)

Requirement (user, 2026-08-31): any index-format change must come with an
upgrade path and a clear error, not a crash or silent misread.

- Add a `manifest.json` to every index directory: format version, index
  type, build parameters (block size, codecs, analyzer), creation date.
- On load, check the version. Mismatch → actionable error:
  `"index format v3, this build requires v4 — run `impact-index update <dir>`
  (or Index.update(path) in Python) to migrate"`.
- Fix `CompressedIndexInformation::read_binary` (src/compress/mod.rs), which
  reads but ignores the stored version today.
- Provide the migration itself: `Index.update(path, dest=None)` — rewrites
  an old-format index into the current format (for P1a: recompute per-block
  metadata from docmeta; for P2: full re-transform). Keep readers for one
  version back where cheap; otherwise migrate.

## P1a. Model-agnostic index statistics for tight bounds (~5–15%)

`BM25TermScorer::max_score` bounds every term and block with the *global*
`min_dl` — the shortest document in the whole collection (often 1–2
tokens), which makes `theta`-pruning far weaker than it could be.

Fix without touching model-agnosticism: at index time, store alongside
`max_value` in `TermBlockInformation` (and per-term):

- `min_dl_in_block`: minimum length among documents in the block
  (or equivalently the minimum norm-relevant statistic)

Any scorer whose TF-component is monotone decreasing in document length
(BM25, LM-Dirichlet query likelihood) can then compute
`max_score(block) = score(max_tf_in_block, min_dl_in_block)` — a much
tighter, still *safe* upper bound, evaluated with whatever model/params
the query chose. Extend `ScoringFunction` with
`max_score_with_dl(max_tf, min_dl)`; the block-metadata cost is 2 bytes
per block (dl quantized, rounded down to stay safe).

This recovers a good share of the pruning benefit that score-baking would
have given, while keeping the index raw.

## P1b. Batch scoring per block (~5–10%)

Today each posting scored costs: vtable `current()` → lazy `Cell` check →
`doc_norms[docid]` f16 random load → scalar divide. Instead, when the
scoring loop actually enters a block (not on shallow advances), score in
batches: decode already yields `docids[128]` / `tfs[128]`; gather the 128
norms into a local array, then compute `idf * tf / (norm + tf)` over the
whole array. Benefits:

- the divide and the f16→f32 conversion vectorize (NEON `vcvt`+`fdiv`,
  AVX2 `vcvtph2ps`+`vdivps` — both already enabled by `target-cpu` flags)
- the norm loads become independent → the CPU pipelines/overlaps the
  cache misses that are currently serialized per posting (this is the
  likely main driver of the 17% x86 gap)
- `current()` becomes a plain array read

To avoid wasted work when WAND/BMW touches only a few postings of a
block, score lazily in chunks (e.g. 32) from the current cursor position
rather than the full 128 up front. Works identically for BM25 and LM —
the batch kernel is just the scorer's `score_block(&tfs, &docids, &mut out)`
method with a default scalar implementation.

## P2. Document reordering: recursive graph bisection (~20–40% + smaller index)

Index-time transform, orthogonal to everything else, benefits every
algorithm including BMP. Renumber doc IDs by recursive graph bisection
("BP", Dhulipala et al. 2016; Mackenzie et al. 2021 for the IR numbers):
similar documents get nearby IDs, so

- deltas shrink → PFOR/bitpacking bit-widths drop (index ~10–30% smaller,
  faster decode)
- block-max values become skewed instead of uniform → block-max pruning in
  WAND/MaxScore/BMP skips far more blocks. Published MS MARCO numbers for
  BMW-style algorithms are 1.2–2× throughput.

Fits naturally as an `IndexTransform` (permute doc IDs, re-sort postings,
rewrite doc metadata). BP is embarrassingly parallel per recursion level →
rayon. ~200 lines. This is the best "new algorithm" ROI in the whole list.

## P3. Kill dynamic dispatch via monomorphization over (cursor × scorer) (~8–15%)

The BM25/LM hot path currently pays **two** vtable indirections per
posting (`ScoringBlockIterator` → inner iterator) plus one for
`ScoringFunction::score`. Since the model is chosen at query time, make
the *combination* generic and dispatch once per query:

```rust
pub trait TermCursor { fn docid(&self) -> DocId; fn value(&self) -> f32; ... }
pub struct ScoringCursor<C: TermCursor, S: ScoringFunction> { inner: C, scorer: S }

fn search_maxscore_typed<C: TermCursor>(cursors: Vec<C>, ...) -> Vec<ScoredDocument>

// entry point: one match per query, monomorphized loops inside
match model {
    Model::BM25 => search_maxscore_typed(make_cursors::<_, BM25TermScorer>(...)),
    Model::LM   => search_maxscore_typed(make_cursors::<_, LMTermScorer>(...)),
    Model::None => search_maxscore_typed(raw_cursors(...)),
}
```

The set of scorers is small and closed (that's the query-time menu), so
the monomorphization cost is a handful of loop instantiations. Keep the
existing `dyn` entry points as fallback wrappers. This is how Rust beats
the JVM's profile-guided devirtualization: compile-time monomorphization
+ inlining, no warmup — and it composes with P1b (the batched
`score_block` inlines into the search loop).

## P4. Cursor API: stop binary-searching every posting (~5%)

`CompressedBlockTermImpactIterator::current()` runs `partition_point`
(≈7 probes for a 128-block) on *every* posting access, even when the
caller is walking sequentially, plus a `Cell<Option<TermImpact>>` check
and a `current_min_docid` `Option` unwrap. The Lucene-style
"shallow advance + lazy current" contract forces this.

Replace with a PISA-style cursor over the decoded block:

- `docid()` / `score()`: plain array reads at `self.index`
- `next()`: `self.index += 1` (+ block reload when exhausted)
- `next_geq(d)`: galloping (exponential) search *from the current
  position* — O(log gap) instead of O(log block), and O(1) when the
  target is the next posting, which is the common case in scoring loops.

This also subsumes old P2 (the builder iterator gets the same cursor).

## P5. u32 doc IDs inside blocks (~3–5%)

`DocId = u64` forces `Vec<u64>` decode buffers. Consequences in
`decode_into_bytes`: the SIMD bitpacker produces `[u32; 128]`, then a
*scalar* widen-and-add loop copies into the u64 buffer; buffers are 1KB
instead of 512B per block (worse L1 behavior); `partition_point` compares
u64s. Keep block-internal storage as `[u32; 128]` offsets (or absolute
u32 — MS MARCO and anything indexable in RAM fits), widen only at the
API boundary (`ScoredDocument`). Combined with P4 the decoded block
becomes two flat arrays `[u32; 128]` + `[f32; 128]` — which is also the
layout SIMD scoring (P7) wants.

## P6. Fuse the MaxScore per-candidate passes (~2–4%)

`search_maxscore` walks `active` three times per candidate: a fold for
the min docid, a filter+sum for `block_ub`, and `retain_mut` to score.
Fuse into one pass (compute next-candidate-min while scoring the current
one). Also:
- `passive.remove(i)` is O(n) — swap with last + re-sort-on-insert instead
- accumulate in f32, not f64 (the f32↔f64 conversions are pure overhead;
  with 8-bit quantized impacts even i32 accumulation works and is exact)
- `TopScoredDocuments::add`: use `heap.peek_mut()` (one sift) instead of
  pop+push (two sifts)
- WAND's `find_pivot_term` re-sorts the full cursor list per iteration —
  irrelevant for 5-term BM25 queries, but replace with incremental
  reinsertion of the moved cursor before pointing SPLADE queries
  (dozens–hundreds of terms) at it.

## P7. SIMD block scoring — portable via explicit 128-bit kernels (~2–5%)

With P1+P5 in place, the inner scoring work becomes "dequantize 128 u8s,
multiply-add" and "compare 128 u32 docids" — ideal SIMD shape. Two
portable options, in order of preference:

1. Write the per-block loops over fixed `[u32; 128]` / chunks of 4–8 with
   no data-dependent branches; LLVM auto-vectorizes these for both NEON
   and AVX2 (already enabled via `target-cpu` flags in `.cargo/config.toml`).
   Verify with `cargo asm` / `--emit asm`.
2. Where autovectorization fails, `std::simd` (portable SIMD) on nightly,
   or the `wide` crate on stable — both compile to NEON and AVX2.

Avoid raw `core::arch` intrinsics except as a last resort (double
implementation, and aarch64 prefetch intrinsics aren't stable anyway).

Note on `BitPacker8x` (256-bit, ~2× decode on AVX2): its wire format
differs from `BitPacker4x`, and on ARM it only has a scalar fallback.
Only worth it behind a per-index format flag if indexes never move
between machines. Default should stay `BitPacker4x` = one portable format.

## P8. Build/toolchain (free ~3–8%)

- `Cargo.toml` has **no `[profile.release]` section** — defaults are
  16 codegen units, thin-local LTO. Add:
  ```toml
  [profile.release]
  lto = "fat"
  codegen-units = 1
  ```
  With this much cross-module `dyn` code, fat LTO gives LLVM its only
  chance to devirtualize/inline across units. (Do *not* add
  `panic = "abort"` — pyo3 needs unwinding.)
- PGO via `cargo-pgo`: works on both macOS/ARM and Linux/x86; branchy
  search loops are exactly the code shape PGO helps (5–15% typical).
  Profile with the real MS MARCO query workload.
- Check that maturin builds actually pick up `.cargo/config.toml`
  rustflags (`maturin develop --release -v`).

## P9. Storage-layer options (smaller / situational)

- **Elias-Fano**: `EliasFanoCompressor::read` calls
  `EliasFano::deserialize_from` per block — heap-allocates and copies the
  whole structure on every block decode. If EF stays as an option, use a
  zero-copy view over the mmap bytes (recent `sucds` versions support
  borrowed access; 0.5 is old). If not, document PFOR as the default and
  EF as cold-storage only.
- **Block layout**: docids and impacts live in two separate files, so each
  block decode touches two distant cache/page locations. Interleaving
  per-block (docid bytes then impact bytes, contiguous) halves the fetch
  targets and gives one prefetch address. Cheap format change, modest win.
- **Prefetch**: after `move_iterator` selects the next block, touch its
  first byte(s) before finishing the current one. Portable "poor man's
  prefetch" (a volatile read) avoids unstable intrinsics.

## P10. Route to BMP when applicable (deferred — bakes scores)

The repo already contains the strongest published option: BMP (SIGIR
2024) dominates MaxScore/BMW for quantized BM25 and learned sparse at
typical k. But `convert_to_bmp` quantizes *scores*, which freezes the
model at conversion time — incompatible with the query-time-model
constraint. Revisit together with P1-later as an optional per-model
acceleration structure next to the raw index.

## Suggested order

1. P8 (one paragraph of TOML, measure)
2. P4 + P5 together (one refactor of the iterator/decode layer)
3. P3 (monomorphize cursor × scorer) + P1b (batched scoring) — same
   refactor of the scoring layer, do them in one pass
4. P1a (tight model-agnostic bounds; small format addition)
5. P2 (independent; biggest algorithmic win, benefits everything)
6. P6, P7, P9 as measured profiles dictate

Re-profile after each step (`profile_search` bin) on both machines; the
estimates interact (P1b/P3 shrink the pie that P4 optimizes).

---

## Earlier priority list (superseded)

- ~~Priority 1: Monomorphize the search loop~~ → P3 (do after P1)
- ~~Priority 2: Optimize the uncompressed (builder) iterator~~ → P4
- ~~Priority 3: Precompute BM25 scores at index time~~ → P1 (extended:
  fold IDF in too, search the quantized index directly, drop the scoring
  layer entirely)
- ~~Priority 4: Reduce retain_mut overhead in MaxScore~~ → P6
- ~~Priority 5: SIMD block scoring~~ → P7

## Completed optimizations

- **MmapBuffer::as_bytes()**: eliminated heap allocation on every block decode
  when using mmap (was copying mmap data into Vec)
- **Cached block metadata in CompressedBlockTermImpactIterator**: `max_block_value()`,
  `max_block_doc_id()`, `min_block_doc_id()` are now plain field reads instead of
  `RefCell::borrow()` calls
- **Cell instead of RefCell** for `current_value` in compressed iterator
- **Cached block metadata in ScoringBlockIterator**: all block metadata refreshed
  once per `next_min_doc_id()`, eliminating inner vtable calls from WAND tight loops
- **Pre-computed BM25 constants**: `k1*(1-b)` and `k1*b/avgdl` computed once
- **f16 doc norms**: precomputed BM25 norms stored as f16 (17.5MB vs 35MB for u32
  doc_lengths), reducing cache pressure and eliminating per-posting norm arithmetic
- **`get_unchecked` for doc norms**: skip bounds checking in BM25 hot path
- **`target-cpu=x86-64-v3`** in `.cargo/config.toml`: enables AVX2/F16C/BMI2
  for x86_64 builds (was defaulting to SSE2-only)
