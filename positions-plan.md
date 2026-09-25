# Positions + structured queries: implementation plan

Goal: store token positions when indexing with a standard BoW model (tf values,
query-time scoring), and expose Terrier-matchop-style structured operators —
`#band`/`#syn`/`#combine` (doc-level) and `#1`/`#uwN` (positional) — on top of
the existing WAND/MaxScore machinery, without touching the learned-impact path.

Design stance (from the Lucene/Terrier comparison):
- **Terrier's architecture**: every operator evaluates to a *virtual posting
  list* (a `BlockTermImpactIterator`), whose "tf" is scored by the query-time
  scoring model like an ordinary term. WAND/MaxScore loops stay unchanged.
- **Lucene's discipline**: every composite cursor must report honest
  `max_value()` / cost estimates so dynamic pruning stays safe; positions are
  a separate, lazily-read stream (`positions.dat`, like Lucene's `.pos`).

Settled decisions:
- Exact token positions (no Terrier-style coarse blocks).
- Stopword removal leaves **position gaps** (Lucene behavior): `#1(new york)`
  must not match "new the york". **Revised 2026-09-18:** now an index-time
  option, `position_gaps`; `pipeline="terrier"`/`"terrier-pisa"` default to
  no gaps (Terrier 5). Matchop queries drop stop words inside `#1`, so
  `#1(bank of america)` = `#1(bank america)`, which can only match "bank of
  america" without gaps.
- Positions are **opt-in per index** (`positions=true`), BoW-only: rejected
  unless the posting value is a count-like tf. Learned-impact indices are
  untouched.
- Virtual-term scoring: phrase/window tf scored by the query-time model with
  **sum of children's idfs** (Lucene's convention for phrases); doc-level
  `#band` emits tf = 1, `#syn` sums children tfs (Terrier semantics).
  **Revised 2026-09-18: aligned on Terrier 5 throughout** (verified
  score-for-score against Terrier 5.11 on a mini corpus). Each operator is
  one virtual term: `#syn` df = sum of dfs; `#band` tf = 1, df = sum of dfs;
  `#1`/`#uwN` df = N/100 (Ivory heuristic); `#uwN` tf = seed-occurrence
  count (Terrier's `ProximityIterablePosting`). Nested `#combine` flattens
  (weights multiply, duplicate clauses merge); BM25 `k3` (opt-in) normalizes
  weights by the max, then `(k3+1)w/(k3+w)`. Rationale: sum-of-idfs
  overstates rarity (wrong for `#syn`), and no other convention was more
  principled than Terrier's. User-facing description: python/docs/bow.rst, "How
  structured queries are scored".
- Positional data lives in a **separate stream** at every level (builder file,
  compressed file, checkpoint) so non-positional reads stay byte-identical.

---

## Phase 1 — Capture and forward storage

### 1.1 Analyzer: keep token order and positions

`src/vocab/analyzer.rs`

- `tokenize` currently lowercases + filters stopwords inline, losing offsets.
  Split into: produce `(token, position)` pairs where `position` is the
  pre-stopword token index; stopword filtering then drops pairs but keeps the
  gap.
- New `analyze_doc_positional(&mut self, text) -> Vec<(TermIndex, Vec<u32>)>`
  (per-term sorted position lists; tf = positions.len()). Keep `analyze_doc`
  as-is for the non-positional path.
- Same for the thread-safe `tokenize_and_stem` used by `add_texts_batch`
  (positional variant returning `(String, Vec<u32>)`).

### 1.2 Builder: `positions.dat` alongside `postings.dat`

`src/builder.rs`, `src/bow.rs`

- `BuilderOptions` gains `positions: bool` (default false).
- `TermsImpacts` gains, when enabled:
  - `positions_file: BufWriter<File>` (`positions.dat`) + tracked position;
  - per-term in-memory `Vec<Vec<u32>>` parallel to `postings` in
    `PostingsInformation` (serde-skipped when empty so non-positional
    checkpoints are unchanged).
- `add_impact` gains a positional variant `add_impact_with_positions(term_ix,
  docid, tf, positions)`; assert `positions.len() == tf as usize` and
  positions strictly increasing. The non-positional `add` on a positional
  index is an error (all-or-nothing per index).
- `flush(term_ix)`: after writing the fixed-size `(docid, value)` records,
  write each posting's positions to `positions.dat` as delta-vbyte
  (first position absolute, then gaps). No per-posting count byte — the
  reader knows tf from the value stream. Record the page's start offset.
- `TermIndexPageInformation` (`src/index.rs`) gains
  `positions_position: u64` (0 = none). Bump `FORWARD_INDEX_VERSION` 1 → 2;
  `read_binary` keeps accepting v1 (positions_position = 0 everywhere).
- Checkpointing: the CBOR tuple becomes
  `(postings_information, pos, positions_pos, doc_id)`. The existing
  "incompatible checkpoint → delete checkpoint.cbor and re-index" panic
  already covers old checkpoints; keep reading the old 3-tuple when
  `positions` is off so non-positional resumption is unaffected.
- `BOWIndexBuilder`: `positions=true` constructor flag; `add_text` /
  `add_texts_batch` route through the positional analyzer; pre-tokenized
  `add` gains `positions: Option<&[Vec<u32>]>`. Doc length stays
  sum-of-tf = token count (unchanged, already correct for BoW).

### 1.3 BoW gating + manifest

`src/manifest.rs`, `src/base.rs`

- `Manifest` gains `features: Vec<String>` (serde-default empty, so old
  manifests parse); positional indices record `"positions"`. Additive — no
  `CURRENT_FORMAT_VERSION` bump needed for non-positional directories, since
  their layout is unchanged and old readers ignore unknown JSON fields is not
  needed (field is optional both ways).
- `positions=true` rejected at builder construction unless `V` is i32/i64 or
  the caller passes tf-valued f32 (accept f32 but document that the value
  must equal the position count; the assert in 1.2 enforces it).
- Actionable error path: any positional operator on an index without the
  feature → `"index was built without positions; rebuild with
  positions=true"` (never a panic mid-search).

### 1.4 Forward-index read path

`src/builder.rs` (`SparseBuilderIndex` + its iterator)

- The forward iterator gains lazy positional access (trait below, §3.1):
  on `current()`, if positions were requested, decode that posting's run.
  Because pages interleave (a term's pages are scattered through
  `postings.dat`), each page's positions run is located by
  `positions_position` + skipping over preceding postings' runs
  (skip = decode-and-discard vbytes; runs are short, and the forward index
  is not the hot search path).

### Phase 1 tests

- Analyzer: positions with stopword gaps, stemming, possessive filter.
- Round-trip: index docs with positions → read back per-posting position
  lists exactly (including multi-page terms, i.e. force `in_memory_threshold`
  small).
- Checkpoint/resume with positions on.
- Old index (no positions) loads; positional access returns None; positional
  op errors actionably.

---

## Phase 2 — Compressed path

### 2.1 View plumbing: positions through the transform

`src/index.rs`

- `SparseIndexView` gains a defaulted method:
  ```rust
  /// Per-term iterator aligned with `iterator(term_ix)`, yielding each
  /// posting's positions. None if the index stores no positions.
  fn positions_iterator<'a>(&'a self, term_ix: TermIndex)
      -> Option<Box<dyn Iterator<Item = Vec<u32>> + 'a>> { None }
  ```
  (Alignment contract: same length/order as `iterator`.) Implemented by the
  forward index; the blanket `SparseIndexView for T: SparseIndex` impl
  forwards a matching `SparseIndex` method.

### 2.2 CompressionTransform + format v5

`src/compress/mod.rs`

- `CompressionTransform.process`: when the source view has positions, each
  parallel chunk also fills a `positions_buf` (delta-vbyte per posting, no
  counts — tf comes from the impact stream, which for BoW is exact integer
  tf; assert on write). Written to `positions.dat` with the same
  chunk-sequential offset fixup as docids/impacts.
- `TermBlockInformation` gains `positions_position_range: (u64, u64)`
  (0,0 = none) and `num_positions: u32` (sum of the block's tfs, stored so
  the position stream can be sized/decoded and validated without first
  summing decoded impacts). `COMPRESSED_INDEX_VERSION` 4 → 5:
  - `write_binary`/`read_binary`: v5 appends the positions range +
    `num_positions` per block; `read_binary` keeps reading v4
    (ranges = (0,0), num_positions = 0).
  - Register `(2, migrate_v2_to_v3)`-style step in `manifest::update_index`
    per the established convention (metadata-only rewrite; v4 indices have
    no positions so migration just rewrites the metadata file).

### 2.2b Position compression codec

Positions are the dominant size cost of this feature (typically ≥ the
docids+impacts streams combined), so they get the same pluggable-codec
treatment as the other two streams, not a hardcoded encoding:

- New `PositionsCompressor: Compressor<u32>` (`#[typetag::serde]`, like
  `DocIdCompressor`/`ImpactCompressor`) + `PositionsCompressorFactory`;
  `CompressionTransform` gains `positions_compressor_factory` and
  `CompressedIndexInformation` serializes the chosen codec alongside the
  other two. Codec name joins the manifest `codecs` summary.
- **Encoding unit = the block's flattened delta stream** (Lucene `.pos`
  model): concatenate, over the block's postings in order, each posting's
  position deltas (first position absolute per posting, then gaps). Run
  boundaries are implied by the tfs; `num_positions` gives the total. This
  flattening is what makes block codecs effective — per-posting runs are too
  short to bitpack individually.
- Two codecs in v1:
  - `VBytePositions` — byte-aligned baseline, simplest decode, and the
    reference implementation for tests.
  - `BitPackingPositions` (default) — FOR-bitpacked chunks of 128 deltas via
    the existing `BitPacker4x` dependency (same as the doc-id codec), vbyte
    tail for the remainder. Position gaps within a document are small
    (avg ≈ doclen/tf), so expect ~5–8 bits/position vs vbyte's ≥8 — roughly
    25–40 % smaller, and SIMD block decode is faster than per-byte vbyte on
    the phrase-heavy path.
- The builder-side `positions.dat` (Phase 1) intentionally stays plain
  delta-vbyte: it's a write-once intermediate the transform re-encodes, same
  status as the uncompressed `postings.dat`.

### 2.3 Compressed cursor: lazy positional decode

`src/compress/mod.rs` (block iterator / P4-P5 cursors)

- Positional access decodes the *current block's* position runs on first
  request: mmap `positions.dat`, slice the block's range, walk runs using
  the block's decoded tfs to find the current posting's run. Cache per-block
  decode state so sequential `positions()` calls within a block are O(run).
- Non-positional queries never touch `positions.dat` (never even mmap it):
  zero regression on the existing benchmarks (P1–P5 fast paths untouched —
  positional access goes through the dyn path only in v1).

### 2.4 Other transforms

- `transforms/reorder.rs` (P2 graph bisection): v1 = reordering a positional
  index **preserves positions** by copying each posting's run while
  rewriting postings (positions are docid-independent, so they follow the
  posting). If that turns out nontrivial in the code, fallback v1: reorder
  drops positions with a loud warning + manifest feature removed.
- `transforms/split.rs`: split indices do **not** support positions in v1
  (actionable error), documented.
- BMP conversion ignores positions (by design).

### Phase 2 tests

- Compressed round-trip identity vs forward index positions.
- v4 index loads under v5 code; `update_index` migrates; positional op on a
  v4/non-positional index errors actionably.
- Size report on a real corpus (expect positions ≈ 2–4× docs+freqs; document
  in README).

---

## Phase 3 — Structured queries (doc-level), then positional operators

### 3.1 Cursor traits

`src/index.rs`, `src/search/cursor.rs`

- `BlockTermImpactIterator` gains one defaulted method:
  ```rust
  /// Positions of current()'s document. None if unavailable.
  fn positions(&mut self) -> Option<&[u32]> { None }
  ```
  (Object-safe; `&mut self` is fine — the callers that need it are the
  positional composites, which own their children. TermCursor mirrors it
  with a defaulted method + blanket-impl forwarding.)
- Composite cursors additionally need a cost estimate; `length()` already
  serves (AND: min of children, OR: sum, capped by `max_doc_id`).

### 3.2 Query AST

New `src/query.rs`:

```rust
pub enum QueryNode {
    Term    { term: TermIndex, weight: f32 },
    Combine { children: Vec<(f32, QueryNode)> },        // weighted sum (root-level OR)
    Syn     { children: Vec<QueryNode> },               // OR, tf = sum of children tfs
    Band    { children: Vec<QueryNode> },               // AND, tf = 1
    Phrase  { terms: Vec<TermIndex> },                  // #1 — positional
    Window  { terms: Vec<TermIndex>, width: u32 },      // #uwN — positional
}
```

- Nesting rules mirror matchop: `Syn`/`Band` take any semantic node;
  `Phrase`/`Window` take terms only (v1); `Combine` is the root combinator.
- A small text parser for the matchop subset
  (`#combine:0=2(a #1(new york) #band(b c))`) lives beside it — pure
  convenience, the AST is the API.

### 3.3 Evaluation: composites as virtual terms

New `src/search/ops.rs`:

- `AndCursor`, `OrCursor` over child `Box<dyn BlockTermImpactIterator>`:
  leapfrog / heap-merge via `next_min_doc_id` (the advance-to contract is
  already exactly Lucene's `advance(target)`).
- **Layering (the subtle part)**: positional composites are built from *raw*
  cursors of the inner index (positions live below scoring), produce a
  virtual tf, and are then wrapped by the scoring layer. Concretely:
  - `ScoringModel` gains
    `compound_scorer(&self, dfs: &[u64], max_tf: f32) -> Box<dyn ScoringFunction>`
    (default: sum-of-idf behavior for BM25; models without a sensible
    compound form fall back to `term_scorer(min(dfs), max_tf)`).
  - `ScoredIndex` gains `evaluate(&self, node: &QueryNode) -> Box<dyn
    BlockTermImpactIterator>`: `Term` → existing scored iterator;
    `Phrase`/`Window` → raw positional children → `PhraseCursor` (virtual
    tf) → `ScoringBlockIterator` with the compound scorer; `Band`/`Syn` →
    composites over recursively evaluated children (already-scored, since
    their semantics are score-level per the settled decisions:
    `Syn` composes *tfs* so it too sits below scoring when children are
    plain terms — v1 restriction: `Syn`/`Band` over terms compose tfs below
    scoring; over non-term children they compose scores and are documented
    as such).
  - Raw (unscored) indices evaluate the same tree with tf as the value
    directly — works for quantized-impact experimentation.
- `PhraseCursor`: doc-level AND leapfrog over children; **verification
  inside `next_min_doc_id`** (only stop on docs where the phrase/window
  count > 0), so WAND/MaxScore need zero changes and never see a
  zero-impact posting. Two-phase deferral is explicitly Phase 4.
  Position intersection: standard merge for adjacency (`#1`), min-window
  cover for `#uwN`.
- **Bounds** (pruning safety): `max_value()` = compound scorer's
  `max_score(max_possible_tf)` with `max_possible_tf` = min over children of
  child max tf (safe: phrase tf ≤ every child's tf). `max_block_value()` =
  `max_value()` (trait default — correct, no intra-composite block pruning
  in v1). `max_doc_id` = min over children. `min_dl` from children's
  `min_dl` (max of them is safe for dl-monotone scorers… v1: 0 sentinel,
  always safe).

### 3.4 Search entry + Python

`src/search/mod.rs`, `src/py/mod.rs`

- `search_wand_query(index, &QueryNode, top_k)` / `search_maxscore_query`:
  root `Combine`'s children become the flat cursor list handed to the
  existing `search_wand_core` / `search_maxscore_core` (they're generic over
  `TermCursor`; the dyn blanket impl covers composites). Flat
  `HashMap` queries keep the existing entry points and the P3 monomorphized
  fast path untouched.
- Python: accept a nested structure
  (`{"combine": [[2.0, {"term": 42}], [1.0, {"phrase": [7, 9]}]]}`) or the
  matchop string via the analyzer (`index.search_ops("#combine(a #1(new
  york))", k)` resolving tokens through the vocab). Regenerate
  `python/impact_index.pyi` via `cargo run --bin stub_gen` (stubs are
  auto-generated — never hand-edit).

### Phase 3 tests

- Composite correctness vs brute force on the `TestIndex`/synthetic corpora
  (exhaustive scoring reference for AND/OR/phrase/window).
- Pruning safety: WAND and MaxScore results over structured queries ==
  exhaustive evaluation results (same harness style as the existing
  P3 identity tests).
- Phrase semantics: stopword-gap non-matches; multi-page/multi-block docs.
- Python round-trip incl. matchop parsing.

---

## Phase 4 — Optimizations (only if profiling asks)

**Outcome (measured 2026-09-01, `benches/structured.rs`, 50k docs / 1k
vocab / seeded, top_k 10, max_block_size 128):**

- **Composite block-max bounds: implemented, measured, REVERTED.** Both an
  eager (per-advance) and a lazy (per-consulted-docid, `Cell`-cached)
  variant of honest composite block bounds (`E = min` child block end;
  sum/min of child block maxes per operator; correctness verified by
  domination + pruning-identity tests) showed **no improvement beyond
  noise on any structured query** — at this block size and k, term-level
  bounds already terminate the search before block granularity matters —
  while costing **+17–19 % on bare `#syn`**: with a single top-level
  cursor, WAND/MaxScore consult `max_block_value()` once per candidate
  (1:1 with advances; ~8.7k union candidates for a 3-term mid-df `#syn`),
  so the O(children) bound computation is pure per-candidate overhead and
  laziness has nothing to memoize. Re-attempt only with a design whose
  per-candidate cost is O(1) (e.g. incremental bound maintenance), and
  re-measure with larger top_k / smaller blocks where block bounds have
  headroom. The `test_composite_block_bounds_dominate` invariant test
  remains in `tests/query.rs` as the safety harness for any re-attempt.
- **Two-phase positional verification: deferred on data.** A standalone
  phrase/window query cannot benefit (there are no other clauses to fail
  first — every reported doc must be verified regardless), and the
  targeted shape (`combine_mixed`: cheap terms + `#1` phrase) already
  runs at ~0.75× the flat 4-term query's cost, so verification is not the
  dominating term there. Revisit only if a real workload shows
  phrase-under-combine dominating.
- `benches/structured.rs` (14 benchmarks: flat/phrase/window/band/syn/
  mixed × wand/maxscore, seeded) is the harness for any future attempt.

The items below are kept for reference:

- **Two-phase iteration**: split `PhraseCursor` matching into doc-level
  approximation + `matches()` verification, invoked after the cheap clauses
  in WAND's scoring loop (Lucene's biggest positional win; requires a small
  algorithm-side hook).
- **`max_value_up_to(doc_id)` bound API** replacing block-based bounds for
  composites (Lucene `getMaxScore(upTo)`), enabling block-max pruning
  through operator trees.
- Smarter position codecs behind `PositionsCompressor` (§2.2b) if profiling
  asks (e.g. PFOR with exceptions, Elias-Fano experiments).
- Monomorphized phrase cursor for the compressed+BM25 fast path.

---

## Order of work & risk

| Step | Size | Risk | Depends on |
|---|---|---|---|
| 1.1 analyzer positions | S | low | — |
| 1.2 builder + positions.dat + checkpoint | M | medium (checkpoint compat) | 1.1 |
| 1.3 manifest feature + gating | S | low | 1.2 |
| 1.4 forward read path | S | low | 1.2 |
| 2.1 view plumbing | S | low | 1.4 |
| 2.2 transform + v5 metadata + migration | M | medium (format) | 2.1 |
| 2.3 lazy compressed positions | M | medium | 2.2 |
| 2.4 reorder pass-through | S–M | low | 2.2 |
| 3.1 cursor trait additions | S | low | — |
| 3.2 AST (+ parser) | S | low | — |
| 3.3 composites + compound scoring | L | high (bounds correctness) | 2.3, 3.1, 3.2 |
| 3.4 entry points + Python + stubs | M | low | 3.3 |

Doc-level operators (`Band`/`Syn`/`Combine`, §3.2–3.4 minus phrase) have no
format dependency and can be built in parallel with Phases 1–2 against
existing indices — a natural early deliverable.
