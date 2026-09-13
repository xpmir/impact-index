# Benchmarks

Full methodology, per-system comparisons, and per-setting ablations for
impact-index's BM25 bag-of-words path. See the [README](README.md#performance)
for the headline numbers; this document has the detail behind them.

All numbers: BM25 on MS MARCO passage (8.8M docs, 6,980 queries, top-100,
single-threaded). ARM measured 2026-08 on Apple M-series; x86 measured
2026-09 on x86-64/AVX2, all in one session so x86 numbers are directly
comparable to each other.

## Comparison against reference systems

Reference systems disagree on tokenizer/stemmer/stopwords, so one
impact-index build compared against all of them would be apples-to-oranges
for whichever ones it doesn't match. `examples/benchmark.py --suite
comparison` builds impact-index twice instead, each time matching one
family's own defaults, and reports two separate comparisons below.

### Lucene-aligned (Porter stemmer, Lucene's ~33-word stopword list)

Matches Pyserini's own defaults:

| System | ARM q/s | x86 q/s | Index size | MRR@10 |
|--------|---------|---------|-----------|--------|
| **impact-index** (compressed, MaxScore) | **278** | **98.2 ± 1.1** | 0.62 GB | 0.1858 |
| **impact-index** (compressed + reordered, MaxScore) | **295** | **104.5 ± 0.4** | 0.65 GB | 0.1858 |
| impact-index (compressed, WAND/BMW) | — | 71.9 ± 0.1 | 0.62 GB | 0.1858 |
| Pyserini (Lucene) | 213 | 112.0 ± 1.6 | 0.58 GB | 0.1855 |

Result overlap vs Pyserini: @10=0.985, @100=0.989. (Identical for
impact-index's MaxScore and WAND — both are exact top-k algorithms, so
they must agree.)

### Terrier-aligned (Snowball/Porter2 stemmer, Terrier's own ~730-word stopword list)

Matches PISA's and Terrier 5's own defaults:

| System | ARM q/s | x86 q/s | Index size | MRR@10 |
|--------|---------|---------|-----------|--------|
| **impact-index** (compressed, MaxScore) | — | **223.1 ± 1.1** | 0.57 GB | 0.1885 |
| impact-index (compressed, WAND/BMW) | — | 185.5 ± 0.3 | 0.57 GB | 0.1885 |
| Terrier 5 (PyTerrier) | 68 | 26.7 ± 0.1 | 1.31 GB | 0.1881 |
| PISA (Block-Max WAND) | — | 177.4 ± 1.6 | 0.60 GB | 0.1854 |
| PISA (MaxScore) | — | 143.8 ± 0.4 | 0.60 GB | 0.1854 |

- Result overlap vs PISA (Block-Max WAND): @10=0.865, @100=0.894. Terrier 5 itself only reaches @10=0.878 against PISA — its stemmer is classic Porter, not Porter2 (a residual mismatch on top of a pipeline-order difference: PISA and Terrier 5 share the same ~730-word Terrier stopword list, but PISA filters it after stemming against the raw word list, while Terrier 5's Java pipeline filters before stemming — same list, different stage).
- No ARM measurement for this pipeline (Terrier 5/PISA were only benchmarked on x86), nor for impact-index's reordered variant (only measured in the Lucene-aligned config above).

MaxScore is impact-index's headline algorithm in both tables. Its own
WAND/BMW is included for transparency but is markedly slower at this top-k
(see below); PISA is likewise shown with both its algorithms. impact-index's
WAND/BMW throughput was raised 50% on x86 (5% on ARM) by maintaining cursor
order incrementally — a single bubble-down swap per cursor advance instead
of a full `sort_by` — mirroring PISA's own `block_max_wand_query`.

Compressed index is lossless (same results as raw) in both configurations.

Reproduce with `examples/benchmark.py --suite comparison --systems
impact-index,pyserini,terrier,pisa` (add `--with pyserini --with
python-terrier` to the `uv run` invocation, plus a JVM).

Notes:
- **Terrier 5** runs through PyTerrier (single-pass index, one query at a time via `pt.terrier.Retriever`, adding some Python overhead per query), using its default exhaustive DAAT matching — stock Terrier 5.11 has no WAND/block-max pruning, unlike impact-index and Lucene.
- **Java**: measured with whatever JDK was already on the host, OpenJDK 25 — both Terrier 5 (min Java 11+) and Pyserini (min Java 21+) ran fine under it.
- **PISA** runs through [`pyterrier-pisa`](https://github.com/terrierteam/pyterrier_pisa) — no JVM needed, but Linux x86_64 wheels only, so no ARM number. Its 0.60 GB excludes PISA's raw forward/inverted-index files, matching impact-index's own raw-vs-compressed split.
- **impact-index's WAND/BMW trailing MaxScore** is a known effect of top_k=100: WAND pruning needs the top-k threshold θ to rise fast, but at top_k=100 it stays low for a long time — 92% of loop iterations are single-document catch-ups with no pruning benefit (only ~18,500 of ~234,600 per query actually score/reject/skip). Real algorithmic property, not a bug — though PISA's WAND still beating its own MaxScore suggests some implementation headroom beyond that.

## Ablation: impact-index's own settings

Same dataset/queries as above, single-threaded MaxScore, top-100, all builds
compressed (`block_size=128`, `nbits=0`, no reordering). Reproduce with
`examples/benchmark.py --suite ablation`.

Mixing stemmer and stopword list in one table conflates two pipelines that
don't otherwise share a baseline, so each is isolated separately below.

**Porter/Lucene pipeline** (matches Pyserini's own defaults):

| Config | Build (s) | Size (MB) | q/s | MRR@10 |
|--------|-----------|-----------|-----|--------|
| No stopwords | 272 | 717.2 | 76.3 ± 0.1 | 0.1863 |
| + Lucene stopwords (~33 words) | 257 | 639.1 | 98.2 ± 1.1 | 0.1858 |
| + positions (on top of Lucene stopwords) | 368 | 951.7 | 98.5 ± 0.9 | 0.1858 |

**Snowball/Terrier pipeline** (matches PISA's and Terrier 5's own defaults):

| Config | Build (s) | Size (MB) | q/s | MRR@10 |
|--------|-----------|-----------|-----|--------|
| Lucene stopwords (~33 words) | 259 | 639.3 | 98.0 ± 0.7 | 0.1854 |
| Terrier stopwords (~730 words) | 246 | 583.3 | 223.1 ± 1.1 | 0.1885 |

- **Stopwords are the dominant lever** in either pipeline: every stopword occurrence still gets scored and skipped at query time. Dropping them entirely costs +12% index size and -22% throughput vs. Lucene's short list; switching from Lucene's ~33-word list to Terrier's ~730-word one more than doubles throughput.
- **Porter vs. Snowball stemming** makes almost no difference at fixed stopwords (98.2 vs. 98.0 q/s, 0.1858 vs. 0.1854 MRR@10 on the shared row) — the two stemmers only diverge on a minority of inflected forms.
- **`positions=True`** adds ~43% build time and ~49% index size (stored per occurrence, not per posting), with no query-time cost or MRR change unless the query uses a positional operator (`#1`/`#uwN`).

`stemmer=None` isn't included: `BOWIndexBuilder`'s raw-text path requires a
`TextAnalyzer`, and only `"porter"`/`"snowball"` exist today.
