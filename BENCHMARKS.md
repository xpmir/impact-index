# Benchmarks

Full methodology, per-system comparisons, and per-setting ablations for
impact-index's BM25 bag-of-words path. See the [README](README.md#performance)
for the headline numbers.

- BM25 on MS MARCO passage (8.8M docs, 6,980 queries, top-100, single-threaded).
- ARM: Apple M-series, 2026-08. x86: Intel Xeon Silver 4214, 2026-09-17 (one session).

## Comparison against reference systems

`examples/benchmark.py --suite comparison` builds impact-index twice, each
time matching one reference family's own tokenizer/stemmer/stopwords, and
reports two groups below.

### Lucene-aligned — `pipeline="pyserini"`

Porter stemmer, Lucene ~33-word stopword list, pre-stem filtering.

| System | ARM q/s | x86 q/s | Index size | MRR@10 |
|--------|---------|---------|-----------|--------|
| **impact-index** (compressed, MaxScore) | **278** | **96 ± 0** | 0.62 GB | 0.1858 |
| **impact-index** (compressed + reordered, MaxScore) | **295** | **102 ± 0** | 0.65 GB | 0.1859 |
| impact-index (compressed, WAND/BMW) | — | 70 ± 0 | 0.62 GB | 0.1858 |
| Pyserini (Lucene) | 213 | 99 ± 1 | 0.59 GB | 0.1855 |

- Result overlap vs Pyserini: @10=0.985, @100=0.989.
- ARM numbers from the 2026-08 session (not re-measured since; unaffected by anything below).

### Terrier-aligned — `pipeline="terrier"`

Snowball/Porter2 stemmer, Terrier ~730-word stopword list, post-stem
filtering, PISA's own tokenizer. Matches PISA's defaults; close to but not
verified identical to real Terrier 5's (unverified: its own Java
tokenizer). For real Terrier 5's stemmer instead, use
`pipeline="terrier", stemmer="porter"`.

| System | x86 q/s | Index size | MRR@10 |
|--------|---------|-----------|--------|
| **impact-index** (compressed, MaxScore) | **220 ± 1** | 0.51 GB | 0.1883 |
| impact-index (compressed, WAND/BMW) | 183 ± 0 | 0.51 GB | 0.1883 |
| Terrier 5 (PyTerrier) | 26 ± 0 | 1.31 GB | 0.1877 |
| PISA (Block-Max WAND) | 177 ± 1 | 0.60 GB | 0.1854 |
| PISA (MaxScore) | 150 ± 1 | 0.60 GB | 0.1854 |

- Result overlap vs PISA (Block-Max WAND), full 6,980-query set: @10=0.901, @100=0.924.
- Terrier 5 itself: @10=0.878 vs PISA (different stemmer/stopword-filter order).
- No ARM measurement (PISA/Terrier 5 are x86-only); no reordered variant.

MaxScore is impact-index's headline algorithm in both tables; WAND/BMW is
included for transparency but is slower at top_k=100 (θ rises slowly, most
of the loop is unpruned catch-up — algorithmic, not a bug). Compressed
index is lossless in both configurations.

Reproduce with `examples/benchmark.py --suite comparison --systems
impact-index,pyserini,terrier,pisa` (add `--with pyserini --with
python-terrier` to the `uv run` invocation, plus a JVM).

Notes:
- **Terrier 5** via PyTerrier, one query at a time, exhaustive DAAT (no WAND/block-max pruning in stock Terrier 5.11).
- **Java**: OpenJDK 25.
- **PISA** via [`pyterrier-pisa`](https://github.com/terrierteam/pyterrier_pisa), Linux x86_64 only. Size excludes raw forward/inverted files.

## Ablation: impact-index's own settings

Single-threaded MaxScore, top-100, all builds compressed (`block_size=128`,
`nbits=0`, no reordering). Reproduce with `examples/benchmark.py --suite
ablation`.

**Porter/Lucene pipeline** (`pipeline="pyserini"`):

| Config | Build (s) | Size (MB) | q/s | MRR@10 |
|--------|-----------|-----------|-----|--------|
| No stopwords | 278 | 717.1 | 74 ± 0 | 0.1863 |
| + Lucene stopwords (~33 words) | 255 | 639.0 | 96 ± 0 | 0.1858 |
| + positions (on top of Lucene stopwords) | 384 | 951.8 | 95 ± 0 | 0.1858 |

**Snowball/Terrier pipeline** (`pipeline="terrier"` for the second row; first row keeps Lucene's tokenizer/timing to isolate the stemmer):

| Config | Build (s) | Size (MB) | q/s | MRR@10 |
|--------|-----------|-----------|-----|--------|
| Lucene stopwords (~33 words) | 266 | 639.2 | 97 ± 0 | 0.1851 |
| Terrier stopwords (~730 words) | 250 | 517.2 | 220 ± 1 | 0.1883 |

- Stopwords are the dominant lever: Terrier's ~730-word list more than doubles throughput vs. Lucene's ~33-word one.
- Porter vs. Snowball stemming: negligible difference at fixed stopwords.
- `positions=True`: +~50% build time, +~49% index size, no query-time cost.

`stemmer=None` isn't listed: `BOWIndexBuilder`'s raw-text path requires a
`TextAnalyzer`, and only `"porter"`/`"snowball"` exist today.
