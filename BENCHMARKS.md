# Benchmarks

Full methodology, per-system comparisons, and per-setting ablations for
impact-index's BM25 bag-of-words path. See the [README](README.md#performance)
for the headline numbers.

- BM25 on MS MARCO passage (8.8M docs, 6,980 queries, top-100, single-threaded).
- ARM: Apple M-series, 2026-08. x86: Intel Xeon Silver 4214, 2026-09-17 (one session).

## Comparison against reference systems

`examples/benchmark.py --suite comparison` builds impact-index three times,
each time matching one reference family's own tokenizer/stemmer/stopwords,
and reports three groups below.

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

Snowball/Porter2 stemmer, Terrier ~730-word stopword list, pre-stem
filtering (checked against the raw word list, before stemming), PISA's own
tokenizer. Pre-stem filtering matches real Terrier 5's default
`termpipelines=Stopwords,PorterStemmer` order — verified by dumping an
isolated Terrier 5 index's lexicon directly (e.g. "because" is absent
outright, never surviving as stemmed "becaus"). For real Terrier 5's
stemmer instead of Snowball, use `pipeline="terrier", stemmer="porter"`.

| System | x86 q/s | Index size | MRR@10 |
|--------|---------|-----------|--------|
| **impact-index** (compressed, MaxScore) | **230 ± 1** | 0.50 GB | 0.1882 |
| impact-index (compressed, WAND/BMW) | 198 ± 1 | 0.50 GB | 0.1882 |
| Terrier 5 (PyTerrier) | 26 ± 0 | 1.31 GB | 0.1877 |
| PISA (Block-Max WAND) | 215 ± 1 | 0.60 GB | 0.1854 |
| PISA (MaxScore) | 181 ± 1 | 0.60 GB | 0.1854 |

- Result overlap vs **Terrier 5**, full 6,980-query set: @10=0.963, @100=0.966.
- Result overlap vs **PISA** (Block-Max WAND): @10=0.819, @100=0.852 — see
  the PISA-aligned section below for why, and for a pipeline that instead
  gets high agreement with PISA specifically.
- No ARM measurement (PISA/Terrier 5 are x86-only); no reordered variant.

### PISA-aligned — `pipeline="terrier-pisa"`

Same tokenizer/stemmer as `"terrier"` above, but no stop words filtered at
index time. `pyterrier_pisa` (the wrapper this project's PISA numbers are
built with) never removes stop words from PISA's index, at any setting —
"the" alone is indexed in 87% of docs (df=7,714,561/8,841,823 passages).
Architectural, not a bug: PISA's `stops=` only feeds its own native CLI
tool, not this wrapper's `index()` call. `"terrier"` (above) matches real
Terrier 5 instead, which *does* filter at index time — the two pipelines
trade off fidelity to one reference system against the other; no single
build matches both.

Query time still filters Terrier's list, though: PISA's own query
processing *does* exclude stop words (confirmed -- querying it with only
stop words returns no results), just not at index time. Matching that
asymmetry (unfiltered index, filtered query) is what gets high agreement
below — filtering neither side measurably hurt real-query overlap with
PISA (MS MARCO queries are stopword-heavy, so leaving them in let their
small-but-nonzero idf-weighted contributions perturb rankings PISA never
considers), and it slowed search down for no benefit (unfiltered queries
have to traverse "the"'s multi-million-posting list for nothing).

| System | x86 q/s | Index size | MRR@10 |
|--------|---------|-----------|--------|
| **impact-index** (compressed, MaxScore) | **235 ± 2** | 0.64 GB | 0.1866 |
| impact-index (compressed, WAND/BMW) | 194 ± 0 | 0.64 GB | 0.1866 |
| PISA (Block-Max WAND) | 215 ± 1 | 0.60 GB | 0.1854 |
| PISA (MaxScore) | 181 ± 1 | 0.60 GB | 0.1854 |

- Result overlap vs **PISA** (Block-Max WAND), full 6,980-query set:
  @10=0.976, @100=0.979.
- No ARM measurement (PISA is x86-only); no reordered variant.

MaxScore is impact-index's headline algorithm throughout; WAND/BMW is
included for transparency but is slower at top_k=100 (θ rises slowly, most
of the loop is unpruned catch-up — algorithmic, not a bug). Compressed
index is lossless in every configuration.

Reproduce with `examples/benchmark.py --suite comparison --systems
impact-index,pyserini,terrier,pisa` (add `--with pyserini --with
python-terrier` to the `uv run` invocation, plus a JVM), or narrow to one
group with `--only lucene-aligned` / `--only terrier-aligned` / `--only
pisa-aligned`.

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
| Terrier stopwords (~730 words) | 234 | 509.1 | 230 ± 1 | 0.1882 |

- Stopwords are the dominant lever: Terrier's ~730-word list more than doubles throughput vs. Lucene's ~33-word one.
- Porter vs. Snowball stemming: negligible difference at fixed stopwords.
- `positions=True`: +~50% build time, +~49% index size, no query-time cost.

`stemmer=None` isn't listed: `BOWIndexBuilder`'s raw-text path requires a
`TextAnalyzer`, and only `"porter"`/`"snowball"` exist today.
