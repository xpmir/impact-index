# Benchmarks

Full methodology, per-system comparisons, and per-setting ablations for
impact-index's BM25 bag-of-words path, plus learned sparse retrieval with
SPLADE-v3 ([below](#learned-sparse-splade-v3-on-ms-marco)). See the
[README](README.md#performance) for the headline numbers.

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

## Structured queries vs Terrier 5

`examples/structured_terrier.py` runs Terrier-matchop queries built from
MS MARCO dev/small through both systems. Each query is reduced to its unique
non-stop words `w1 .. wn`, then:

- `bow`: `#combine(w1 .. wn)`
- `phrase`: bow plus `#1(wi wi+1)` for each adjacent pair
- `uw8`: bow plus `#uw8(wi wi+1)` for each adjacent pair
- `syn`: `#combine(#syn(w1 w2) w3 .. wn)`
- `band`: bow plus `#band(w1 w2)`
- `sdm`: `#combine:0=0.85:1=0.1:2=0.05(#combine(bow) #combine(#1 pairs) #combine(#uw8 pairs))`

impact-index: `pipeline="terrier", stemmer="porter", positions=True` (so no
position gaps, like Terrier), compressed, MaxScore, `BM25Scoring(k1=0.9,
b=0.4, k3=8)`. Terrier 5.11: block index (`blocks=True`), BM25 with the
same k1/b, via PyTerrier. Top-100, single-threaded, Apple M4 Max, 2026-09-18.

| Family | Queries | impact-index q/s | Terrier 5 q/s | Speed-up | Overlap@10 | Overlap@100 |
|--------|--------:|-----------------:|--------------:|---------:|-----------:|------------:|
| bow    | 6,980 | 530 | 35.8 | 15× | 0.982 | 0.986 |
| phrase | 6,598 | 174 | 19.8 |  9× | 0.984 | 0.986 |
| uw8    | 6,598 | 196 | 19.5 | 10× | 0.983 | 0.986 |
| syn    | 6,598 | 383 | 31.2 | 12× | 0.977 | 0.982 |
| band   | 6,598 | 421 | 26.1 | 16× | 0.981 | 0.985 |
| sdm    | 6,598 | 128 | 14.4 |  9× | 0.984 | 0.986 |

- Structured operators add no disagreement of their own: every family's
  overlap is within noise of `bow`. On a small hand-built corpus, all
  operators give scores equal to Terrier's up to a constant factor
  (Terrier uses log2 and keeps BM25's `(k1+1)` factor).
- The remaining ~1.5% comes from bag-of-words: impact-index uses PISA's
  tokenizer, not Terrier's Java one, so some document lengths and tfs
  differ slightly and near-ties reorder.
- Terrier's time includes PyTerrier's per-query Python overhead (a
  DataFrame per query), as in the bag-of-words comparison above.
- Index size: impact-index 0.76 GB (with positions), Terrier 1.6 GB (with
  blocks).

Reproduce with `uv run --with . --with python-terrier
examples/structured_terrier.py --output-dir <dir>` (builds both indices
on first run).

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

## Learned sparse: SPLADE-v3 on MS MARCO

`naver/splade-v3` impacts, MS MARCO passage (8.8M docs), dev/small
(6,980 queries), single-threaded search, Intel Xeon Silver 4214,
2026-09-24/25, at retrieval depths k=10, 100 and 1000 (one run per depth).
Exact = WAND/MaxScore over the raw (float) index. R@k is relevance recall
against the MS MARCO qrels; ExactR@k is the fraction of the exact top-k
that a method retrieves (its agreement with exact search).

### Top-10

| Index / search | ms/query | Index size | MRR@10 | nDCG@10 | R@10 | ExactR@10 |
|----------------|---------:|-----------:|-------:|--------:|-----:|----------:|
| Raw, WAND (exact) | 419 | 17.5 GB | 0.4026 | 0.4697 | 0.6937 | 1 |
| Raw, MaxScore (exact) | 231 | 17.5 GB | 0.4026 | 0.4697 | 0.6937 | 1 |
| Compressed 8-bit, MaxScore | 212 | 3.2 GB | 0.4020 | 0.4694 | 0.6936 | |
| Compressed 16-bit, MaxScore | 215 | 4.6 GB | 0.4026 | 0.4697 | 0.6937 | |
| Split 0.9 + 16-bit, MaxScore | 188 | 4.7 GB | 0.4026 | 0.4697 | 0.6937 | |
| BMP, α=1 β=1 (safe) | 27.3 | 13.8 GB | 0.4021 | 0.4694 | 0.6944 | 0.974 |
| BMP, α=0.95 β=1 | 18.2 | 13.8 GB | 0.4018 | 0.4690 | 0.6934 | 0.967 |
| BMP, α=0.9 β=1 | 14.2 | 13.8 GB | 0.4027 | 0.4702 | 0.6951 | 0.925 |
| BMP, α=0.8 β=1 | 9.4 | 13.8 GB | 0.4036 | 0.4709 | 0.6944 | 0.740 |
| BMP, α=1 β=0.8 | 22.0 | 13.8 GB | 0.4019 | 0.4691 | 0.6926 | 0.970 |
| BMP, α=0.9 β=0.8 | 12.8 | 13.8 GB | 0.4024 | 0.4698 | 0.6940 | 0.916 |
| BMP, α=0.8 β=0.6 | 6.7 | 13.8 GB | 0.4033 | 0.4699 | 0.6896 | 0.697 |
| Seismic, qc=5 hf=0.9 | **0.89** | 9.7 GB | 0.4024 | 0.4693 | 0.6924 | 0.983 |
| Seismic, qc=10 hf=0.8 | 1.28 | 9.7 GB | 0.4026 | 0.4696 | 0.6933 | 0.993 |
| Seismic, qc=30 hf=0.9 | 1.33 | 9.7 GB | 0.4026 | 0.4697 | 0.6937 | 0.993 |
| Seismic, qc=10 hf=0.7 | 1.98 | 9.7 GB | 0.4028 | 0.4699 | 0.6938 | 0.994 |
| Seismic, qc=20 hf=0.7 | 2.59 | 9.7 GB | 0.4027 | 0.4699 | 0.6938 | 0.996 |
| Seismic, qc=20 hf=0.6 | 4.30 | 9.7 GB | 0.4027 | 0.4699 | 0.6938 | 0.997 |
| Seismic n_postings=3500, qc=5 hf=0.9 | 0.68 | 8.1 GB | 0.4002 | 0.4660 | 0.6857 | 0.961 |
| Seismic n_postings=3500, qc=10 hf=0.7 | 1.56 | 8.1 GB | 0.4011 | 0.4673 | 0.6882 | 0.980 |
| Seismic n_postings=3500, qc=20 hf=0.6 | 2.81 | 8.1 GB | 0.4013 | 0.4677 | 0.6893 | 0.985 |

### Top-100 and top-1000

Same settings retrieving 100 and 1000 documents (ms/query grows with the
depth). Exact MaxScore: R@100 0.9242, R@1000 0.9873.

| Index / search | ms (k=100) | R@100 | ExactR@100 | ms (k=1000) | R@1000 | ExactR@1000 |
|----------------|-----------:|------:|-----------:|------------:|-------:|------------:|
| Raw, MaxScore (exact) | 334 | 0.9242 | 1 | 445 | 0.9873 | 1 |
| Split 0.9 + 16-bit, MaxScore | 321 | 0.9242 | | 530 | 0.9873 | |
| BMP, α=1 β=1 (safe) | 106 | 0.9245 | 0.975 | 381 | 0.9872 | 0.973 |
| BMP, α=0.95 β=1 | 76 | 0.9243 | 0.973 | 309 | 0.9872 | 0.972 |
| BMP, α=0.9 β=1 | 55 | 0.9249 | 0.959 | 243 | 0.9872 | 0.965 |
| BMP, α=0.8 β=1 | 28 | 0.9239 | 0.863 | 140 | 0.9871 | 0.920 |
| BMP, α=1 β=0.8 | 91 | 0.9237 | 0.971 | 334 | 0.9872 | 0.970 |
| BMP, α=0.9 β=0.8 | 50 | 0.9238 | 0.954 | 221 | 0.9870 | 0.961 |
| BMP, α=0.8 β=0.6 | 20 | 0.9239 | 0.818 | 103 | 0.9872 | 0.874 |
| Seismic, qc=5 hf=0.9 | **2.6** | 0.9182 | 0.946 | **7.3** | 0.9760 | 0.840 |
| Seismic, qc=10 hf=0.8 | 4.6 | 0.9221 | 0.976 | 8.7 | 0.9823 | 0.905 |
| Seismic, qc=30 hf=0.9 | 4.4 | 0.9227 | 0.980 | 8.5 | 0.9842 | 0.929 |
| Seismic, qc=10 hf=0.7 | 6.2 | 0.9221 | 0.978 | 9.9 | 0.9824 | 0.908 |
| Seismic, qc=20 hf=0.7 | 7.7 | 0.9230 | 0.985 | 12.4 | 0.9842 | 0.935 |
| Seismic, qc=20 hf=0.6 | 11.1 | 0.9233 | 0.986 | 15.1 | 0.9845 | 0.936 |
| Seismic n_postings=3500, qc=5 hf=0.9 | 1.9 | 0.9025 | 0.894 | 4.5 | 0.9537 | 0.736 |
| Seismic n_postings=3500, qc=10 hf=0.7 | 4.1 | 0.9102 | 0.935 | 7.1 | 0.9642 | 0.804 |
| Seismic n_postings=3500, qc=20 hf=0.6 | 6.6 | 0.9122 | 0.949 | 10.1 | 0.9673 | 0.837 |

### Findings

- **Seismic** (default build, `n_postings=6000`; `qc` = `query_cut`,
  `hf` = `heap_factor`) retrieves 98-99.7% of the exact top-10 at
  0.9-4.3 ms/query, 50-260x faster than exact MaxScore. The heap factor
  drives the cost more than the query cut. Build: 412 s, 35 GB peak
  (including the raw index held in RAM).
- **Seismic loses agreement with depth**: 99.5% of the exact top-10 but
  98% of the top-100 and 84-94% of the top-1000 (R@1000 0.976-0.985
  against 0.987). The search parameters barely move the top-1000 ceiling
  (~94%): it comes from the build (posting lists statically pruned to
  `n_postings` entries); the pruned `n_postings=3500` build is worse at
  every depth. It is still 30-60x faster than exact search at k=1000.
- **BMP** (block size 64): even the safe setting (α=β=1) is not exact
  (8-bit score quantization: 97.4% of the exact top-10) and is 30x slower
  than Seismic at similar top-10 agreement. Its agreement holds with depth
  (97% of the exact top-1000, R@1000 equal to exact search), but at k=1000
  the speed-up over exact MaxScore mostly vanishes (381 vs 445 ms). Lower
  α trades agreement for speed quickly (α=0.8: 74% of the top-10).
  Conversion (`to_bmp_streaming`): 387 s, ~33 GB peak, 13.8 GB file; the
  loaded searcher needs ~45 GB of RAM.
- At depth 10, MRR@10 hardly separates the methods (0.401-0.404 even at
  70% agreement with the exact top-10): ExactR@k does.
- Exact search is slow on SPLADE-v3 (long queries): MaxScore is ~2x faster
  than WAND. 8-bit compression costs 0.0006 MRR@10; 16-bit is lossless.
- **Split index** (0.9 quantile, 16-bit, after the split fix): the same
  metrics as exact search at every depth in a quarter of the space; 18%
  faster than raw MaxScore at k=10, as fast at k=100, 19% slower at
  k=1000.
- Encoding (SPLADE-v3, fp16, 2x RTX 2080 Ti) took 8 h.

Reproduce with `examples/splade_benchmark.py --model naver/splade-v3
--fp16 --top-k 10 --output-dir <dir> --compressed-index nbits=8 --seismic ''
--seismic-search 'query-cut=10 heap-factor=0.7' --bmp-search 'alpha=1 beta=1'
...`. The encoded index can be shared between runs with `--index-dir`
(and memory-mapped with `--mmap-index`); every run's results are saved as
TREC files under `<output-dir>/runs/` as soon as it is searched (with a
`.npz` copy that a restarted benchmark reuses).
