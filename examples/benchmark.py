#!/usr/bin/env python3
# /// script
# requires-python = ">=3.9"
# dependencies = [
#     "ir-datasets",
#     "tqdm",
#     "numpy",
#     "cbor2",
#     "snowballstemmer",
#     "pyterrier-pisa ; sys_platform == 'linux' and platform_machine == 'x86_64'",
# ]
# ///
"""
Unified BM25 benchmark harness: impact-index's own settings (ablation) and
impact-index vs Pyserini/Terrier 5/PISA (comparison) on MS MARCO passage.

Replaces the old examples/benchmark_bm25.py and examples/benchmark_ablation.py
with one script that shares build/measure/MRR logic between both kinds of run
and adds two-level, content-addressed caching so repeated invocations don't
silently re-measure a stale or mismatched build:

  - Index-level cache: a built index is identified by a hash of the
    parameters that actually change its on-disk bytes (system, dataset,
    stemmer, stop words, positions, compression settings, reordering). Each
    cache directory carries a `bench_manifest.json` recording the resolved
    parameters, build time, and host/timestamp, so a human can audit what is
    actually sitting on disk. Comparing against the config's hash (rather
    than a human-chosen directory name) means a cache hit is a guarantee, not
    a hope. (Named `bench_manifest.json`, not `manifest.json`, to avoid
    colliding with impact-index's own per-index format-version manifest.)
  - Search-level cache: search results are cached separately (JSON files
    under `<output-dir>/results/`), keyed by the index's cache key plus
    search-affecting parameters (top_k, k1, b, algorithm, query set). Each
    entry accumulates per-repeat q/s samples plus MRR@10, timestamp and host,
    so `--render` can turn accumulated repeats into mean +/- std tables
    without re-running search.

Two independently selectable suites (`--suite`):
  - `ablation`: sweeps impact-index's own settings (stemmer, stop-word
    family, positions) against each other. Only needs impact_index +
    ir_datasets: no Java, no PISA, no Pyserini.
  - `comparison`: impact-index vs Pyserini, Terrier 5 (via PyTerrier), and
    PISA (via pyterrier-pisa, Linux x86_64 only). Stopwords and stemming
    matter a lot on MS MARCO (its queries are dominated by "what is X" /
    "how to Y" patterns, and Lucene's default stopword list doesn't filter
    "what/how/does/do" while Terrier's does), so impact-index is built
    TWICE, once per engine family it's compared against, and reported as two
    separate groups instead of one table that would silently compare configs
    with different amounts of query work:
      - lucene-aligned:  impact-index[Lucene]  (Porter + Lucene stopwords)  vs Pyserini
      - terrier-aligned: impact-index[Terrier] (Snowball/Porter2 + Terrier stopwords)
                         vs Terrier 5 and PISA

Configs for both suites are plain Python dataclasses in CONFIGS below (not an
external JSON file) -- easiest to read, diff, and comment inline. What's
actually a JSON file is everything downstream of a config: the index
manifests and the search-result cache described above.

Dependencies: Pyserini and python-terrier are NOT in this script's PEP 723
header (only ir-datasets/tqdm/numpy/cbor2/snowballstemmer, plus
pyterrier-pisa's existing Linux x86_64 marker) because most runs (`--systems
impact-index`, i.e. the ablation suite or an impact-index-only comparison
run) need neither a JVM nor those packages, and their import cost/footprint
shouldn't be paid unconditionally. Their imports are lazy/local inside the
specific functions that need them, gated by `--systems`, so an
`impact-index`-only run never touches them. To actually compare against
Pyserini/Terrier 5, add them at invocation time and provide a JVM:

    uv run --with . --with pyserini --with python-terrier \\
        examples/benchmark.py --suite comparison --systems impact-index,pyserini,terrier,pisa

Java requirements when pyserini/python-terrier are added: Terrier 5 (via
PyTerrier) needs Java 11+, Pyserini needs Java 21 -- a single OpenJDK 21
install (e.g. Temurin) satisfies both. Set JAVA_HOME if not picked up
automatically. PISA is pure C++/pybind11 and needs no JVM at all, but ships
wheels for Linux x86_64 only -- pass `--systems ...,pisa` on any other
platform and it will simply be skipped with a warning (not an error).

Usage:
    # Ablation only (impact-index settings against each other; no Java, no PISA)
    uv run --with . examples/benchmark.py --suite ablation --output-dir /tmp/bench

    # Full comparison, 10 search repeats per config
    uv run --with . --with pyserini --with python-terrier \\
        examples/benchmark.py --suite comparison \\
        --systems impact-index,pyserini,terrier,pisa \\
        --repeats 10 --output-dir /tmp/bench

    # Re-run search only with more repeats (indices already built and cached)
    uv run --with . examples/benchmark.py --suite ablation --repeats 20 --output-dir /tmp/bench

    # Force fresh search measurements without rebuilding indices
    uv run --with . examples/benchmark.py --suite ablation --reset-search-statistics --output-dir /tmp/bench

    # Force a full rebuild (index + search) for the selected configs
    uv run --with . examples/benchmark.py --suite ablation --reset-index --output-dir /tmp/bench

    # Run (or re-render) just one config by (sub)name
    uv run --with . examples/benchmark.py --suite ablation --only terrier-stopwords --output-dir /tmp/bench

    # Turn accumulated JSON results into the markdown tables used in
    # README.md / BENCHMARKS.md (reads --output-dir/results/, writes nothing,
    # prints markdown to stdout for manual review before pasting in)
    uv run --with . examples/benchmark.py --render all --output-dir /tmp/bench

Iterating on the Rust source: `uv run --with .` builds impact-index as a
local path dependency, and its change detection can be unreliable --
notably, if the working tree isn't a git repository (e.g. a scratch copy
made with `rsync` for remote benchmarking), source edits can be silently
ignored in favor of a stale cached build, with no warning. If numbers
aren't moving after a source change, force a rebuild:
    uv run --reinstall-package impact-index --with . examples/benchmark.py ...

Note on `stop_words="lucene"`/`"terrier"`: the comparison-suite configs below
assume `BOWIndexBuilder(stop_words=...)` accepts these two string presets
(in addition to `True`/a list/`None`). At the time this script was written
that preset feature lives in a separate, not-yet-merged branch -- if you see
`stop_words must be True, a list of strings, or None`, you have an
impact-index build predating that feature; merge it first.
"""

import argparse
import gc
import hashlib
import json
import os
import platform
import re
import shutil
import socket
import subprocess
import sys
import time
from dataclasses import dataclass, replace
from functools import partial
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import ir_datasets
from tqdm import tqdm

import impact_index
from impact_index import (
    BitPackingCompressor,
    CompressionTransform,
    GlobalImpactQuantizer,
    SplitIndexTransform,
)

ALL_SYSTEMS = ("impact-index", "pyserini", "terrier", "pisa")


# --------------------------------------------------------------------------
# Config
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class Compression:
    """impact-index compression settings. `None` on a Config means "raw/uncompressed"."""

    nbits: int = 0  # 0 = lossless integer bitpacking (best for BM25 term frequencies)
    block_size: int = 128
    split_quantiles: Optional[Tuple[float, ...]] = None
    reorder: bool = False  # recursive-graph-bisection reorder, fused with compression

    def cache_fields(self) -> dict:
        return {
            "nbits": self.nbits,
            "block_size": self.block_size,
            "split_quantiles": list(self.split_quantiles)
            if self.split_quantiles
            else None,
            "reorder": self.reorder,
        }

    def label(self) -> str:
        prefix = (
            "Split(" + ",".join(f"{q:.2f}" for q in self.split_quantiles) + ") "
            if self.split_quantiles
            else ""
        )
        suffix = " + reordered" if self.reorder else ""
        return f"{prefix}Compressed nb={self.nbits} bs={self.block_size}{suffix}"

    def dir_tag(self) -> str:
        parts = []
        if self.split_quantiles:
            parts.append(
                "split_"
                + "_".join(f"{q:.2f}".replace(".", "") for q in self.split_quantiles)
            )
        parts.append(f"nb{self.nbits}_bs{self.block_size}")
        if self.reorder:
            parts.append("reorder")
        return "_".join(parts)


@dataclass(frozen=True)
class Config:
    """One benchmarked config: a build recipe plus the search parameters to measure it with."""

    name: str
    system: str  # "impact-index" | "pyserini" | "terrier" | "pisa"
    group: (
        str  # which report table this feeds, e.g. "ablation-lucene", "lucene-aligned"
    )
    dataset: str = "msmarco-passage/dev/small"
    # impact-index-only build knobs (ignored by other systems, which use their own defaults):
    # `pipeline` is the only place another system's tokenizer/stop-word-filter
    # timing is selected -- "pyserini" (Lucene's tokenizer, pre-stem stop
    # words) or "terrier" (PISA's own tokenizer -- which truncates every
    # contraction/possessive at its first apostrophe, not just a trailing
    # 's -- post-stem stop words). `stemmer`/`stop_words` below compose on
    # top of it, overriding just that piece of the pipeline's defaults.
    pipeline: Optional[str] = None  # "pyserini" | "terrier" | None
    stemmer: str = "porter"  # "porter" | "snowball"
    stop_words: Any = "lucene"  # "lucene" | "terrier" | True | None
    positions: bool = False
    compression: Optional[Compression] = None  # None = raw/uncompressed index
    # search params:
    algorithm: str = "maxscore"  # impact-index: maxscore|wand ; pisa: block_max_wand|block_max_maxscore
    top_k: int = 100
    k1: float = 0.9
    b: float = 0.4
    # impact-index-only: BM25 idf variant -- "bm25" (Robertson/Sparck-Jones,
    # no +1 -- matches PISA and Terrier) or "lucene" (Lucene's
    # BM25Similarity.idf, +1 -- matches Pyserini). Ignored by other systems,
    # which use their own (fixed) idf formula.
    bm25_variant: str = "bm25"
    max_queries: int = 0

    def display(self) -> str:
        return self.name


TERRIER_STOPWORDS = (
    "a abaft abafter abaftest about abouter aboutest above abover abovest accordingly aer aest afore after afterer afterest afterward afterwards again against aid ain albeit all aller allest alls allyou almost along alongside already also although always amid amidst among amongst an and andor anear anent another any anybody anyhow anyone anything anywhere apart aparter apartest appear appeared appearing appears appropriate appropriated appropriater appropriates appropriatest appropriating are ares around as ases aside asides aslant astraddle astraddler astraddlest astride astrider astridest at athwart atop atween aught aughts available availabler availablest awfully b be became because become becomes becoming becominger becomingest becomings been before beforehand beforehander beforehandest behind behinds below beneath beside besides better bettered bettering betters between betwixt beyond bist both but buts by by-and-by byandby c cannot canst cant canted cantest canting cants cer certain certainer certainest cest chez circa co come-on come-ons comeon comeons concerning concerninger concerningest consequently considering could couldst cum d dday ddays describe described describes describing despite despited despites despiting did different differenter differentest do doe does doing doings done doner dones donest dos dost doth downs downward downwarder downwardest downwards during e each eg eight either else elsewhere enough ere et etc even evened evenest evens evenser evensest ever every everybody everyone everything everywhere ex except excepted excepting excepts exes f fact facts failing failings few fewer fewest figupon figuponed figuponing figupons five followthrough for forby forbye fore forer fores forever former formerer formerest formerly formers fornenst forwhy four fourscore frae from fs further furthered furtherer furtherest furthering furthermore furthers g get gets getting go gone good got gotta gotten h had hadst hae hardly has hast hath have haves having he hence her hereafter hereafters hereby herein hereupon hers herself him himself his hither hitherer hitherest hoo hoos how how-do-you-do howbeit howdoyoudo however huh humph i idem idemer idemest ie if ifs immediate immediately immediater immediatest in inasmuch inc indeed indicate indicated indicates indicating info information insofar instead into inward inwarder inwardest inwards is it its itself j k l latter latterer latterest latterly latters layabout layabouts less lest lot lots lotted lotting m main make many mauger maugre mayest me meanwhile meanwhiles midst midsts might mights more moreover most mostly much mucher muchest must musth musths musts my myself n natheless nathless neath neaths necessarier necessariest necessary neither nethe nethermost never nevertheless nigh nigher nighest nine no no-one nobodies nobody noes none noone nor nos not nothing nothings notwithstanding nowhere nowheres o of off offest offs often oftener oftenest oh on one oneself onest ons onto or orer orest other others otherwise otherwiser otherwisest ought oughts our ours ourself ourselves out outed outest outs outside outwith over overall overaller overallest overalls overs own owned owning owns owt p particular particularer particularest particularly particulars per perhaps plaintiff please pleased pleases plenties plenty pro probably provide provided provides providing q qua que quite r rath rathe rather rathest re really regarding relate related relatively res respecting respectively s said saider saidest same samer sames samest sans sanserif sanserifs sanses saved sayid sayyid seem seemed seeminger seemingest seemings seems send sent senza serious seriouser seriousest seven several severaler severalest shall shalled shalling shalls she should shoulded shoulding shoulds since sine sines sith six so sobeit soer soest some somebody somehow someone something sometime sometimer sometimes sometimest somewhat somewhere stop stopped such summat sup supped supping sups syn syne t ten than that the thee their theirs them themselves then thence thener thenest there thereafter thereby therefore therein therer therest thereupon these they thine thing things this thises thorough thorougher thoroughest thoroughly those thou though thous thouses three thro through througher throughest throughout thru thruer thruest thus thy thyself till tilled tilling tills to together too toward towarder towardest towards two u umpteen under underneath unless unlike unliker unlikest until unto up upon uponed uponing upons upped upping ups us use used usedest username usually v various variouser variousest verier veriest versus very via vis-a-vis vis-a-viser vis-a-visest viz vs w was wast we were wert what whatever whateverer whateverest whatsoever whatsoeverer whatsoeverest wheen when whenas whence whencesoever whenever whensoever where whereafter whereas whereby wherefrom wherein whereinto whereof whereon wheresoever whereto whereupon wherever wherewith wherewithal whether which whichever whichsoever while whiles whilst whither whithersoever whoever whomever whose whoso whosoever why with withal within without would woulded woulding woulds x y ye yet yon yond yonder you your yours yourself yourselves z zillion"
).split()
# Kept here (not just relied on via the stop_words="terrier" preset) because
# search_pisa/search_terrier never build through BOWIndexBuilder and don't
# need it; this constant only documents what "terrier stopwords" means for
# readers of this file, it is not otherwise used below.


ABLATION_CONFIGS: List[Config] = [
    Config(
        "no-stopwords",
        "impact-index",
        group="ablation-porter",
        pipeline="pyserini",
        stemmer="porter",
        # [] not None: with `pipeline` set, `stop_words=None` means "use
        # the pipeline's own default list" (Python can't distinguish an
        # omitted argument from an explicit None) -- see BOWIndexBuilder's
        # docstring. [] is the explicit "no stop words at all" signal.
        stop_words=[],
        bm25_variant="lucene",
    ),
    Config(
        "lucene-stopwords",
        "impact-index",
        group="ablation-porter",
        pipeline="pyserini",
        stemmer="porter",
        stop_words="lucene",
        bm25_variant="lucene",
    ),
    Config(
        "lucene-stopwords_positions",
        "impact-index",
        group="ablation-porter",
        pipeline="pyserini",
        stemmer="porter",
        stop_words="lucene",
        positions=True,
        bm25_variant="lucene",
    ),
    Config(
        # Deliberately NOT pipeline="terrier": this isolates the stemmer
        # variable (snowball vs porter) at fixed Lucene stopwords/tokenizer,
        # so it must keep the Lucene tokenizer/filter timing, not switch to
        # PISA's -- see the "Porter vs. Snowball stemming" ablation note.
        "snowball_lucene-stopwords",
        "impact-index",
        group="ablation-snowball",
        stemmer="snowball",
        stop_words="lucene",
    ),
    Config(
        "snowball_terrier-stopwords",
        "impact-index",
        group="ablation-snowball",
        pipeline="terrier",
        stemmer="snowball",
        stop_words="terrier",
    ),
]
# Every ablation config is measured compressed (block_size=128, nbits=0, no
# reordering, no split) -- that's the shape a user would actually deploy;
# raw/uncompressed size and MaxScore-only search are what's reported.
ABLATION_CONFIGS = [
    replace(c, compression=Compression(nbits=0, block_size=128))
    for c in ABLATION_CONFIGS
]


COMPARISON_CONFIGS: List[Config] = [
    # --- Lucene-aligned: impact-index[Lucene] vs Pyserini ---
    Config(
        "impact-index (compressed, MaxScore)",
        "impact-index",
        group="lucene-aligned",
        pipeline="pyserini",
        stemmer="porter",
        stop_words="lucene",
        compression=Compression(nbits=0, block_size=128),
        algorithm="maxscore",
        bm25_variant="lucene",
    ),
    Config(
        "impact-index (compressed + reordered, MaxScore)",
        "impact-index",
        group="lucene-aligned",
        pipeline="pyserini",
        stemmer="porter",
        stop_words="lucene",
        compression=Compression(nbits=0, block_size=128, reorder=True),
        algorithm="maxscore",
        bm25_variant="lucene",
    ),
    Config(
        "impact-index (compressed, WAND/BMW)",
        "impact-index",
        group="lucene-aligned",
        pipeline="pyserini",
        stemmer="porter",
        stop_words="lucene",
        compression=Compression(nbits=0, block_size=128),
        algorithm="wand",
        bm25_variant="lucene",
    ),
    Config("Pyserini (Lucene)", "pyserini", group="lucene-aligned"),
    # --- Terrier-aligned: impact-index[Terrier] vs Terrier 5 and PISA ---
    Config(
        "impact-index (compressed, MaxScore)",
        "impact-index",
        group="terrier-aligned",
        pipeline="terrier",
        stemmer="snowball",
        stop_words="terrier",
        compression=Compression(nbits=0, block_size=128),
        algorithm="maxscore",
    ),
    Config(
        "impact-index (compressed, WAND/BMW)",
        "impact-index",
        group="terrier-aligned",
        pipeline="terrier",
        stemmer="snowball",
        stop_words="terrier",
        compression=Compression(nbits=0, block_size=128),
        algorithm="wand",
    ),
    Config("Terrier 5 (PyTerrier)", "terrier", group="terrier-aligned"),
    Config(
        "PISA (Block-Max WAND)",
        "pisa",
        group="terrier-aligned",
        algorithm="block_max_wand",
    ),
    Config(
        "PISA (MaxScore)",
        "pisa",
        group="terrier-aligned",
        algorithm="block_max_maxscore",
    ),
]

SUITES = {"ablation": ABLATION_CONFIGS, "comparison": COMPARISON_CONFIGS}


# --------------------------------------------------------------------------
# Cache keys, manifests
# --------------------------------------------------------------------------


def _hash(fields: dict) -> str:
    payload = json.dumps(fields, sort_keys=True, default=str)
    return hashlib.sha256(payload.encode()).hexdigest()[:16]


def index_cache_fields(cfg: Config) -> dict:
    """Parameters that change an index's on-disk bytes -- the index cache key."""
    if cfg.system == "impact-index":
        return {
            "system": cfg.system,
            "dataset": cfg.dataset,
            "pipeline": cfg.pipeline,
            "stemmer": cfg.stemmer,
            "stop_words": cfg.stop_words,
            "positions": cfg.positions,
            "compression": cfg.compression.cache_fields() if cfg.compression else None,
        }
    # Reference systems build with their own fixed defaults; only the corpus matters.
    return {"system": cfg.system, "dataset": cfg.dataset}


def index_cache_key(cfg: Config) -> str:
    return _hash(index_cache_fields(cfg))


def search_cache_fields(cfg: Config, index_key: str) -> dict:
    """Parameters that change search results/timing on top of a fixed index."""
    return {
        "index_key": index_key,
        "algorithm": cfg.algorithm,
        "top_k": cfg.top_k,
        "k1": cfg.k1,
        "b": cfg.b,
        "bm25_variant": cfg.bm25_variant,
        "max_queries": cfg.max_queries,
        "dataset": cfg.dataset,
    }


def search_cache_key(cfg: Config, index_key: str) -> str:
    return _hash(search_cache_fields(cfg, index_key))


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def write_json(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w") as f:
        json.dump(data, f, indent=2, sort_keys=True)
    tmp.replace(path)


def read_json(path: Path) -> Optional[dict]:
    if not path.exists():
        return None
    with open(path) as f:
        return json.load(f)


def dir_size_mb(path: Path, exclude_prefixes: Tuple[str, ...] = ()) -> float:
    total = 0
    for f in path.rglob("*"):
        if f.is_file() and not f.name.startswith(exclude_prefixes):
            total += f.stat().st_size
    return total / 1024 / 1024


# --------------------------------------------------------------------------
# MRR / qrels (shared by both suites)
# --------------------------------------------------------------------------


def load_qrels(dataset) -> Dict[str, set]:
    qrels: Dict[str, set] = {}
    for qrel in dataset.qrels_iter():
        if qrel.relevance > 0:
            qrels.setdefault(qrel.query_id, set()).add(str(qrel.doc_id))
    return qrels


def compute_mrr(results: Dict[str, list], qrels: Dict[str, set], k: int = 10) -> float:
    rr_sum = 0.0
    n_queries = 0
    for qid, hits in results.items():
        if qid not in qrels:
            continue
        n_queries += 1
        relevant = qrels[qid]
        for rank, (docid, _score) in enumerate(hits[:k], 1):
            if str(docid) in relevant:
                rr_sum += 1.0 / rank
                break
    return rr_sum / n_queries if n_queries > 0 else 0.0


# --------------------------------------------------------------------------
# impact-index: build/load (recursive: compressed/reordered index is built
# from a cached raw index, which is itself cached under its own key)
# --------------------------------------------------------------------------


def _write_manifest(
    index_dir: Path, cfg: Config, index_key: str, build_seconds: Optional[float]
) -> None:
    write_json(
        index_dir / "bench_manifest.json",
        {
            "index_key": index_key,
            "fields": index_cache_fields(cfg),
            "config_name": cfg.name,
            "build_seconds": build_seconds,
            "built_at": now_iso(),
            "host": socket.gethostname(),
            "impact_index_version": getattr(impact_index, "__version__", None),
        },
    )


def build_or_load_impact_index(
    cfg: Config, indices_root: Path, dataset, force_rebuild: bool
):
    """Returns (index_object, index_key, index_dir, built_now: bool)."""
    index_key = index_cache_key(cfg)
    index_dir = indices_root / "impact-index" / index_key
    manifest_path = index_dir / "bench_manifest.json"

    if not force_rebuild and manifest_path.exists():
        index = impact_index.Index.load(str(index_dir), in_memory=True)
        return index, index_key, index_dir, False

    build_seconds = None
    if cfg.compression is None:
        # Raw index: build directly from the corpus. Wipe any stale directory
        # first -- a leftover file set from a different config or an
        # interrupted build (mismatched file versions) can otherwise be
        # silently mixed into the new build instead of causing a clean error.
        shutil.rmtree(index_dir, ignore_errors=True)
        index_dir.mkdir(parents=True, exist_ok=True)
        builder = impact_index.BOWIndexBuilder(
            str(index_dir),
            dtype="int32",
            pipeline=cfg.pipeline,
            stemmer=cfg.stemmer,
            stop_words=cfg.stop_words,
            positions=cfg.positions,
        )
        batch = []
        BATCH_SIZE = 10000
        start = time.perf_counter()
        for doc in tqdm(dataset.docs_iter(), desc=f"Indexing [{cfg.name}]"):
            text = doc.text if hasattr(doc, "text") else str(doc)
            doc_id = int(doc.doc_id) if doc.doc_id.isdigit() else hash(doc.doc_id)
            batch.append((doc_id, text))
            if len(batch) >= BATCH_SIZE:
                builder.add_texts(batch)
                batch = []
        if batch:
            builder.add_texts(batch)
        index = builder.build(in_memory=True)
        build_seconds = time.perf_counter() - start
    else:
        # Compressed/reordered index: get the raw source first (recursively
        # cached under its own key), then transform it into this cfg's dir.
        raw_cfg = replace(cfg, compression=None)
        raw_index, _, _, _ = build_or_load_impact_index(
            raw_cfg, indices_root, dataset, force_rebuild=force_rebuild
        )

        shutil.rmtree(index_dir, ignore_errors=True)
        index_dir.mkdir(parents=True, exist_ok=True)
        comp = cfg.compression
        start = time.perf_counter()
        if comp.split_quantiles:
            doc_ids_compressor = BitPackingCompressor()
            impact_compressor = (
                impact_index.BitPackedIntCompressor()
                if comp.nbits == 0
                else GlobalImpactQuantizer(comp.nbits)
            )
            transform = SplitIndexTransform(
                list(comp.split_quantiles),
                CompressionTransform(
                    comp.block_size, doc_ids_compressor, impact_compressor
                ),
            )
            transform.process(str(index_dir), raw_index)
            index = impact_index.Index.load(str(index_dir), in_memory=True)
        elif comp.reorder:
            index = raw_index.reorder(
                str(index_dir), block_size=comp.block_size, nbits=comp.nbits
            )
        else:
            index = raw_index.compress(
                str(index_dir), block_size=comp.block_size, nbits=comp.nbits
            )
        build_seconds = time.perf_counter() - start

    _write_manifest(index_dir, cfg, index_key, build_seconds)
    return index, index_key, index_dir, True


class ImpactIndexHandle:
    """Keeps one loaded impact-index resident in memory across all repeats/configs that share it."""

    def __init__(self, raw_index):
        self.raw_index = raw_index
        self._analyzer = raw_index.analyzer()
        self._scored_cache: Dict[Tuple[float, float, str], Any] = {}

    def _scored(self, k1: float, b: float, variant: str):
        key = (k1, b, variant)
        if key not in self._scored_cache:
            scoring = impact_index.BM25Scoring(k1=k1, b=b, variant=variant)
            self._scored_cache[key] = self.raw_index.with_scoring(scoring)
        return self._scored_cache[key]

    def search_once(self, queries, cfg: Config):
        scored = self._scored(cfg.k1, cfg.b, cfg.bm25_variant)
        search_fn = (
            scored.search_wand if cfg.algorithm == "wand" else scored.search_maxscore
        )
        results = {}
        start = time.perf_counter()
        for qid, text in queries:
            query = self._analyzer.analyze_query(text)
            results[qid] = (
                []
                if not query
                else [(h.docid, h.score) for h in search_fn(query, cfg.top_k)]
            )
        elapsed = time.perf_counter() - start
        return results, elapsed


# --------------------------------------------------------------------------
# Pyserini (lazy import; only reached when "pyserini" is in --systems)
# --------------------------------------------------------------------------


def build_or_load_pyserini_index(
    cfg: Config, indices_root: Path, dataset, force_rebuild: bool
):
    index_key = index_cache_key(cfg)
    index_dir = indices_root / "pyserini" / index_key
    lucene_dir = index_dir / "lucene"
    manifest_path = index_dir / "bench_manifest.json"

    if not force_rebuild and manifest_path.exists():
        return lucene_dir, index_key, index_dir, False

    shutil.rmtree(index_dir, ignore_errors=True)
    jsonl_dir = index_dir / "collection"
    jsonl_dir.mkdir(parents=True, exist_ok=True)
    jsonl_path = jsonl_dir / "docs.jsonl"
    with open(jsonl_path, "w") as f:
        for doc in tqdm(dataset.docs_iter(), desc="Writing JSONL (Pyserini)"):
            text = doc.text if hasattr(doc, "text") else str(doc)
            json.dump({"id": doc.doc_id, "contents": text}, f)
            f.write("\n")

    start = time.perf_counter()
    subprocess.run(
        [
            "python",
            "-m",
            "pyserini.index.lucene",
            "--collection",
            "JsonCollection",
            "--input",
            str(jsonl_dir),
            "--index",
            str(lucene_dir),
            "--generator",
            "DefaultLuceneDocumentGenerator",
            "--threads",
            "4",
        ],
        check=True,
    )
    build_seconds = time.perf_counter() - start
    _write_manifest(index_dir, cfg, index_key, build_seconds)
    return lucene_dir, index_key, index_dir, True


class PyseriniHandle:
    def __init__(self, lucene_dir: Path):
        os.environ.setdefault(
            "OPENAI_API_KEY", "unused"
        )  # pyserini.encode instantiates an OpenAI client at import time
        from pyserini.search.lucene import LuceneSearcher

        self.searcher = LuceneSearcher(str(lucene_dir))
        self._bm25_set: Optional[Tuple[float, float]] = None

    def search_once(self, queries, cfg: Config):
        if self._bm25_set != (cfg.k1, cfg.b):
            self.searcher.set_bm25(cfg.k1, cfg.b)
            self._bm25_set = (cfg.k1, cfg.b)
        results = {}
        start = time.perf_counter()
        for qid, text in queries:
            hits = self.searcher.search(text, k=cfg.top_k)
            results[qid] = [(hit.docid, hit.score) for hit in hits]
        elapsed = time.perf_counter() - start
        return results, elapsed


# --------------------------------------------------------------------------
# PISA (lazy import; only reached when "pisa" is in --systems)
# --------------------------------------------------------------------------


def pisa_available() -> bool:
    return platform.system() == "Linux" and platform.machine() in ("x86_64", "AMD64")


def build_or_load_pisa_index(
    cfg: Config, indices_root: Path, dataset, force_rebuild: bool
):
    from pyterrier_pisa import PisaIndex

    index_key = index_cache_key(cfg)
    index_dir = indices_root / "pisa" / index_key
    manifest_path = index_dir / "bench_manifest.json"

    if not force_rebuild and manifest_path.exists():
        index = PisaIndex(
            str(index_dir), stemmer="porter2", threads=os.cpu_count() or 4
        )
        return index, index_key, index_dir, False

    shutil.rmtree(index_dir, ignore_errors=True)
    index_dir.mkdir(parents=True, exist_ok=True)
    index = PisaIndex(
        str(index_dir),
        stemmer="porter2",  # Snowball Porter2 -- PISA's own analysis, not Lucene-matched
        threads=os.cpu_count() or 4,
        index_encoding="block_simdbp",
    )

    def doc_iter():
        for doc in tqdm(dataset.docs_iter(), desc="Indexing (PISA)"):
            text = doc.text if hasattr(doc, "text") else str(doc)
            yield {"docno": doc.doc_id, "text": text}

    start = time.perf_counter()
    index.index(doc_iter())
    build_seconds = time.perf_counter() - start
    _write_manifest(index_dir, cfg, index_key, build_seconds)
    return index, index_key, index_dir, True


def pisa_index_size_mb(pisa_dir: Path) -> float:
    """Excludes PISA's raw forward/uncompressed-inverted index files, kept on disk
    alongside the final index -- same convention as impact-index's raw-vs-compressed split."""
    return dir_size_mb(pisa_dir, exclude_prefixes=("fwd", "inv."))


class PisaHandle:
    def __init__(self, index):
        self.index = index
        self._bm25_cache: Dict[Tuple[float, float, int, str], Any] = {}

    def _bm25(self, cfg: Config):
        key = (cfg.k1, cfg.b, cfg.top_k, cfg.algorithm)
        if key not in self._bm25_cache:
            # threads=1: PisaIndex.bm25() otherwise inherits the *indexing* thread
            # count as its retrieval thread count too, which is pure overhead here
            # since queries are searched one at a time.
            self._bm25_cache[key] = self.index.bm25(
                k1=cfg.k1,
                b=cfg.b,
                num_results=cfg.top_k,
                query_algorithm=cfg.algorithm,
                threads=1,
            )
        return self._bm25_cache[key]

    def search_once(self, queries, cfg: Config):
        bm25 = self._bm25(cfg)
        results = {}
        start = time.perf_counter()
        for qid, text in queries:
            hits = bm25.search(text)
            results[qid] = [
                (str(row.docno), float(row.score)) for row in hits.itertuples()
            ]
        elapsed = time.perf_counter() - start
        return results, elapsed


# --------------------------------------------------------------------------
# Terrier 5 (via PyTerrier). PyTerrier and Pyserini cannot share a process
# (both use pyjnius, and the JVM classpath is fixed by whichever starts it
# first), so everything touching Terrier runs in a worker subprocess with
# its own JVM. The worker loads the index ONCE and loops all repeats
# in-process (see module docstring / handoff note: reloading a multi-GB
# index per repeat would dominate the measurement).
# --------------------------------------------------------------------------

SCRIPT_PATH = str(Path(__file__).resolve())


def _run_terrier_worker(spec: dict, work_dir: Path) -> dict:
    spec_file = work_dir / f"terrier_worker_{os.getpid()}_{time.time_ns()}.json"
    out_file = work_dir / f"terrier_worker_out_{os.getpid()}_{time.time_ns()}.json"
    spec["out"] = str(out_file)
    write_json(spec_file, spec)
    subprocess.run(
        [sys.executable, SCRIPT_PATH, "--terrier-worker", str(spec_file)], check=True
    )
    result = read_json(out_file)
    spec_file.unlink(missing_ok=True)
    out_file.unlink(missing_ok=True)
    return result


def build_or_load_terrier_index(cfg: Config, indices_root: Path, force_rebuild: bool):
    index_key = index_cache_key(cfg)
    index_dir = indices_root / "terrier" / index_key
    manifest_path = index_dir / "bench_manifest.json"

    if not force_rebuild and manifest_path.exists():
        return index_dir, index_key, False

    shutil.rmtree(index_dir, ignore_errors=True)
    index_dir.mkdir(parents=True, exist_ok=True)
    result = _run_terrier_worker(
        {"action": "index", "dataset": cfg.dataset, "index_dir": str(index_dir)},
        indices_root,
    )
    _write_manifest(index_dir, cfg, index_key, result["build_seconds"])
    return index_dir, index_key, True


def terrier_worker_index(dataset, index_dir: Path) -> float:
    import pyterrier as pt

    if not pt.java.started():
        pt.java.init()

    def doc_iter():
        for doc in tqdm(dataset.docs_iter(), desc="Indexing (Terrier)"):
            text = doc.text if hasattr(doc, "text") else str(doc)
            yield {"docno": doc.doc_id, "text": text}

    # Terrier defaults: UTF tokenizer, Porter stemmer, Terrier stop word list.
    # Single-pass indexing: inverted index only, matching the search-only Pyserini index.
    indexer = pt.IterDictIndexer(
        str(index_dir),
        meta={"docno": 20},
        threads=4,
        type=pt.terrier.IndexingType.SINGLEPASS,
    )
    start = time.perf_counter()
    indexer.index(doc_iter())
    return time.perf_counter() - start


def terrier_worker_search(
    index_dir: Path, queries, qrels: Dict[str, list], top_k, k1, b, repeats: int
) -> dict:
    import pyterrier as pt

    if not pt.java.started():
        pt.java.init()

    index = pt.IndexFactory.of(str(index_dir))
    # Exhaustive DAAT (daat.Full) -- the only matching in stock Terrier 5.11;
    # WAND/block-max dynamic pruning is not shipped in the terrier assemblies.
    retriever = pt.terrier.Retriever(
        index, wmodel="BM25", controls={"bm25.k_1": k1, "bm25.b": b}, num_results=top_k
    )

    samples = []
    mrr = None
    for _ in range(repeats):
        results = {}
        start = time.perf_counter()
        for qid, text in queries:
            # Terrier's query parser chokes on punctuation; keep alphanumerics only.
            clean = re.sub(r"[^A-Za-z0-9 ]", " ", text).strip()
            if not clean:
                results[qid] = []
                continue
            hits = retriever.search(clean)
            results[qid] = [
                (str(d), float(s)) for d, s in zip(hits["docno"], hits["score"])
            ]
        elapsed = time.perf_counter() - start
        new_mrr = compute_mrr(results, {k: set(v) for k, v in qrels.items()})
        if mrr is not None and abs(new_mrr - mrr) > 1e-9:
            raise RuntimeError(f"Terrier MRR drift across repeats: {mrr} vs {new_mrr}")
        mrr = new_mrr
        samples.append(
            {
                "qps": len(queries) / elapsed,
                "elapsed_s": elapsed,
                "timestamp": now_iso(),
                "host": socket.gethostname(),
            }
        )
    return {"samples": samples, "mrr_at_10": mrr}


def terrier_worker_main(spec_file: str) -> None:
    spec = read_json(Path(spec_file))
    if spec["action"] == "index":
        dataset = ir_datasets.load(spec["dataset"])
        build_seconds = terrier_worker_index(dataset, Path(spec["index_dir"]))
        write_json(Path(spec["out"]), {"build_seconds": build_seconds})
    else:
        result = terrier_worker_search(
            Path(spec["index_dir"]),
            [(qid, text) for qid, text in spec["queries"]],
            spec["qrels"],
            spec["top_k"],
            spec["k1"],
            spec["b"],
            spec["repeats"],
        )
        write_json(Path(spec["out"]), result)


class TerrierHandle:
    """No persistent in-process state: each call re-runs a worker subprocess
    that loads the index once and loops all requested repeats internally, so
    the multi-GB index load cost is paid once per call, not once per repeat."""

    def __init__(self, index_dir: Path, indices_root: Path):
        self.index_dir = index_dir
        self.indices_root = indices_root

    def search_repeats(
        self, queries, qrels: Dict[str, set], cfg: Config, repeats: int
    ) -> Tuple[list, float]:
        result = _run_terrier_worker(
            {
                "action": "search",
                "index_dir": str(self.index_dir),
                "queries": queries,
                "qrels": {k: list(v) for k, v in qrels.items()},
                "top_k": cfg.top_k,
                "k1": cfg.k1,
                "b": cfg.b,
                "repeats": repeats,
            },
            self.indices_root,
        )
        return result["samples"], result["mrr_at_10"]


# --------------------------------------------------------------------------
# Search-level cache + repeat runner (generic across systems)
# --------------------------------------------------------------------------


def results_path(output_dir: Path, cfg: Config, search_key: str) -> Path:
    return output_dir / "results" / cfg.system / f"{search_key}.json"


def run_search_repeats(
    handle,
    cfg: Config,
    index_key: str,
    queries,
    qrels: Dict[str, set],
    output_dir: Path,
    repeats: int,
    reset: bool,
) -> dict:
    """Runs (only) as many additional repeats as needed to reach `repeats` samples,
    reusing the already-loaded `handle` -- no index reload happens here."""
    search_key = search_cache_key(cfg, index_key)
    path = results_path(output_dir, cfg, search_key)
    existing = None if reset else read_json(path)
    samples = list(existing["samples"]) if existing else []
    mrr = existing.get("mrr_at_10") if existing else None

    if isinstance(handle, TerrierHandle):
        needed = repeats - len(samples)
        if needed > 0:
            new_samples, new_mrr = handle.search_repeats(queries, qrels, cfg, needed)
            if mrr is not None and abs(new_mrr - mrr) > 1e-9:
                raise RuntimeError(
                    f"MRR drift detected for {cfg.name!r} ({cfg.group}): cached={mrr} new={new_mrr}. "
                    "Search is supposed to be deterministic at fixed top_k -- stopping rather than averaging over a bug."
                )
            mrr = new_mrr
            samples.extend(new_samples)
    else:
        while len(samples) < repeats:
            results, elapsed = handle.search_once(queries, cfg)
            qps = len(queries) / elapsed
            new_mrr = compute_mrr(results, qrels)
            if mrr is not None and abs(new_mrr - mrr) > 1e-9:
                raise RuntimeError(
                    f"MRR drift detected for {cfg.name!r} ({cfg.group}): cached={mrr} new={new_mrr}. "
                    "Search is supposed to be deterministic at fixed top_k -- stopping rather than averaging over a bug."
                )
            mrr = new_mrr
            samples.append(
                {
                    "qps": qps,
                    "elapsed_s": elapsed,
                    "timestamp": now_iso(),
                    "host": socket.gethostname(),
                }
            )

    payload = {
        "config_name": cfg.name,
        "group": cfg.group,
        "system": cfg.system,
        "fields": search_cache_fields(cfg, index_key),
        "n_queries": len(queries),
        "mrr_at_10": mrr,
        "samples": samples,
    }
    write_json(path, payload)
    return payload


# --------------------------------------------------------------------------
# Orchestration
# --------------------------------------------------------------------------


def mean_std(values: List[float]) -> Tuple[float, float]:
    n = len(values)
    m = sum(values) / n
    var = sum((v - m) ** 2 for v in values) / n if n > 1 else 0.0
    return m, var**0.5


def load_queries(dataset, max_queries: int):
    queries = []
    for i, q in enumerate(dataset.queries_iter()):
        if max_queries and i >= max_queries:
            break
        qid = q.query_id if hasattr(q, "query_id") else str(i)
        text = q.text if hasattr(q, "text") else str(q)
        queries.append((qid, text))
    return queries


def run_configs(configs: List[Config], args) -> None:
    output_dir = Path(args.output_dir)
    indices_root = output_dir / "indices"
    systems = set(args.systems.split(","))
    unknown = systems - set(ALL_SYSTEMS)
    if unknown:
        raise SystemExit(f"Unknown --systems entries: {sorted(unknown)}")

    configs = [c for c in configs if c.system in systems]
    if args.only:
        configs = [
            c for c in configs if any(s in c.name or s in c.group for s in args.only)
        ]
    if args.dataset:
        configs = [replace(c, dataset=args.dataset) for c in configs]
    if args.max_queries:
        configs = [replace(c, max_queries=args.max_queries) for c in configs]
    if args.top_k:
        configs = [replace(c, top_k=args.top_k) for c in configs]

    if not configs:
        print("No configs selected (check --systems/--suite/--only).")
        return

    # Group configs by (dataset, index_cache_key) so each distinct index is
    # built/loaded exactly once and kept resident across every repeat and
    # every search-parameter variant (algorithm/top_k/k1/b) that shares it.
    by_dataset: Dict[str, List[Config]] = {}
    for c in configs:
        by_dataset.setdefault(c.dataset, []).append(c)

    for dataset_name, dataset_configs in by_dataset.items():
        dataset = ir_datasets.load(dataset_name)
        qrels = load_qrels(dataset)
        # Queries are keyed by max_queries so a smoke-test run (--max-queries)
        # and a full run don't collide in the search cache (see search_cache_fields).
        by_max_queries: Dict[int, list] = {}

        groups: Dict[Tuple[str, str], List[Config]] = {}
        for c in dataset_configs:
            groups.setdefault((c.system, index_cache_key(c)), []).append(c)

        for (system, index_key), group_configs in groups.items():
            rep_cfg = group_configs[0]

            print(
                f"\n=== [{system}] index {index_key} ({', '.join(c.name for c in group_configs)}) ==="
            )
            if system == "impact-index":
                raw_or_final, _, index_dir, built = build_or_load_impact_index(
                    rep_cfg, indices_root, dataset, force_rebuild=args.reset_index
                )
                handle = ImpactIndexHandle(raw_or_final)
                size_fn = partial(dir_size_mb, index_dir)
            elif system == "pyserini":
                lucene_dir, _, index_dir, built = build_or_load_pyserini_index(
                    rep_cfg, indices_root, dataset, force_rebuild=args.reset_index
                )
                handle = PyseriniHandle(lucene_dir)
                # index_dir also holds the raw JSONL "collection" dump used only to
                # build the index -- only the actual `lucene/` index counts.
                size_fn = partial(dir_size_mb, lucene_dir)
            elif system == "pisa":
                if not pisa_available():
                    print(
                        "  PISA unavailable on this platform (Linux x86_64 only) -- skipping."
                    )
                    continue
                pisa_index, _, index_dir, built = build_or_load_pisa_index(
                    rep_cfg, indices_root, dataset, force_rebuild=args.reset_index
                )
                handle = PisaHandle(pisa_index)
                # PISA's quantized/block-encoded files are created lazily on the
                # first index.bm25() call (inside search), not at index() time --
                # measuring here (before any search has run) would see only the
                # raw fwd/inv files and undercount. Measure lazily, after search.
                size_fn = partial(pisa_index_size_mb, index_dir)
            elif system == "terrier":
                index_dir, _, built = build_or_load_terrier_index(
                    rep_cfg, indices_root, force_rebuild=args.reset_index
                )
                handle = TerrierHandle(index_dir, indices_root)
                size_fn = partial(dir_size_mb, index_dir)
            else:
                raise AssertionError(system)

            reset_search = args.reset_search_statistics or (args.reset_index and built)
            for cfg in group_configs:
                # Each cfg in a group shares the same index but may specify its
                # own query-set size; cache per distinct max_queries value.
                if cfg.max_queries not in by_max_queries:
                    by_max_queries[cfg.max_queries] = load_queries(
                        dataset, cfg.max_queries
                    )
                queries = by_max_queries[cfg.max_queries]
                payload = run_search_repeats(
                    handle,
                    cfg,
                    index_key,
                    queries,
                    qrels,
                    output_dir,
                    args.repeats,
                    reset_search,
                )
                qps_values = [s["qps"] for s in payload["samples"]]
                m, sd = mean_std(qps_values)
                print(
                    f"  {cfg.name:45s} algo={cfg.algorithm:16s} "
                    f"{m:7.1f} +/- {sd:5.1f} q/s  (n={len(qps_values)})  "
                    f"MRR@10={payload['mrr_at_10']:.4f}  size={size_fn():.1f}MB"
                )

            del handle
            gc.collect()


# --------------------------------------------------------------------------
# Rendering: turn cached JSON results into README.md/BENCHMARKS.md-style
# markdown tables. Prints to stdout; does not touch the docs itself, since
# table structure/prose must be preserved exactly (see module docstring).
# --------------------------------------------------------------------------


def _load_group_results(output_dir: Path, configs: List[Config]) -> Dict[str, dict]:
    """Maps `f"{group}::{name}::{algorithm}"` -> cached search payload, best-effort."""
    out = {}
    for cfg in configs:
        index_key = index_cache_key(cfg)
        search_key = search_cache_key(cfg, index_key)
        payload = read_json(results_path(output_dir, cfg, search_key))
        if payload:
            out[f"{cfg.group}::{cfg.name}::{cfg.algorithm}"] = payload
    return out


def _qps_str(payload: Optional[dict]) -> str:
    if not payload or not payload.get("samples"):
        return "n/a"
    m, sd = mean_std([s["qps"] for s in payload["samples"]])
    return f"{m:.0f} +/- {sd:.0f}"


def _total_build_seconds(cfg: Config, output_dir: Path) -> Optional[float]:
    """Sums the raw index's build time with the compression step's, since a
    compressed config's own manifest only records the incremental compress()
    call -- the raw index it was compressed from is a separate cached build
    (see build_or_load_impact_index) with its own manifest and build time."""
    total = 0.0
    seen_any = False
    cfg_chain = [cfg]
    if cfg.compression is not None:
        cfg_chain.append(replace(cfg, compression=None))
    for c in cfg_chain:
        index_dir = output_dir / "indices" / "impact-index" / index_cache_key(c)
        manifest = read_json(index_dir / "bench_manifest.json")
        if manifest and manifest.get("build_seconds"):
            total += manifest["build_seconds"]
            seen_any = True
    return total if seen_any else None


def render_ablation(output_dir: Path) -> None:
    results = _load_group_results(output_dir, ABLATION_CONFIGS)

    def row(cfg: Config) -> str:
        payload = results.get(f"{cfg.group}::{cfg.name}::{cfg.algorithm}")
        build_seconds = _total_build_seconds(cfg, output_dir)
        build_s = f"{build_seconds:.0f}" if build_seconds is not None else "(cached)"
        size_mb = dir_size_mb(
            output_dir / "indices" / "impact-index" / index_cache_key(cfg)
        )
        mrr = f"{payload['mrr_at_10']:.4f}" if payload else "n/a"
        return (
            f"| {cfg.name} | {build_s} | {size_mb:.1f} | {_qps_str(payload)} | {mrr} |"
        )

    print("\n**Porter/Lucene pipeline** (matches Pyserini's own defaults):\n")
    print("| Config | Build (s) | Size (MB) | q/s | MRR@10 |")
    print("|--------|-----------|-----------|-----|--------|")
    for cfg in ABLATION_CONFIGS:
        if cfg.group == "ablation-porter":
            print(row(cfg))

    print(
        "\n**Snowball/Terrier pipeline** (matches PISA's and Terrier 5's own defaults):\n"
    )
    print("| Config | Build (s) | Size (MB) | q/s | MRR@10 |")
    print("|--------|-----------|-----------|-----|--------|")
    for cfg in ABLATION_CONFIGS:
        if cfg.group == "ablation-snowball":
            print(row(cfg))


def _comparison_index_size_mb(cfg: Config, output_dir: Path) -> float:
    """Same per-system size accounting as run_configs' size_fn, but computed
    fresh from disk at render time (after any PISA search has materialized
    its lazily-built quantized files, and excluding Pyserini's raw JSONL
    "collection" dump, which lives next to -- not inside -- its lucene/ dir)."""
    index_key = index_cache_key(cfg)
    index_dir = output_dir / "indices" / cfg.system / index_key
    if cfg.system == "pyserini":
        return dir_size_mb(index_dir / "lucene")
    if cfg.system == "pisa":
        return pisa_index_size_mb(index_dir)
    return dir_size_mb(index_dir)


def render_comparison(output_dir: Path) -> None:
    results = _load_group_results(output_dir, COMPARISON_CONFIGS)
    by_group_name = {(c.group, c.name): c for c in COMPARISON_CONFIGS}

    def qps(name: str, group: str, algorithm: str = "maxscore") -> str:
        return _qps_str(results.get(f"{group}::{name}::{algorithm}"))

    def mrr(name: str, group: str, algorithm: str = "maxscore") -> str:
        p = results.get(f"{group}::{name}::{algorithm}")
        return f"{p['mrr_at_10']:.4f}" if p else "n/a"

    def size(name: str, group: str) -> str:
        cfg = by_group_name.get((group, name))
        if not cfg:
            return "n/a"
        return f"{_comparison_index_size_mb(cfg, output_dir) / 1024:.2f} GB"

    print("\n### Lucene-aligned (Porter stemmer, Lucene's ~33-word stopword list)\n")
    print("| System | x86 q/s | Index size | MRR@10 |")
    print("|--------|---------|-----------|--------|")
    for name, algo in [
        ("impact-index (compressed, MaxScore)", "maxscore"),
        ("impact-index (compressed + reordered, MaxScore)", "maxscore"),
        ("impact-index (compressed, WAND/BMW)", "wand"),
        ("Pyserini (Lucene)", "maxscore"),
    ]:
        print(
            f"| {name} | {qps(name, 'lucene-aligned', algo)} | {size(name, 'lucene-aligned')} | "
            f"{mrr(name, 'lucene-aligned', algo)} |"
        )

    print(
        "\n### Terrier-aligned (Snowball/Porter2 stemmer, Terrier's own ~730-word stopword list)\n"
    )
    print("| System | x86 q/s | Index size | MRR@10 |")
    print("|--------|---------|-----------|--------|")
    for name, algo in [
        ("impact-index (compressed, MaxScore)", "maxscore"),
        ("impact-index (compressed, WAND/BMW)", "wand"),
        ("Terrier 5 (PyTerrier)", "maxscore"),
        ("PISA (Block-Max WAND)", "block_max_wand"),
        ("PISA (MaxScore)", "block_max_maxscore"),
    ]:
        print(
            f"| {name} | {qps(name, 'terrier-aligned', algo)} | {size(name, 'terrier-aligned')} | "
            f"{mrr(name, 'terrier-aligned', algo)} |"
        )


# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--output-dir", type=str, required=False, default="bench_output"
    )
    parser.add_argument(
        "--suite", choices=["ablation", "comparison", "all"], default="all"
    )
    parser.add_argument(
        "--systems",
        type=str,
        default="impact-index",
        help="comma-separated subset of " + ",".join(ALL_SYSTEMS),
    )
    parser.add_argument(
        "--repeats", type=int, default=5, help="search-only repeats per config"
    )
    parser.add_argument(
        "--reset-search-statistics",
        action="store_true",
        help="ignore cached search results; keep cached indices",
    )
    parser.add_argument(
        "--reset-index",
        action="store_true",
        help="force a full rebuild (index + search) for selected configs",
    )
    parser.add_argument(
        "--only",
        action="append",
        default=None,
        help="filter configs by substring of name/group (repeatable)",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default=None,
        help="override dataset for all selected configs",
    )
    parser.add_argument(
        "--max-queries",
        type=int,
        default=0,
        help="override query-count limit for all selected configs (0 = all)",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=0,
        help="override top_k for all selected configs (0 = use config default)",
    )
    parser.add_argument(
        "--render",
        choices=["ablation", "comparison", "all"],
        default=None,
        help="print markdown tables from cached results and exit",
    )
    parser.add_argument("--terrier-worker", type=str, help=argparse.SUPPRESS)
    args = parser.parse_args()

    if args.terrier_worker:
        terrier_worker_main(args.terrier_worker)
        return

    output_dir = Path(args.output_dir)

    if args.render:
        if args.render in ("ablation", "all"):
            print("=" * 70)
            print("ABLATION")
            print("=" * 70)
            render_ablation(output_dir)
        if args.render in ("comparison", "all"):
            print("\n" + "=" * 70)
            print("COMPARISON")
            print("=" * 70)
            render_comparison(output_dir)
        return

    suites = ["ablation", "comparison"] if args.suite == "all" else [args.suite]
    for suite in suites:
        print(f"\n{'#' * 70}\n# SUITE: {suite}\n{'#' * 70}")
        run_configs(SUITES[suite], args)


if __name__ == "__main__":
    main()
