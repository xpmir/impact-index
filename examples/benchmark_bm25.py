#!/usr/bin/env python3
# /// script
# requires-python = ">=3.9"
# dependencies = [
#     "ir-datasets",
#     "pyserini",
#     "python-terrier>=0.11",
#     "tqdm",
#     "numpy",
#     "cbor2",
#     "snowballstemmer",
# ]
# ///
"""
Benchmark BM25: impact-index vs Pyserini vs Terrier 5 on MS MARCO passage.

Java requirements: Terrier 5 (via PyTerrier) needs Java 11+, Pyserini needs
Java 21 — a single OpenJDK 21 install (e.g. Temurin) satisfies both. Set
JAVA_HOME if it is not picked up automatically.

Usage:
    # First time: build all indices and benchmark
    uv run --with . examples/benchmark_bm25.py --output-dir /tmp/bm25_bench

    # Re-run search only (indices already built)
    uv run --with . examples/benchmark_bm25.py --output-dir /tmp/bm25_bench

    # Limit queries for quick test
    uv run --with . examples/benchmark_bm25.py --output-dir /tmp/bm25_bench --max-queries 100

    # With compressed index
    uv run --with . examples/benchmark_bm25.py --output-dir /tmp/bm25_bench \\
        --compressed-index 'nbits=16 block-size=128'

    # With split compressed index
    uv run --with . examples/benchmark_bm25.py --output-dir /tmp/bm25_bench \\
        --compressed-index 'split=0.9 nbits=8'
"""

import argparse
import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional

import ir_datasets
from tqdm import tqdm

import impact_index
from impact_index import (
    BitPackingCompressor,
    CompressionTransform,
    GlobalImpactQuantizer,
    SplitIndexTransform,
)


# --- Query Analyzer (loads saved vocabulary for use after index build) ---


class QueryAnalyzer:
    """Wraps a TextAnalyzer for query analysis."""

    def __init__(self, analyzer):
        self.analyzer = analyzer

    def analyze_query(self, text: str) -> Dict[int, float]:
        return self.analyzer.analyze_query(text)


# --- Compressed Index Config (same as splade_benchmark.py) ---


class CompressedIndexConfig:
    """Configuration for a compressed index."""

    def __init__(
        self,
        nbits: int = 0,  # 0 = lossless integer bitpacking (best for BM25)
        block_size: int = 128,
        split_quantiles: Optional[List[float]] = None,
    ):
        self.nbits = nbits
        self.block_size = block_size
        self.split_quantiles = split_quantiles

    @classmethod
    def parse(cls, config_str: str) -> "CompressedIndexConfig":
        config = cls()
        for part in config_str.split():
            if "=" not in part:
                raise ValueError(f"Invalid config part: {part}. Expected key=value.")
            key, value = part.split("=", 1)
            key = key.strip().lower().replace("-", "_")
            if key == "nbits":
                config.nbits = int(value)
            elif key == "block_size":
                config.block_size = int(value)
            elif key == "split":
                config.split_quantiles = [float(q) for q in value.split(",")]
            else:
                raise ValueError(f"Unknown config key: {key}")
        return config

    def get_dir_name(self) -> str:
        parts = []
        if self.split_quantiles:
            quantiles_str = "_".join(
                f"{q:.2f}".replace(".", "") for q in self.split_quantiles
            )
            parts.append(f"split_{quantiles_str}")
        parts.append("pfor")  # PFOR-delta codec
        parts.append(f"nb{self.nbits}")
        parts.append(f"bs{self.block_size}")
        return "_".join(parts)

    def get_display_name(self) -> str:
        if self.split_quantiles:
            return f"Split({','.join(f'{q:.2f}' for q in self.split_quantiles)}) nb={self.nbits} bs={self.block_size}"
        return f"Compressed nb={self.nbits} bs={self.block_size}"

    def is_split(self) -> bool:
        return self.split_quantiles is not None


# --- Index Building ---


def build_impact_index(dataset, index_dir: Path, k1: float, b: float):
    """Build a BM25 impact-index from ir_datasets. Skips if already built."""
    done_file = index_dir / ".done"

    if done_file.exists():
        print(f"Impact-index already built at {index_dir}, loading...")
        index = impact_index.Index.load(str(index_dir), in_memory=True)
        scoring = impact_index.BM25Scoring(k1=k1, b=b)
        scored_index = index.with_scoring(scoring)
        analyzer = QueryAnalyzer(index.analyzer())
        return index, scored_index, analyzer

    print(f"Building impact-index at {index_dir}...")
    index_dir.mkdir(parents=True, exist_ok=True)

    options = impact_index.BuilderOptions()
    # Default in_memory_threshold=128 matches block-max pruning block size

    builder = impact_index.BOWIndexBuilder(
        str(index_dir),
        options=options,
        dtype="int32",
        stemmer="porter",  # Lucene-compatible Porter (ported from PorterStemmer.java)
        stop_words=True,  # use Lucene's default English stop words
    )

    # Collect documents in batches for parallel text analysis
    BATCH_SIZE = 10000
    batch = []
    for doc in tqdm(dataset.docs_iter(), desc="Indexing (impact-index)"):
        text = doc.text if hasattr(doc, "text") else str(doc)
        doc_id = int(doc.doc_id) if doc.doc_id.isdigit() else hash(doc.doc_id)
        batch.append((doc_id, text))
        if len(batch) >= BATCH_SIZE:
            builder.add_texts(batch)
            batch = []
    if batch:
        builder.add_texts(batch)

    index = builder.build(in_memory=True)
    done_file.touch()

    scoring = impact_index.BM25Scoring(k1=k1, b=b)
    scored_index = index.with_scoring(scoring)
    analyzer = QueryAnalyzer(index.analyzer())

    return index, scored_index, analyzer


def build_compressed_index(
    source_index: impact_index.Index,
    output_dir: Path,
    config: CompressedIndexConfig,
    k1: float,
    b: float,
):
    """Build a compressed (and optionally split) index. Skips if already built."""
    done_file = output_dir / ".done"

    if done_file.exists():
        print(f"  Compressed index already built at {output_dir}, loading...")
        index = impact_index.Index.load(str(output_dir), in_memory=True)
    else:
        output_dir.mkdir(parents=True, exist_ok=True)

        if config.is_split():
            # Split requires the full transform pipeline
            doc_ids_compressor = BitPackingCompressor()
            if config.nbits == 0:
                impact_compressor = impact_index.BitPackedIntCompressor()
            else:
                impact_compressor = GlobalImpactQuantizer(config.nbits)
            compression_transform = CompressionTransform(
                config.block_size, doc_ids_compressor, impact_compressor
            )
            transform = SplitIndexTransform(
                config.split_quantiles, compression_transform
            )
            transform.process(str(output_dir), source_index)
            index = impact_index.Index.load(str(output_dir), in_memory=True)
        else:
            # Simple compression: use Index.compress()
            index = source_index.compress(
                str(output_dir),
                block_size=config.block_size,
                nbits=config.nbits,
            )

        done_file.touch()

    scoring = impact_index.BM25Scoring(k1=k1, b=b)
    return index.with_scoring(scoring)


def build_pyserini_index(dataset, index_dir: Path):
    """Build a Pyserini BM25 index from ir_datasets. Skips if already built."""
    lucene_dir = index_dir / "lucene"
    done_file = index_dir / ".done"

    if done_file.exists():
        print(f"Pyserini index already built at {lucene_dir}")
        return lucene_dir

    print(f"Building Pyserini index at {index_dir}...")
    jsonl_dir = index_dir / "collection"
    jsonl_dir.mkdir(parents=True, exist_ok=True)

    jsonl_path = jsonl_dir / "docs.jsonl"
    with open(jsonl_path, "w") as f:
        for doc in tqdm(dataset.docs_iter(), desc="Writing JSONL"):
            text = doc.text if hasattr(doc, "text") else str(doc)
            json.dump({"id": doc.doc_id, "contents": text}, f)
            f.write("\n")

    cmd = [
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
        # Don't store positions, doc vectors, or raw text — only the search index
    ]
    print(f"Running: {' '.join(cmd)}")
    subprocess.run(cmd, check=True)
    done_file.touch()

    return lucene_dir


# PyTerrier and Pyserini cannot share a process: both use pyjnius, and the
# JVM classpath is fixed by whichever library starts it first. Everything
# touching Terrier therefore runs in a worker subprocess (its own JVM),
# spawned via `--terrier-worker`.


def init_pyterrier():
    """Initialize PyTerrier (starts the JVM; Terrier 5 requires Java 11+)."""
    import pyterrier as pt

    if not pt.java.started():
        pt.java.init()
    return pt


def run_terrier_subprocess(spec: dict, work_dir: Path):
    """Run a Terrier action (index/search) in a subprocess with its own JVM."""
    spec_file = work_dir / "terrier_worker_spec.json"
    with open(spec_file, "w") as f:
        json.dump(spec, f)
    subprocess.run(
        [sys.executable, __file__, "--terrier-worker", str(spec_file)],
        check=True,
    )
    spec_file.unlink()


def build_terrier_index(dataset_name: str, index_dir: Path):
    """Build a Terrier 5 index (in a subprocess). Skips if already built."""
    done_file = index_dir / ".done"

    if done_file.exists():
        print(f"Terrier index already built at {index_dir}")
        return index_dir

    run_terrier_subprocess(
        {"action": "index", "dataset": dataset_name, "index_dir": str(index_dir)},
        index_dir.parent,
    )
    return index_dir


def terrier_worker_index(dataset, index_dir: Path):
    """Worker-side Terrier index build."""
    pt = init_pyterrier()
    done_file = index_dir / ".done"
    print(f"Building Terrier index at {index_dir}...")
    index_dir.mkdir(parents=True, exist_ok=True)

    def doc_iter():
        for doc in tqdm(dataset.docs_iter(), desc="Indexing (Terrier)"):
            text = doc.text if hasattr(doc, "text") else str(doc)
            yield {"docno": doc.doc_id, "text": text}

    # Terrier defaults: UTF tokenizer, Porter stemmer, Terrier stop word list.
    # Single-pass indexing: inverted index only (no direct index), matching
    # the search-only Pyserini index
    indexer = pt.IterDictIndexer(
        str(index_dir),
        meta={"docno": 20},
        threads=4,
        type=pt.terrier.IndexingType.SINGLEPASS,
    )
    indexer.index(doc_iter())
    done_file.touch()

    return index_dir


# --- Search ---


def search_impact_index(analyzer, scored_index, queries, top_k, method="wand"):
    """Search with impact-index BM25 and return results + timing."""
    search_fn = (
        scored_index.search_wand if method == "wand" else scored_index.search_maxscore
    )
    results = {}
    start = time.perf_counter()

    for qid, text in tqdm(queries, desc=f"Searching (impact-index {method.upper()})"):
        query = analyzer.analyze_query(text)
        if not query:
            results[qid] = []
            continue
        hits = search_fn(query, top_k)
        results[qid] = [(h.docid, h.score) for h in hits]

    elapsed = time.perf_counter() - start
    return results, elapsed


def search_pyserini(lucene_dir, queries, top_k, k1=0.9, b=0.4):
    """Search with Pyserini BM25 and return results + timing."""
    # pyserini.encode instantiates an OpenAI client at import time; a dummy
    # key avoids an import error (the client is never used here)
    os.environ.setdefault("OPENAI_API_KEY", "unused")
    from pyserini.search.lucene import LuceneSearcher

    searcher = LuceneSearcher(str(lucene_dir))
    searcher.set_bm25(k1, b)

    results = {}
    start = time.perf_counter()

    for qid, text in tqdm(queries, desc="Searching (Pyserini)"):
        hits = searcher.search(text, k=top_k)
        results[qid] = [(hit.docid, hit.score) for hit in hits]

    elapsed = time.perf_counter() - start
    return results, elapsed


def search_terrier(index_dir: Path, queries, top_k, k1=0.9, b=0.4):
    """Search with Terrier 5 BM25 (in a subprocess) and return results + timing."""
    out_file = index_dir.parent / "terrier_results.json"
    run_terrier_subprocess(
        {
            "action": "search",
            "index_dir": str(index_dir),
            "queries": queries,
            "top_k": top_k,
            "k1": k1,
            "b": b,
            "out": str(out_file),
        },
        index_dir.parent,
    )
    with open(out_file) as f:
        payload = json.load(f)
    out_file.unlink()
    return payload["results"], payload["elapsed"]


def terrier_worker_search(index_dir: Path, queries, top_k, k1, b, out_file: Path):
    """Worker-side Terrier BM25 search; writes results + timing to out_file."""
    pt = init_pyterrier()

    index = pt.IndexFactory.of(str(index_dir))
    retriever = pt.terrier.Retriever(
        index,
        wmodel="BM25",
        controls={"bm25.k_1": k1, "bm25.b": b},
        num_results=top_k,
    )

    results = {}
    start = time.perf_counter()

    for qid, text in tqdm(queries, desc="Searching (Terrier)"):
        # Terrier's query parser chokes on punctuation; keep alphanumerics only
        clean = re.sub(r"[^A-Za-z0-9 ]", " ", text).strip()
        if not clean:
            results[qid] = []
            continue
        hits = retriever.search(clean)
        results[qid] = [
            (str(d), float(s)) for d, s in zip(hits["docno"], hits["score"])
        ]

    elapsed = time.perf_counter() - start
    with open(out_file, "w") as f:
        json.dump({"results": results, "elapsed": elapsed}, f)


def terrier_worker_main(spec_file: str):
    """Entry point for the Terrier worker subprocess."""
    with open(spec_file) as f:
        spec = json.load(f)
    if spec["action"] == "index":
        dataset = ir_datasets.load(spec["dataset"])
        terrier_worker_index(dataset, Path(spec["index_dir"]))
    else:
        terrier_worker_search(
            Path(spec["index_dir"]),
            [(qid, text) for qid, text in spec["queries"]],
            spec["top_k"],
            spec["k1"],
            spec["b"],
            Path(spec["out"]),
        )


# --- Comparison ---


def compare_results(impact_results, pyserini_results, top_k):
    """Compare ranking results between the two systems."""
    overlap_at_10 = []
    overlap_at_k = []
    n_queries = 0

    for qid in impact_results:
        if qid not in pyserini_results:
            continue
        n_queries += 1

        impact_docs = [str(docid) for docid, _ in impact_results[qid]]
        pyserini_docs = [docid for docid, _ in pyserini_results[qid]]

        # Overlap at 10
        set_i10 = set(impact_docs[:10])
        set_p10 = set(pyserini_docs[:10])
        if set_i10 or set_p10:
            overlap_at_10.append(
                len(set_i10 & set_p10) / max(len(set_i10), len(set_p10))
            )

        # Overlap at top_k
        set_ik = set(impact_docs[:top_k])
        set_pk = set(pyserini_docs[:top_k])
        if set_ik or set_pk:
            overlap_at_k.append(len(set_ik & set_pk) / max(len(set_ik), len(set_pk)))

    avg_overlap_10 = sum(overlap_at_10) / len(overlap_at_10) if overlap_at_10 else 0
    avg_overlap_k = sum(overlap_at_k) / len(overlap_at_k) if overlap_at_k else 0

    return {
        "n_queries": n_queries,
        "avg_overlap@10": avg_overlap_10,
        f"avg_overlap@{top_k}": avg_overlap_k,
    }


def compute_mrr(results, qrels, k=10):
    """Compute MRR@k given results and qrels."""
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


def load_qrels(dataset):
    """Load qrels from ir_datasets, returning {qid: set(doc_ids)}."""
    qrels = {}
    for qrel in dataset.qrels_iter():
        if qrel.relevance > 0:
            qid = qrel.query_id
            if qid not in qrels:
                qrels[qid] = set()
            qrels[qid].add(str(qrel.doc_id))
    return qrels


def get_dir_size_mb(path: Path) -> float:
    """Get total size of a directory in MB."""
    total = sum(f.stat().st_size for f in path.rglob("*") if f.is_file())
    return total / 1024 / 1024


# --- Main ---


def main():
    parser = argparse.ArgumentParser(
        description="Benchmark BM25: impact-index vs Pyserini vs Terrier 5"
    )
    parser.add_argument("--output-dir", type=str)
    parser.add_argument("--terrier-worker", type=str, help=argparse.SUPPRESS)
    parser.add_argument("--dataset", type=str, default="msmarco-passage/dev/small")
    parser.add_argument(
        "--max-queries", type=int, default=0, help="Limit number of queries (0 = all)"
    )
    parser.add_argument("--top-k", type=int, default=100)
    parser.add_argument("--k1", type=float, default=0.9)
    parser.add_argument("--b", type=float, default=0.4)
    parser.add_argument(
        "--compressed-index",
        type=str,
        action="append",
        dest="compressed_indices",
        metavar="CONFIG",
        help=(
            "Add a compressed index configuration. Can be specified multiple times. "
            "Format: 'key=value key=value ...'. Keys: nbits (default 8), "
            "block-size (default 128), split (comma-separated quantiles). "
            "Examples: 'nbits=16', 'split=0.9 nbits=8'"
        ),
    )
    args = parser.parse_args()

    # Terrier runs in its own subprocess (separate JVM from Pyserini)
    if args.terrier_worker:
        terrier_worker_main(args.terrier_worker)
        return

    if not args.output_dir:
        parser.error("--output-dir is required")

    # Parse compressed index configurations
    index_configs = []
    if args.compressed_indices:
        for config_str in args.compressed_indices:
            index_configs.append(CompressedIndexConfig.parse(config_str))
    else:
        # Default: always test with a compressed index (block_size=128)
        # where block-max optimizations are effective
        index_configs.append(CompressedIndexConfig(nbits=0, block_size=128))

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    impact_dir = output_dir / "impact_index"
    pyserini_dir = output_dir / "pyserini"
    pyserini_dir.mkdir(parents=True, exist_ok=True)
    terrier_dir = output_dir / "terrier"

    print(f"Dataset: {args.dataset}")
    print(f"BM25 params: k1={args.k1}, b={args.b}")
    print(f"Top-k: {args.top_k}")

    dataset = ir_datasets.load(args.dataset)

    # --- Build indices ---
    raw_index, scored_index, analyzer = build_impact_index(
        dataset, impact_dir, k1=args.k1, b=args.b
    )
    lucene_dir = build_pyserini_index(dataset, pyserini_dir)
    build_terrier_index(args.dataset, terrier_dir)

    # Build compressed indices if requested
    compressed_scored = []
    for config in index_configs:
        config_dir = output_dir / f"index_{config.get_dir_name()}"
        print(f"\n=== Building {config.get_display_name()} ===")
        scored = build_compressed_index(
            raw_index, config_dir, config, k1=args.k1, b=args.b
        )
        compressed_scored.append((config, scored))

    # --- Load queries ---
    queries = []
    for i, query in enumerate(dataset.queries_iter()):
        if args.max_queries > 0 and i >= args.max_queries:
            break
        qid = query.query_id if hasattr(query, "query_id") else str(i)
        text = query.text if hasattr(query, "text") else str(query)
        queries.append((qid, text))
    print(f"\nLoaded {len(queries)} queries")

    # --- Search ---
    all_results = {}

    print("\n--- impact-index BM25 (MaxScore) ---")
    impact_ms_results, impact_ms_time = search_impact_index(
        analyzer, scored_index, queries, args.top_k, method="maxscore"
    )
    all_results["impact-index MaxScore"] = (impact_ms_results, impact_ms_time)
    impact_results = impact_ms_results  # for comparison below
    print(f"Time: {impact_ms_time:.2f}s ({len(queries) / impact_ms_time:.1f} q/s)")

    # Search compressed indices (MaxScore only — WAND is much slower)
    for config, scored in compressed_scored:
        name = config.get_display_name()
        methods = ["maxscore"]

        for method in methods:
            label = f"{name} ({method.upper()})"
            print(f"\n--- {label} ---")
            results, elapsed = search_impact_index(
                analyzer, scored, queries, args.top_k, method=method
            )
            all_results[label] = (results, elapsed)
            print(f"Time: {elapsed:.2f}s ({len(queries) / elapsed:.1f} q/s)")

    print("\n--- Pyserini BM25 ---")
    pyserini_results, pyserini_time = search_pyserini(
        lucene_dir, queries, args.top_k, k1=args.k1, b=args.b
    )
    all_results["Pyserini"] = (pyserini_results, pyserini_time)
    print(f"Time: {pyserini_time:.2f}s ({len(queries) / pyserini_time:.1f} q/s)")

    print("\n--- Terrier 5 BM25 ---")
    terrier_results, terrier_time = search_terrier(
        terrier_dir, queries, args.top_k, k1=args.k1, b=args.b
    )
    all_results["Terrier 5"] = (terrier_results, terrier_time)
    print(f"Time: {terrier_time:.2f}s ({len(queries) / terrier_time:.1f} q/s)")

    # --- Summary ---
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)

    # Index sizes
    print("\nIndex sizes:")
    print(f"  impact-index:  {get_dir_size_mb(impact_dir):.2f} MB")
    for config in index_configs:
        config_dir = output_dir / f"index_{config.get_dir_name()}"
        print(f"  {config.get_display_name()}: {get_dir_size_mb(config_dir):.2f} MB")
    print(f"  Pyserini:      {get_dir_size_mb(lucene_dir):.2f} MB")
    print(f"  Terrier 5:     {get_dir_size_mb(terrier_dir):.2f} MB")

    # Search performance
    print("\nSearch performance:")
    for label, (results, elapsed) in all_results.items():
        qps = len(queries) / elapsed
        print(f"  {label:40s} {elapsed:7.2f}s  ({qps:7.1f} q/s)")

    # MRR@10
    qrels = load_qrels(dataset)
    print("\nMRR@10:")
    for label, (results, _) in all_results.items():
        mrr = compute_mrr(results, qrels, k=10)
        print(f"  {label:40s} {mrr:.4f}")

    # Result comparison (all vs Pyserini)
    print("\nResult overlap vs Pyserini:")
    for label, (results, _) in all_results.items():
        if label == "Pyserini":
            continue
        metrics = compare_results(results, pyserini_results, args.top_k)
        print(
            f"  {label:40s} @10={metrics['avg_overlap@10']:.4f}  "
            f"@{args.top_k}={metrics[f'avg_overlap@{args.top_k}']:.4f}"
        )

    # Show sample differences
    print("\n--- Sample query comparisons (impact-index MaxScore vs Pyserini) ---")
    shown = 0
    for qid, text in queries[:50]:
        if qid not in pyserini_results or qid not in impact_results:
            continue
        i_docs = [str(d) for d, _ in impact_results[qid][:5]]
        p_docs = [d for d, _ in pyserini_results[qid][:5]]
        if set(i_docs[:5]) != set(p_docs[:5]):
            print(f"\nQuery '{text}' (qid={qid}):")
            print(f"  impact-index top5: {i_docs}")
            print(f"  Pyserini     top5: {p_docs}")
            i_scores = [f"{s:.4f}" for _, s in impact_results[qid][:5]]
            p_scores = [f"{s:.4f}" for _, s in pyserini_results[qid][:5]]
            print(f"  impact scores:     {i_scores}")
            print(f"  Pyserini scores:   {p_scores}")
            shown += 1
            if shown >= 5:
                break


if __name__ == "__main__":
    main()
