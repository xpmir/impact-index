#!/usr/bin/env python3
# /// script
# requires-python = ">=3.9"
# dependencies = [
#     "ir-datasets",
#     "cbor2",
#     "snowballstemmer",
# ]
# ///
"""Profile search performance on MS MARCO.

Usage:
    uv run --with . examples/profile_search.py --output-dir /tmp/bm25_bench
"""

import argparse
import re
import time
from collections import defaultdict
from pathlib import Path
from typing import Dict

import impact_index


class QueryAnalyzer:
    def __init__(self, index_dir: Path, language: str = "english"):
        import cbor2
        import snowballstemmer

        vocab_path = index_dir / "vocab.cbor"
        with open(vocab_path, "rb") as f:
            vocab_data = cbor2.load(f)
        self.term_to_id: Dict[str, int] = vocab_data["term_to_id"]
        self.stemmer = snowballstemmer.stemmer(language)

    def analyze_query(self, text: str) -> Dict[int, float]:
        tf: Dict[int, float] = defaultdict(float)
        for token in re.findall(r"\w+", text.lower(), re.UNICODE):
            stemmed = self.stemmer.stemWord(token)
            term_id = self.term_to_id.get(stemmed)
            if term_id is not None:
                tf[term_id] += 1.0
        return dict(tf)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=str, required=True)
    parser.add_argument("--dataset", type=str, default="msmarco-passage/dev/small")
    parser.add_argument("--max-queries", type=int, default=500)
    parser.add_argument("--top-k", type=int, default=100)
    parser.add_argument("--k1", type=float, default=0.9)
    parser.add_argument("--b", type=float, default=0.4)
    parser.add_argument(
        "--method", type=str, default="maxscore", choices=["wand", "maxscore"]
    )
    parser.add_argument(
        "--index", type=str, default="compressed", choices=["raw", "compressed"]
    )
    parser.add_argument("--warmup", type=int, default=1, help="Number of warmup passes")
    parser.add_argument("--passes", type=int, default=3, help="Number of timed passes")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    impact_dir = output_dir / "impact_index"

    # Load analyzer
    analyzer = QueryAnalyzer(impact_dir)

    # Load index
    if args.index == "compressed":
        # Find compressed index dir
        compressed_dirs = list(output_dir.glob("index_bp_*"))
        if not compressed_dirs:
            compressed_dirs = list(output_dir.glob("index_nb*"))
        if not compressed_dirs:
            print("No compressed index found")
            return
        idx_dir = compressed_dirs[0]
        print(f"Loading compressed index from {idx_dir}")
        index = impact_index.Index.load(str(idx_dir), in_memory=True)
    else:
        print(f"Loading raw index from {impact_dir}")
        index = impact_index.Index.load(str(impact_dir), in_memory=True)

    doc_meta = impact_index.DocMetadata.load(str(impact_dir))
    scoring = impact_index.BM25Scoring(k1=args.k1, b=args.b)
    scored_index = index.with_scoring(scoring, doc_meta)

    search_fn = (
        scored_index.search_wand
        if args.method == "wand"
        else scored_index.search_maxscore
    )

    # Load queries
    import ir_datasets

    dataset = ir_datasets.load(args.dataset)
    queries = []
    for i, query in enumerate(dataset.queries_iter()):
        if args.max_queries > 0 and i >= args.max_queries:
            break
        text = query.text if hasattr(query, "text") else str(query)
        queries.append(text)

    # Pre-analyze queries
    analyzed = [analyzer.analyze_query(q) for q in queries]
    analyzed = [q for q in analyzed if q]  # remove empty
    print(f"Loaded {len(analyzed)} queries")

    # Warmup
    for w in range(args.warmup):
        print(f"Warmup pass {w + 1}/{args.warmup}...")
        for q in analyzed:
            search_fn(q, args.top_k)

    # Timed passes
    print(
        f"\nProfiling {args.method.upper()} on {args.index} index ({args.passes} passes)..."
    )
    print(">>> Attach profiler now (e.g., sample <pid>) <<<")
    print(f"PID: {__import__('os').getpid()}")
    time.sleep(2)  # give time to attach profiler

    for p in range(args.passes):
        start = time.perf_counter()
        for q in analyzed:
            search_fn(q, args.top_k)
        elapsed = time.perf_counter() - start
        qps = len(analyzed) / elapsed
        print(f"  Pass {p + 1}: {elapsed:.2f}s ({qps:.1f} q/s)")


if __name__ == "__main__":
    main()
