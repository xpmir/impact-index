#!/usr/bin/env python3
# /// script
# requires-python = ">=3.9"
# dependencies = [
#     "numpy",
#     "impact-index",
# ]
# ///
"""
Debug script to compare WAND vs MaxScore search results.

This script helps identify which queries produce different results
between WAND and MaxScore algorithms.
"""

import argparse
import json
from pathlib import Path
from collections import defaultdict

import numpy as np


def load_index_and_queries(output_dir: Path):
    """Load index and query encodings from benchmark output."""
    import impact_index

    # Load the index
    index_dir = output_dir / "impact_index"
    if not index_dir.exists():
        raise FileNotFoundError(f"Index not found at {index_dir}")

    index = impact_index.Index.load(str(index_dir), in_memory=True)

    # Load doc_ids
    doc_ids_file = index_dir / "doc_ids.json"
    with open(doc_ids_file) as f:
        doc_ids = json.load(f)

    return index, doc_ids


def compare_results(index, query_encodings, top_k=100, num_queries=None):
    """Compare WAND and MaxScore results for each query."""
    differences = []

    if num_queries:
        query_encodings = query_encodings[:num_queries]

    for i, query in enumerate(query_encodings):
        if not query:
            continue

        wand_results = index.search_wand(query, top_k)
        maxscore_results = index.search_maxscore(query, top_k)

        # Convert to sets and dicts for comparison
        wand_docs = {h.docid: h.score for h in wand_results}
        maxscore_docs = {h.docid: h.score for h in maxscore_results}

        wand_doc_set = set(wand_docs.keys())
        maxscore_doc_set = set(maxscore_docs.keys())

        # Find differences
        only_wand = wand_doc_set - maxscore_doc_set
        only_maxscore = maxscore_doc_set - wand_doc_set

        if only_wand or only_maxscore:
            diff_info = {
                "query_idx": i,
                "only_wand": list(only_wand),
                "only_maxscore": list(only_maxscore),
                "wand_count": len(wand_results),
                "maxscore_count": len(maxscore_results),
            }

            # Check score differences for common docs
            common = wand_doc_set & maxscore_doc_set
            score_diffs = []
            for doc in common:
                w_score = wand_docs[doc]
                m_score = maxscore_docs[doc]
                if abs(w_score - m_score) > 1e-5:
                    score_diffs.append({
                        "doc": doc,
                        "wand_score": float(w_score),
                        "maxscore_score": float(m_score),
                        "diff": float(abs(w_score - m_score))
                    })

            if score_diffs:
                diff_info["score_diffs"] = score_diffs[:5]  # First 5

            differences.append(diff_info)

            if len(differences) <= 5:
                print(f"\nQuery {i}: WAND returned {len(wand_results)}, MaxScore returned {len(maxscore_results)}")
                if only_wand:
                    print(f"  Only in WAND ({len(only_wand)}): {list(only_wand)[:10]}...")
                    # Show scores of these docs in WAND
                    for doc in list(only_wand)[:3]:
                        print(f"    Doc {doc}: WAND score = {wand_docs[doc]:.6f}")
                if only_maxscore:
                    print(f"  Only in MaxScore ({len(only_maxscore)}): {list(only_maxscore)[:10]}...")
                if score_diffs:
                    print(f"  Score differences in common docs: {len(score_diffs)}")

    return differences


def detailed_query_comparison(index, query, top_k=100):
    """Do a detailed comparison for a single query."""
    print(f"\n=== Detailed Query Analysis ===")
    print(f"Query terms: {len(query)}")
    print(f"Query weights: min={min(query.values()):.4f}, max={max(query.values()):.4f}")

    wand_results = index.search_wand(query, top_k)
    maxscore_results = index.search_maxscore(query, top_k)

    print(f"\nWAND returned {len(wand_results)} results")
    print(f"MaxScore returned {len(maxscore_results)} results")

    # Compare top results
    print("\nTop 10 WAND results:")
    for i, h in enumerate(wand_results[:10]):
        print(f"  {i+1}. doc={h.docid}, score={h.score:.6f}")

    print("\nTop 10 MaxScore results:")
    for i, h in enumerate(maxscore_results[:10]):
        print(f"  {i+1}. doc={h.docid}, score={h.score:.6f}")

    # Check if same docs
    wand_docs = set(h.docid for h in wand_results)
    maxscore_docs = set(h.docid for h in maxscore_results)

    only_wand = wand_docs - maxscore_docs
    only_maxscore = maxscore_docs - wand_docs

    print(f"\nDocs only in WAND: {len(only_wand)}")
    print(f"Docs only in MaxScore: {len(only_maxscore)}")

    if only_wand:
        wand_map = {h.docid: h.score for h in wand_results}
        print("\nDocs only in WAND (sorted by score):")
        for doc in sorted(only_wand, key=lambda d: -wand_map[d])[:10]:
            print(f"  doc={doc}, score={wand_map[doc]:.6f}")

    return wand_results, maxscore_results


def main():
    parser = argparse.ArgumentParser(description="Debug WAND vs MaxScore differences")
    parser.add_argument("--output-dir", type=str, required=True, help="Benchmark output directory")
    parser.add_argument("--top-k", type=int, default=100, help="Number of results to retrieve")
    parser.add_argument("--num-queries", type=int, default=None, help="Number of queries to test")
    parser.add_argument("--query-idx", type=int, default=None, help="Analyze specific query index")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)

    print(f"Loading index from {output_dir}...")
    index, doc_ids = load_index_and_queries(output_dir)
    print(f"Index loaded from {output_dir}")

    # For testing, we need to encode queries or load them
    # Since this is a debug script, let's create some random queries
    # based on the index vocabulary

    # Try to load saved query encodings
    query_file = output_dir / "query_encodings.json"
    if query_file.exists():
        print(f"Loading query encodings from {query_file}...")
        with open(query_file) as f:
            query_data = json.load(f)
        query_encodings = [{int(k): v for k, v in q.items()} for q in query_data]
        print(f"Loaded {len(query_encodings)} queries")
    else:
        print("\nNote: No saved query encodings found.")
        print("Run benchmark with --save-queries to save encodings, or generating random queries...")

        # Generate random queries for testing
        np.random.seed(42)
        num_test_queries = args.num_queries or 100
        query_encodings = []

        vocab_size = index.num_postings()
        for _ in range(num_test_queries):
            # SPLADE-like queries: 50-150 terms with sparse weights
            num_terms = np.random.randint(50, 150)
            terms = np.random.choice(min(vocab_size, 30000), size=num_terms, replace=False)
            # Use log-normal distribution for weights (like SPLADE)
            weights = np.random.lognormal(0, 1.0, size=num_terms).astype(np.float32)
            # Cap to avoid very large weights
            weights = np.minimum(weights, 10.0)
            query = {int(t): float(w) for t, w in zip(terms, weights)}
            query_encodings.append(query)

    if args.query_idx is not None:
        if args.query_idx < len(query_encodings):
            detailed_query_comparison(index, query_encodings[args.query_idx], args.top_k)
        else:
            print(f"Query index {args.query_idx} out of range")
    else:
        print(f"\nComparing {len(query_encodings)} queries...")
        differences = compare_results(index, query_encodings, args.top_k, args.num_queries)

        print(f"\n=== Summary ===")
        print(f"Total queries with differences: {len(differences)}")
        if differences:
            avg_only_wand = np.mean([len(d["only_wand"]) for d in differences])
            avg_only_maxscore = np.mean([len(d["only_maxscore"]) for d in differences])
            print(f"Average docs only in WAND: {avg_only_wand:.1f}")
            print(f"Average docs only in MaxScore: {avg_only_maxscore:.1f}")


if __name__ == "__main__":
    main()
