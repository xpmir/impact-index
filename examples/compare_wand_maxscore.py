#!/usr/bin/env python3
# /// script
# requires-python = ">=3.9"
# dependencies = [
#     "ir-datasets",
#     "torch",
#     "transformers",
#     "numpy",
#     "tqdm",
# ]
# ///
"""
Compare WAND vs MaxScore on the actual SPLADE benchmark queries.
"""

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm
from transformers import AutoModelForMaskedLM, AutoTokenizer


def get_best_device():
    if torch.cuda.is_available():
        return "cuda"
    elif torch.backends.mps.is_available():
        return "mps"
    return "cpu"


class SpladeEncoder:
    def __init__(self, model_name="naver/splade_v2_max"):
        self.device = get_best_device()
        print(f"Loading SPLADE model '{model_name}' on {self.device}...")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
        self.model = AutoModelForMaskedLM.from_pretrained(model_name)
        self.model.to(self.device)
        self.model.eval()

    @torch.no_grad()
    def encode(self, texts, max_length=256):
        inputs = self.tokenizer(
            texts, return_tensors="pt", padding=True,
            truncation=True, max_length=max_length
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        outputs = self.model(**inputs)
        logits = outputs.logits
        weights = torch.max(
            torch.log1p(torch.relu(logits)) * inputs["attention_mask"].unsqueeze(-1),
            dim=1,
        ).values

        results = []
        for i in range(weights.shape[0]):
            sparse_vec = {}
            nonzero = weights[i].nonzero(as_tuple=True)[0]
            for idx in nonzero:
                term_id = idx.item()
                weight = weights[i, term_id].item()
                if weight > 0:
                    sparse_vec[term_id] = weight
            results.append(sparse_vec)
        return results

    def encode_batch(self, texts, batch_size=32):
        results = []
        for i in tqdm(range(0, len(texts), batch_size), desc="Encoding"):
            batch = texts[i:i + batch_size]
            results.extend(self.encode(batch))
        return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=str, required=True)
    parser.add_argument("--dataset", type=str, default="msmarco-passage/dev/small")
    parser.add_argument("--model", type=str, default="naver/splade_v2_max")
    parser.add_argument("--max-queries", type=int, default=0)
    parser.add_argument("--top-k", type=int, default=100)
    args = parser.parse_args()

    import impact_index
    import ir_datasets

    output_dir = Path(args.output_dir)
    index_dir = output_dir / "impact_index"

    print(f"Loading index from {index_dir}...")
    index = impact_index.Index.load(str(index_dir), in_memory=True)

    # Load queries
    print(f"Loading dataset '{args.dataset}'...")
    dataset = ir_datasets.load(args.dataset)

    queries = []
    for i, query in enumerate(dataset.queries_iter()):
        if args.max_queries > 0 and i >= args.max_queries:
            break
        queries.append(query.text if hasattr(query, "text") else str(query))
    print(f"Loaded {len(queries)} queries")

    # Encode queries
    encoder = SpladeEncoder(args.model)
    query_encodings = encoder.encode_batch(queries, batch_size=32)

    # Save for later use
    query_file = output_dir / "query_encodings.json"
    with open(query_file, "w") as f:
        json.dump([{str(k): v for k, v in q.items()} for q in query_encodings], f)
    print(f"Saved query encodings to {query_file}")

    # Compare results
    print("\nComparing WAND vs MaxScore...")
    differences = []
    total_only_wand = 0
    total_only_maxscore = 0
    ranking_diffs_top10 = 0
    ranking_diffs_top100 = 0

    for i, query in enumerate(tqdm(query_encodings, desc="Comparing")):
        if not query:
            continue

        wand_results = index.search_wand(query, args.top_k)
        maxscore_results = index.search_maxscore(query, args.top_k)

        # Check ranking differences (same docs, different positions)
        wand_ranking = [(h.docid, h.score) for h in wand_results]
        maxscore_ranking = [(h.docid, h.score) for h in maxscore_results]

        # Check top 10 ranking
        wand_top10 = [d for d, s in wand_ranking[:10]]
        maxscore_top10 = [d for d, s in maxscore_ranking[:10]]
        if wand_top10 != maxscore_top10:
            ranking_diffs_top10 += 1
            if ranking_diffs_top10 <= 5:
                print(f"\nQuery {i} - TOP 10 RANKING DIFFERS:")
                print(f"  WAND top10:     {wand_top10}")
                print(f"  MaxScore top10: {maxscore_top10}")
                # Show scores
                wand_scores = [s for d, s in wand_ranking[:10]]
                maxscore_scores = [s for d, s in maxscore_ranking[:10]]
                print(f"  WAND scores:     {[f'{s:.4f}' for s in wand_scores]}")
                print(f"  MaxScore scores: {[f'{s:.4f}' for s in maxscore_scores]}")

        wand_docs = set(h.docid for h in wand_results)
        maxscore_docs = set(h.docid for h in maxscore_results)

        only_wand = wand_docs - maxscore_docs
        only_maxscore = maxscore_docs - wand_docs

        if only_wand or only_maxscore:
            total_only_wand += len(only_wand)
            total_only_maxscore += len(only_maxscore)

            # Find positions of different docs
            wand_positions = {h.docid: pos for pos, h in enumerate(wand_results)}
            maxscore_positions = {h.docid: pos for pos, h in enumerate(maxscore_results)}

            min_wand_pos = min(wand_positions.get(d, 999) for d in only_wand) if only_wand else 999
            min_maxscore_pos = min(maxscore_positions.get(d, 999) for d in only_maxscore) if only_maxscore else 999

            differences.append({
                "query_idx": i,
                "only_wand": len(only_wand),
                "only_maxscore": len(only_maxscore),
                "wand_count": len(wand_results),
                "maxscore_count": len(maxscore_results),
                "min_diff_pos_wand": min_wand_pos,
                "min_diff_pos_maxscore": min_maxscore_pos,
            })

            if len(differences) <= 10:
                print(f"\nQuery {i}: WAND={len(wand_results)}, MaxScore={len(maxscore_results)}")
                print(f"  Diff positions: WAND@{min_wand_pos}, MaxScore@{min_maxscore_pos}")
                if only_wand:
                    wand_map = {h.docid: h.score for h in wand_results}
                    print(f"  Only in WAND ({len(only_wand)}): ", end="")
                    for doc in sorted(only_wand, key=lambda d: -wand_map[d])[:5]:
                        print(f"doc{doc}={wand_map[doc]:.4f}@{wand_positions[doc]} ", end="")
                    print()
                if only_maxscore:
                    maxscore_map = {h.docid: h.score for h in maxscore_results}
                    print(f"  Only in MaxScore ({len(only_maxscore)}): ", end="")
                    for doc in sorted(only_maxscore, key=lambda d: -maxscore_map[d])[:5]:
                        print(f"doc{doc}={maxscore_map[doc]:.4f}@{maxscore_positions[doc]} ", end="")
                    print()

    print(f"\n=== Summary ===")
    print(f"Total queries: {len(query_encodings)}")
    print(f"Queries with TOP 10 ranking differences: {ranking_diffs_top10}")
    print(f"Queries with document set differences: {len(differences)}")
    print(f"Total docs only in WAND: {total_only_wand}")
    print(f"Total docs only in MaxScore: {total_only_maxscore}")

    if differences:
        # Analyze positions of differences
        pos_wand = [d["min_diff_pos_wand"] for d in differences]
        pos_maxscore = [d["min_diff_pos_maxscore"] for d in differences]
        print(f"\nPosition analysis of document differences:")
        print(f"  Min position in WAND: {min(pos_wand)}, Max: {max(pos_wand)}")
        print(f"  Min position in MaxScore: {min(pos_maxscore)}, Max: {max(pos_maxscore)}")

    if differences:
        # Save detailed differences
        diff_file = output_dir / "wand_maxscore_diff.json"
        with open(diff_file, "w") as f:
            json.dump(differences, f, indent=2)
        print(f"Saved differences to {diff_file}")


if __name__ == "__main__":
    main()
