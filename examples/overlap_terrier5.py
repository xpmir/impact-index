# /// script
# requires-python = ">=3.9"
# dependencies = [
#     "ir-datasets",
#     "tqdm",
#     "numpy",
#     "cbor2",
#     "snowballstemmer",
#     "python-terrier",
# ]
# ///
"""Result overlap: impact-index's terrier-aligned build vs real Terrier 5,
over the full MS MARCO passage dev/small query set. Both apply Terrier's
stop-word list before stemming, so this is the apples-to-apples fidelity
check that overlap.py (vs PISA) can't give -- PISA, as built by the
pyterrier_pisa wrapper used in this project's benchmarks, doesn't filter
stop words from its index or query scoring at all (see BENCHMARKS.md).

    uv run --with . --with python-terrier examples/overlap_terrier5.py \\
        --output-dir /tmp/bench

Writes ``<output-dir>/../overlap_terrier5.json``. Reuses benchmark.py's
cached index-loading plumbing (run benchmark.py first so the indices
exist); runs Terrier 5's own JVM in-process (no worker subprocess needed --
that isolation is only required to keep PyTerrier and Pyserini apart).
"""

import argparse
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import ir_datasets
import pyterrier as pt

import benchmark as B


def overlap_at_k(hits_a, hits_b, k):
    a = {d for d, _ in hits_a[:k]}
    b = {d for d, _ in hits_b[:k]}
    union = a | b
    if not union:
        return None
    return len(a & b) / len(union)


def mean(xs):
    xs = [x for x in xs if x is not None]
    return sum(xs) / len(xs) if xs else 0.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--output-dir", default="bench_output")
    args = ap.parse_args()
    output_dir = Path(args.output_dir)
    indices_root = output_dir / "indices"

    dataset = ir_datasets.load("msmarco-passage/dev/small")
    queries = B.load_queries(dataset, 0)
    print("n_queries", len(queries))

    ii_cfg = B.Config(
        "impact-index (compressed, MaxScore)",
        "impact-index",
        group="terrier-aligned",
        pipeline="terrier",
        stemmer="snowball",
        stop_words="terrier",
        algorithm="maxscore",
    )
    ii_index, _, _, _ = B.build_or_load_impact_index(
        ii_cfg, indices_root, dataset, force_rebuild=False
    )
    ii_handle = B.ImpactIndexHandle(ii_index)
    ii_scored = ii_handle._scored(ii_cfg.k1, ii_cfg.b, ii_cfg.bm25_variant)
    ii_analyzer = ii_handle._analyzer

    if not pt.java.started():
        pt.java.init()
    terrier_cfg = B.Config("Terrier 5 (PyTerrier)", "terrier", group="terrier-aligned")
    terrier_index_dir, _, _ = B.build_or_load_terrier_index(
        terrier_cfg, indices_root, force_rebuild=False
    )
    t_index = pt.IndexFactory.of(str(terrier_index_dir))
    retriever = pt.terrier.Retriever(
        t_index,
        wmodel="BM25",
        controls={"bm25.k_1": ii_cfg.k1, "bm25.b": ii_cfg.b},
        num_results=100,
    )

    ov10, ov100 = [], []
    for qid, text in queries:
        q = ii_analyzer.analyze_query(text)
        ii_hits = (
            []
            if not q
            else [(str(h.docid), h.score) for h in ii_scored.search_maxscore(q, 100)]
        )
        clean = re.sub(r"[^A-Za-z0-9 ]", " ", text).strip()
        if clean:
            hits = retriever.search(clean)
            t_hits = [(str(d), float(s)) for d, s in zip(hits["docno"], hits["score"])]
        else:
            t_hits = []
        ov10.append(overlap_at_k(ii_hits, t_hits, 10))
        ov100.append(overlap_at_k(ii_hits, t_hits, 100))

    result = {"overlap_at_10": mean(ov10), "overlap_at_100": mean(ov100)}
    print("impact-index[terrier] vs Terrier 5:", result)
    with open(output_dir.parent / "overlap_terrier5.json", "w") as f:
        json.dump(result, f)


if __name__ == "__main__":
    main()
