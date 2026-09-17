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
"""Result overlap: an impact-index build vs PISA, over the full MS MARCO
passage dev/small query set. Reuses benchmark.py's cached index-loading
plumbing (run benchmark.py first so the indices exist).

    uv run --with . --with pyterrier-pisa examples/overlap.py \\
        --output-dir /tmp/bench --pipeline terrier-pisa

``--pipeline`` selects which impact-index build to compare (default
"terrier-pisa", the one actually designed for PISA fidelity -- see
BENCHMARKS.md; pass "terrier" to instead measure the real-Terrier-5-aligned
build's overlap with PISA, expected to be much lower).

Writes ``<output-dir>/../overlap_<pipeline>_vs_pisa.json``.
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import ir_datasets

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


PIPELINE_STOP_WORDS = {"terrier": "terrier", "terrier-pisa": []}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--output-dir", default="bench_output")
    ap.add_argument(
        "--pipeline", default="terrier-pisa", choices=sorted(PIPELINE_STOP_WORDS)
    )
    args = ap.parse_args()
    output_dir = Path(args.output_dir)
    indices_root = output_dir / "indices"

    dataset = ir_datasets.load("msmarco-passage/dev/small")
    queries = B.load_queries(dataset, 0)
    print("n_queries", len(queries))

    # Raw (uncompressed) impact-index is fine -- results are identical to
    # compressed (lossless), and avoids re-touching the compressed cache.
    ii_cfg = B.Config(
        "impact-index (compressed, MaxScore)",
        "impact-index",
        group=f"{args.pipeline}-aligned",
        pipeline=args.pipeline,
        stemmer="snowball",
        stop_words=PIPELINE_STOP_WORDS[args.pipeline],
        algorithm="maxscore",
    )
    ii_index, _, _, _ = B.build_or_load_impact_index(
        ii_cfg, indices_root, dataset, force_rebuild=False
    )
    ii_handle = B.ImpactIndexHandle(ii_index)
    ii_scored = ii_handle._scored(ii_cfg.k1, ii_cfg.b, ii_cfg.bm25_variant)
    ii_analyzer = ii_handle._analyzer

    pisa_cfg = B.Config(
        "PISA (Block-Max WAND)",
        "pisa",
        group=f"{args.pipeline}-aligned",
        algorithm="block_max_wand",
    )
    pisa_index, _, _, _ = B.build_or_load_pisa_index(
        pisa_cfg, indices_root, dataset, force_rebuild=False
    )
    pisa_handle = B.PisaHandle(pisa_index)
    pisa_bm25 = pisa_handle._bm25(pisa_cfg)

    ov10, ov100 = [], []
    for qid, text in queries:
        q = ii_analyzer.analyze_query(text)
        ii_hits = (
            []
            if not q
            else [(str(h.docid), h.score) for h in ii_scored.search_maxscore(q, 100)]
        )
        pisa_rows = pisa_bm25.search(text)
        pisa_hits = [(str(r.docno), float(r.score)) for r in pisa_rows.itertuples()]
        ov10.append(overlap_at_k(ii_hits, pisa_hits, 10))
        ov100.append(overlap_at_k(ii_hits, pisa_hits, 100))

    result = {"overlap_at_10": mean(ov10), "overlap_at_100": mean(ov100)}
    print(f"impact-index[{args.pipeline}] vs PISA:", result)
    with open(output_dir.parent / f"overlap_{args.pipeline}_vs_pisa.json", "w") as f:
        json.dump(result, f)


if __name__ == "__main__":
    main()
