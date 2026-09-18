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
"""Structured (matchop) queries: impact-index vs real Terrier 5.

Builds (or reuses) a positional impact-index with Terrier's own analysis
(``pipeline="terrier", stemmer="porter"``, ``positions=True`` -- the
pipeline also implies Terrier's gap-free positions) scored with Terrier's
BM25 (``k3=8``), and a
Terrier 5 *block* index (``blocks=True``, needed for ``#1``/``#uwN``), then
runs the same matchop query families through both, BM25 everywhere, and
reports result overlap and throughput per family.

Query families are derived from MS MARCO dev/small: each query is reduced to
its unique non-stop-word words (so neither system has to drop anything
inside an operator), then:

- ``bow``     ``#combine(w1 .. wn)``                         (sanity baseline)
- ``phrase``  ``#combine(w1 .. wn #1(w1 w2) .. #1(wn-1 wn))``
- ``uw8``     ``#combine(w1 .. wn #uw8(w1 w2) ..)``
- ``syn``     ``#combine(#syn(w1 w2) w3 .. wn)``
- ``band``    ``#combine(w1 .. wn #band(w1 w2))``
- ``sdm``     ``#combine:0=0.85:1=0.1:2=0.05(#combine(uni) #combine(#1 ..) #combine(#uw8 ..))``

Only queries with >= 2 words enter the non-bow families.

    uv run --with . --with python-terrier examples/structured_terrier.py \\
        --output-dir ~/bench/bench_run_20260917

Writes ``<output-dir>/structured_terrier/<family>.json`` (metrics) and
``<family>.runs.jsonl`` (per-query top-k from both systems, for diagnosis).
"""

import argparse
import json
import re
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import impact_index
import ir_datasets
import pyterrier as pt
from tqdm import tqdm

import benchmark as B

FAMILIES = ["bow", "phrase", "uw8", "syn", "band", "sdm"]


def overlap_at_k(a, b, k):
    sa = {d for d, _ in a[:k]}
    sb = {d for d, _ in b[:k]}
    union = sa | sb
    return None if not union else len(sa & sb) / len(union)


def mean(xs):
    xs = [x for x in xs if x is not None]
    return sum(xs) / len(xs) if xs else 0.0


def pairs(ws):
    return list(zip(ws, ws[1:]))


def make_query(family, ws):
    uni = " ".join(ws)
    if family == "bow":
        return f"#combine({uni})"
    if len(ws) < 2:
        return None
    if family == "phrase":
        return f"#combine({uni} " + " ".join(f"#1({a} {b})" for a, b in pairs(ws)) + ")"
    if family == "uw8":
        return (
            f"#combine({uni} " + " ".join(f"#uw8({a} {b})" for a, b in pairs(ws)) + ")"
        )
    if family == "syn":
        return f"#combine(#syn({ws[0]} {ws[1]}) {' '.join(ws[2:])})".replace(" )", ")")
    if family == "band":
        return f"#combine({uni} #band({ws[0]} {ws[1]}))"
    if family == "sdm":
        ph = " ".join(f"#1({a} {b})" for a, b in pairs(ws))
        uw = " ".join(f"#uw8({a} {b})" for a, b in pairs(ws))
        return f"#combine:0=0.85:1=0.1:2=0.05(#combine({uni}) #combine({ph}) #combine({uw}))"
    raise ValueError(family)


def query_words(text, analyzer):
    """Unique lowercase alphanumeric words that survive impact-index's
    Terrier pipeline (i.e. not stop words, not dropped by the tokenizer)."""
    seen, out = set(), []
    for w in re.findall(r"[a-z0-9]+", text.lower()):
        if w in seen:
            continue
        seen.add(w)
        if analyzer.analyze_query(w):
            out.append(w)
    return out


def build_or_load_terrier_blocks(dataset, index_dir: Path):
    if (index_dir / "data.properties").exists():
        return index_dir
    index_dir.mkdir(parents=True, exist_ok=True)

    def doc_iter():
        for doc in tqdm(dataset.docs_iter(), desc="Indexing (Terrier, blocks)"):
            yield {"docno": doc.doc_id, "text": doc.text}

    start = time.perf_counter()
    pt.IterDictIndexer(
        str(index_dir),
        meta={"docno": 20},
        threads=4,
        blocks=True,
        type=pt.terrier.IndexingType.SINGLEPASS,
    ).index(doc_iter())
    B.write_json(
        index_dir / "build.json", {"build_seconds": time.perf_counter() - start}
    )
    return index_dir


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--output-dir", default="bench_output")
    ap.add_argument("--families", default=",".join(FAMILIES))
    ap.add_argument("--max-queries", type=int, default=0)
    ap.add_argument("--top-k", type=int, default=100)
    ap.add_argument("--warmup", type=int, default=300)
    ap.add_argument("--build-only", action="store_true")
    ap.add_argument("--skip-terrier", action="store_true")
    args = ap.parse_args()

    output_dir = Path(args.output_dir).expanduser()
    indices_root = output_dir / "indices"
    out = output_dir / "structured_terrier"
    out.mkdir(parents=True, exist_ok=True)

    dataset = ir_datasets.load("msmarco-passage/dev/small")
    if not pt.java.started():
        pt.java.init()

    ii_cfg = B.Config(
        "impact-index (positions, terrier/porter)",
        "impact-index",
        group="structured",
        pipeline="terrier",
        stemmer="porter",
        stop_words="terrier",
        positions=True,
        compression=B.Compression(nbits=0, block_size=128),
    )
    ii_index, ii_key, _, _ = B.build_or_load_impact_index(
        ii_cfg, indices_root, dataset, force_rebuild=False
    )
    t_dir = build_or_load_terrier_blocks(dataset, indices_root / "terrier-blocks")
    if args.build_only:
        return

    # k3=8: Terrier's BM25 query-term saturation (weights normalized by the
    # largest one, then (k3+1)w/(k3+w)) -- matters for #combine weights.
    scored = ii_index.with_scoring(
        impact_index.BM25Scoring(
            k1=ii_cfg.k1, b=ii_cfg.b, variant=ii_cfg.bm25_variant, k3=8.0
        )
    )
    analyzer = ii_index.analyzer()

    t_index = pt.IndexFactory.of(str(t_dir))
    retriever = pt.terrier.Retriever(
        t_index,
        wmodel="BM25",
        controls={"bm25.k_1": ii_cfg.k1, "bm25.b": ii_cfg.b},
        num_results=args.top_k,
    )

    queries = B.load_queries(dataset, args.max_queries)
    words = {qid: query_words(text, analyzer) for qid, text in queries}

    def ii_search(q):
        return [
            (str(h.docid), h.score) for h in scored.search_maxscore_query(q, args.top_k)
        ]

    def t_search(q):
        hits = retriever.search(q)
        return [(str(d), float(s)) for d, s in zip(hits["docno"], hits["score"])]

    for family in args.families.split(","):
        fq = [(qid, make_query(family, words[qid])) for qid, _ in queries]
        fq = [(qid, q) for qid, q in fq if q is not None and words[qid]]
        print(f"[{family}] {len(fq)} queries, e.g. {fq[0][1]!r}", flush=True)

        systems = [("impact-index", ii_search)]
        if not args.skip_terrier:
            systems.append(("terrier", t_search))
        runs, qps = {}, {}
        for name, fn in systems:
            for _, q in fq[: args.warmup]:
                fn(q)
            res = {}
            start = time.perf_counter()
            for qid, q in fq:
                res[qid] = fn(q)
            elapsed = time.perf_counter() - start
            runs[name] = res
            qps[name] = len(fq) / elapsed
            print(f"  {name}: {qps[name]:.1f} q/s", flush=True)

        metrics = {
            "family": family,
            "n_queries": len(fq),
            "qps": qps,
            "ii_index_key": ii_key,
        }
        if "terrier" in runs:
            a, b = runs["impact-index"], runs["terrier"]
            for k in (10, 100):
                metrics[f"overlap_at_{k}"] = mean(
                    overlap_at_k(a[q], b[q], k) for q, _ in fq
                )
            print(
                f"  overlap@10={metrics['overlap_at_10']:.3f} @100={metrics['overlap_at_100']:.3f}",
                flush=True,
            )
        B.write_json(out / f"{family}.json", metrics)
        with open(out / f"{family}.runs.jsonl", "w") as f:
            for qid, q in fq:
                f.write(
                    json.dumps(
                        {"qid": qid, "query": q, **{n: r[qid] for n, r in runs.items()}}
                    )
                    + "\n"
                )


if __name__ == "__main__":
    main()
