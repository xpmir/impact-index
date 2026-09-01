"""Structured (Terrier-matchop-style) query demo.

Builds a small positional BM25 index, then searches it with structured
queries: #combine (weighted sum), #syn (synonym/OR), #band (boolean AND),
#1 (exact phrase), and #uwN (unordered window). Phrase/window operators
need an index built with BOWIndexBuilder(..., positions=True); the other
operators work on any index.

Run with: python examples/structured_queries.py
(requires `maturin develop --release` first -- see python/README.md)
"""

import tempfile

import impact_index

SENTENCES = [
    "the quick brown fox jumps over the lazy dog",
    "a quick brown cat naps near the lazy dog",
    "the slow red fox hides from the lazy dog",
    "quick foxes and lazy dogs rarely meet",
    "the lazy dog sleeps all day while the fox runs",
]


def show(scored_index, label, query):
    print(f"\n{label}: {query!r}")
    for doc in scored_index.search_wand_query(query, top_k=5):
        print(f"  doc {doc.docid}: {doc.score:.4f}")


def main():
    with tempfile.TemporaryDirectory() as index_dir:
        # positions=True is opt-in: it costs extra disk and is only needed
        # for #1 (phrase) / #uwN (window) queries below.
        builder = impact_index.BOWIndexBuilder(
            index_dir,
            stemmer="porter",
            stop_words=True,
            positions=True,
        )
        for docid, text in enumerate(SENTENCES):
            builder.add_text(docid, text)
        index = builder.build(in_memory=True)

        compressed = index.compress(f"{index_dir}/compressed")
        scored = compressed.with_scoring(impact_index.BM25Scoring())

        # 1. Matchop string: combine(quick, phrase(brown fox), band(lazy dog)).
        show(
            scored,
            "matchop string",
            "#combine(quick #1(brown fox) #band(lazy dog))",
        )

        # 2. Phrase-only matchop query.
        show(scored, "phrase only", "#1(lazy dog)")

        # 3. The equivalent nested-dict form of query 1. Structured (dict)
        #    queries take term ids directly -- resolve them via the index's
        #    own analyzer/vocabulary first.
        analyzer = compressed.analyzer()

        def term_id(word):
            return next(iter(analyzer.analyze_query(word)))

        nested_query = {
            "combine": [
                [1.0, {"term": term_id("quick")}],
                [1.0, {"phrase": [term_id("brown"), term_id("fox")]}],
                [
                    1.0,
                    {
                        "band": [
                            {"term": term_id("lazy")},
                            {"term": term_id("dog")},
                        ]
                    },
                ],
            ]
        }
        show(scored, "equivalent nested dict", nested_query)


if __name__ == "__main__":
    main()
