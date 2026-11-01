"""Tests for structured (matchop) query search, single and batched."""

import pytest

import impact_index

# Made-up documents (no real data)
DOCS = [
    "the quick brown fox jumps over the lazy dog",
    "a lazy cat sleeps all day",
    "the brown dog barks at the quick cat",
    "foxes and dogs are not cats",
    "quick quick quick fox",
    "a city guide for the budget traveller",
    "the brown fox and the brown dog",
    "dog dog dog cat",
]


def build(tmp_path, positions=False):
    builder = impact_index.BOWIndexBuilder(
        str(tmp_path / "bow"), stemmer="porter", stop_words=True, positions=positions
    )
    for docid, text in enumerate(DOCS):
        builder.add_text(docid, text)
    return builder.build(True)


def term_id(index, word):
    (ix,) = index.analyzer().analyze_query(word).keys()
    return ix


def key(results):
    return [(r.docid, round(r.score, 4)) for r in results]


def test_weighted_combine_equals_weighted_bow(tmp_path):
    # BM25 with k3=None is linear in query weights, so a weighted
    # #combine of plain terms scores exactly like the weighted BOW dict.
    index = build(tmp_path)
    scored = index.with_scoring(impact_index.BM25Scoring(k3=None))
    fox, dog, cat = (term_id(index, w) for w in ("fox", "dog", "cat"))
    bow = {fox: 2.0, dog: 0.5, cat: 1.25}

    combine_str = "#combine:0=2:1=0.5:2=1.25(fox dog cat)"
    combine_dict = {
        "combine": [[2.0, {"term": fox}], [0.5, {"term": dog}], [1.25, {"term": cat}]]
    }
    expected = scored.search_maxscore_many([bow], 10)[0]
    assert expected

    for got in scored.search_maxscore_query_many(
        [combine_str, combine_dict], 10
    ) + scored.search_wand_query_many([combine_str, combine_dict], 10):
        assert [r.docid for r in got] == [r.docid for r in expected]
        for g, e in zip(got, expected):
            assert g.score == pytest.approx(e.score, rel=1e-5)


@pytest.mark.parametrize("scoring", [True, False])
def test_many_equals_single(tmp_path, scoring):
    index = build(tmp_path, positions=True)
    target = index.with_scoring(impact_index.BM25Scoring()) if scoring else index
    fox, dog, brown = (term_id(index, w) for w in ("fox", "dog", "brown"))
    queries = [
        "#combine(quick #1(brown fox) #band(lazy dog))",
        fox,
        {"term": [dog, 2.0]},
        {"combine": [[1.0, {"syn": [fox, dog]}], [0.5, {"term": brown}]]},
        {"phrase": [brown, dog]},
        {"window": {"terms": [fox, brown], "width": 3}},
        "the a and",  # only stop words: no results
        "unknownword",  # out of vocabulary: no results
    ]
    for many, single in [
        (target.search_maxscore_query_many, target.search_maxscore_query),
        (target.search_wand_query_many, target.search_wand_query),
    ]:
        batch = many(queries, 5)
        assert len(batch) == len(queries)
        for q, rs in zip(queries, batch):
            assert key(rs) == key(single(q, 5))
        assert batch[-1] == [] and batch[-2] == []
        assert batch[0]
    assert target.search_maxscore_query_many([], 5) == []


def test_many_reports_failing_query(tmp_path):
    index = build(tmp_path)  # no positions
    scored = index.with_scoring(impact_index.BM25Scoring())
    fox = term_id(index, "fox")

    # Conversion error: raised before searching, names the position
    with pytest.raises(ValueError, match=r"^query 1: unknown query operator 'nope'"):
        scored.search_maxscore_query_many([fox, {"nope": 1}], 5)
    with pytest.raises(ValueError, match=r"^query 0: query parse error"):
        scored.search_maxscore_query_many(["#combine((", fox], 5)
    # Evaluation error (phrase needs positions)
    with pytest.raises(
        ValueError, match=r"^query 2: index was built without positions"
    ):
        scored.search_wand_query_many([fox, "fox", "#1(brown fox)"], 5)
