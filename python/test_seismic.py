"""Tests for converting an index to Seismic format (needs the `seismic` feature)."""

import os

import numpy as np
import pytest

import impact_index

pytestmark = pytest.mark.skipif(
    not hasattr(impact_index, "SeismicSearcher"),
    reason="impact_index built without the `seismic` feature",
)


def build_raw_index(tmpdir, rng, num_docs=200, num_terms=40):
    builder = impact_index.IndexBuilder(tmpdir)
    for doc_id in range(num_docs):
        n = rng.randint(5, 15)
        terms = rng.choice(num_terms, n, replace=False).astype(np.uint64)
        values = np.abs(rng.randn(n)).astype(np.float32) + 0.01
        builder.add(doc_id, terms, values)
    return builder.build(True)


def test_to_seismic_exhaustive_matches_maxscore(tmp_path):
    rng = np.random.RandomState(77)
    raw_dir = str(tmp_path / "raw")
    os.makedirs(raw_dir)
    index = build_raw_index(raw_dir, rng)

    seismic_dir = str(tmp_path / "seismic")
    # Keep all postings and full summaries: search is then exhaustive
    index.to_seismic(
        seismic_dir, n_postings=1_000_000, summary_energy=1.0, max_fraction=1000.0
    )
    searcher = impact_index.SeismicSearcher(seismic_dir)
    assert searcher.num_documents() == 200

    for _ in range(10):
        terms = rng.choice(40, 4, replace=False)
        query = {int(t): float(rng.uniform(0.1, 2.0)) for t in terms}
        expected = index.search_maxscore(query, 10)
        observed = searcher.search(query, 10, query_cut=len(query), heap_factor=0.0)
        assert len(observed) == len(expected)
        for e, o in zip(expected, observed):
            assert o.score == pytest.approx(e.score, rel=1e-2, abs=1e-2)
