"""Tests for BM25Scoring, DocMetadata, ScoredIndex, and BOWIndexBuilder."""

import numpy as np
import pytest

import impact_index


@pytest.fixture
def rng():
    return np.random.RandomState(99)


class TestBM25Scoring:
    def test_default_params(self):
        scoring = impact_index.BM25Scoring()
        # Should not raise
        assert scoring is not None

    def test_custom_params(self):
        scoring = impact_index.BM25Scoring(k1=0.9, b=0.4)
        assert scoring is not None


class TestBOWIndexBuilder:
    def test_build_with_manual_terms(self, rng, tmp_path):
        d = str(tmp_path / "bow")
        import os

        os.makedirs(d)

        builder = impact_index.BOWIndexBuilder(d, dtype="int32")
        for doc_id in range(50):
            n = rng.randint(3, 10)
            terms = rng.choice(20, n, replace=False).astype(np.uint64)
            values = rng.randint(1, 5, n).astype(np.float32)
            builder.add(doc_id, terms, values)

        builder.build(True)
        doc_meta = impact_index.DocMetadata.load(d)
        assert doc_meta.num_docs() == 50
        assert doc_meta.avg_dl() > 0
        assert doc_meta.min_dl() > 0

    def test_scored_search(self, rng, tmp_path):
        d = str(tmp_path / "bow")
        import os

        os.makedirs(d)

        builder = impact_index.BOWIndexBuilder(d, dtype="int32")
        for doc_id in range(50):
            n = rng.randint(3, 10)
            terms = rng.choice(20, n, replace=False).astype(np.uint64)
            values = rng.randint(1, 5, n).astype(np.float32)
            builder.add(doc_id, terms, values)

        index = builder.build(True)
        scoring = impact_index.BM25Scoring(k1=1.2, b=0.75)
        scored = index.with_scoring(scoring)

        results = scored.search_wand({0: 1.0}, 10)
        assert len(results) <= 10
        assert all(r.score > 0 for r in results)

        results_ms = scored.search_maxscore({0: 1.0}, 10)
        assert len(results_ms) <= 10

    def test_add_text_with_stemmer(self, tmp_path):
        d = str(tmp_path / "bow")
        import os

        os.makedirs(d)

        builder = impact_index.BOWIndexBuilder(
            d, dtype="int32", stemmer="snowball", language="english"
        )
        builder.add_text(0, "the cat is running quickly")
        builder.add_text(1, "dogs run faster than cats")
        builder.add_text(2, "running is a good exercise")

        query = builder.analyze_query("running cats")
        assert len(query) > 0

        index = builder.build(True)
        scoring = impact_index.BM25Scoring()
        scored = index.with_scoring(scoring)
        results = scored.search_wand(query, 10)
        assert len(results) > 0

    def test_text_analyzer_load(self, tmp_path):
        d = str(tmp_path / "bow_analyzer")
        import os

        os.makedirs(d)

        builder = impact_index.BOWIndexBuilder(
            d, dtype="int32", stemmer="snowball", language="english"
        )
        builder.add_text(0, "the cat is running quickly")
        builder.add_text(1, "dogs run faster than cats")
        builder.add_text(2, "running is a good exercise")
        builder.build(False)

        # Load analyzer from built index
        analyzer = impact_index.TextAnalyzer.load(
            d, stemmer="snowball", language="english"
        )
        query = analyzer.analyze_query("running cats")
        assert len(query) > 0

        # Unknown terms should be skipped
        query_unknown = analyzer.analyze_query("xyzzyplugh")
        assert len(query_unknown) == 0

        # Search with loaded analyzer's query
        index = impact_index.Index.load(d, True)
        scored = index.with_scoring(impact_index.BM25Scoring())
        results = scored.search_wand(query, 10)
        assert len(results) > 0

    def test_doc_metadata_copy_files(self, rng, tmp_path):
        src = str(tmp_path / "src")
        dst = str(tmp_path / "dst")
        import os

        os.makedirs(src)
        os.makedirs(dst)

        builder = impact_index.BOWIndexBuilder(src, dtype="int32")
        for doc_id in range(10):
            terms = np.array([0, 1], dtype=np.uint64)
            values = np.array([1.0, 2.0], dtype=np.float32)
            builder.add(doc_id, terms, values)
        builder.build(True)

        impact_index.DocMetadata.copy_files(src, dst)
        meta = impact_index.DocMetadata.load(dst)
        assert meta.num_docs() == 10
