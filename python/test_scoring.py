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

    def test_stop_words_true_matches_lucene(self, tmp_path):
        # `stop_words=True` is a back-compat alias for `stop_words="lucene"`
        # and must stay that way -- see get_stop_words tests below and
        # src/vocab/stopwords.rs::test_lucene_family_matches_get_stop_words.
        text = "the cat and the dog are running in the wide open field"

        d_true = str(tmp_path / "bow_true")
        import os

        os.makedirs(d_true)
        b_true = impact_index.BOWIndexBuilder(
            d_true,
            dtype="int32",
            stemmer="snowball",
            language="english",
            stop_words=True,
        )
        b_true.add_text(0, text)
        index_true = b_true.build(True)

        d_lucene = str(tmp_path / "bow_lucene")
        os.makedirs(d_lucene)
        b_lucene = impact_index.BOWIndexBuilder(
            d_lucene,
            dtype="int32",
            stemmer="snowball",
            language="english",
            stop_words="lucene",
        )
        b_lucene.add_text(0, text)
        index_lucene = b_lucene.build(True)

        # Same vocabulary size: both builds dropped exactly the same words.
        assert index_true.num_postings() == index_lucene.num_postings()

        analyzer_true = impact_index.TextAnalyzer.from_index(d_true)
        analyzer_lucene = impact_index.TextAnalyzer.from_index(d_lucene)
        # "the" and "are" are Lucene stop words; both builds must drop them.
        assert analyzer_true.analyze_query("the are") == {}
        assert analyzer_lucene.analyze_query("the are") == {}
        assert analyzer_true.analyze_query(text) == analyzer_lucene.analyze_query(text)

    def test_stop_words_terrier_is_a_superset_of_lucene(self, tmp_path):
        # "however", "particular", "several" are in Terrier's list but not
        # Lucene's 33-word list -- verified against
        # src/vocab/stopwords/{terrier,lucene}/english*.txt.
        #
        # Terrier's family filters post-stem only, against the RAW word
        # list (matching real PISA -- see `StopWordFilterMode::PostStem`),
        # so a word is only actually caught if its own stem happens to
        # equal its raw form. "particular" stems to itself ("particular"),
        # so it's still caught. "however" and "several" stem to "howev"
        # and "sever" respectively -- neither matches the raw list entry,
        # so PISA's real algorithm (faithfully reproduced here) does NOT
        # remove them, even though they're nominally "in the list". This
        # is an intentional, documented quirk of matching PISA exactly, not
        # a bug.
        text = (
            "the cat and the dog are however running in a particular "
            "and several wide open field"
        )

        d_lucene = str(tmp_path / "bow_lucene2")
        import os

        os.makedirs(d_lucene)
        b_lucene = impact_index.BOWIndexBuilder(
            d_lucene,
            dtype="int32",
            stemmer="snowball",
            language="english",
            stop_words="lucene",
        )
        b_lucene.add_text(0, text)
        index_lucene = b_lucene.build(True)

        d_terrier = str(tmp_path / "bow_terrier")
        os.makedirs(d_terrier)
        b_terrier = impact_index.BOWIndexBuilder(
            d_terrier,
            dtype="int32",
            stemmer="snowball",
            language="english",
            stop_words="terrier",
        )
        b_terrier.add_text(0, text)
        index_terrier = b_terrier.build(True)

        # Terrier's much longer list removes more words than Lucene's on
        # this sentence (it drops "particular" too, which Lucene doesn't).
        assert index_terrier.num_postings() < index_lucene.num_postings()

        # Reload each from its saved config and confirm the family that was
        # actually used survives a reload (not silently falling back).
        analyzer_terrier = impact_index.TextAnalyzer.from_index(d_terrier)
        # "particular" stems to itself, so Terrier's post-stem-against-raw
        # check catches it.
        assert analyzer_terrier.analyze_query("particular") == {}
        # "however" stems to "howev", which isn't in the raw list -- PISA's
        # literal algorithm does not catch this, and neither do we.
        assert len(analyzer_terrier.analyze_query("however")) > 0
        analyzer_lucene = impact_index.TextAnalyzer.from_index(d_lucene)
        assert len(analyzer_lucene.analyze_query("however")) > 0

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
