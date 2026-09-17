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

    def test_variant_bm25_and_lucene(self):
        # "bm25" (the true/Robertson formula) is the default; "lucene" is
        # Lucene's BM25Similarity.idf variant. Both must construct fine.
        assert impact_index.BM25Scoring(variant="bm25") is not None
        assert impact_index.BM25Scoring(variant="lucene") is not None

    def test_variant_rejects_unknown_value(self):
        with pytest.raises(ValueError):
            impact_index.BM25Scoring(variant="not-a-real-variant")


class TestPipeline:
    def test_terrier_pipeline_truncates_at_first_apostrophe(self, tmp_path):
        # PISA's own tokenizer (unlike Lucene's) truncates ANY contraction
        # or possessive at the first apostrophe, not just a trailing 's --
        # see Tokenizer::PisaEnglish. "don't" -> "don", "king's" -> "king".
        d = str(tmp_path / "terrier")
        import os

        os.makedirs(d)
        b = impact_index.BOWIndexBuilder(d, dtype="int32", pipeline="terrier")
        b.add_text(0, "don't worry king's castle it's fine")
        b.build(False)
        analyzer = impact_index.TextAnalyzer.from_index(d)
        assert len(analyzer.analyze_query("don king")) == 2

    def test_pyserini_pipeline_keeps_only_trailing_possessive(self, tmp_path):
        # Lucene's EnglishPossessiveFilter only strips a trailing 's --
        # "don't" is left whole (not stemmed down to "don"), "king's" -> "king".
        d = str(tmp_path / "pyserini")
        import os

        os.makedirs(d)
        b = impact_index.BOWIndexBuilder(d, dtype="int32", pipeline="pyserini")
        b.add_text(0, "don't worry king's castle it's fine")
        b.build(False)
        analyzer = impact_index.TextAnalyzer.from_index(d)
        assert len(analyzer.analyze_query("don king")) == 1

    def test_pipeline_composes_with_explicit_stemmer(self, tmp_path):
        # pipeline="terrier" + stemmer="porter": terrier's tokenizer/stop
        # words, but the Porter stemmer instead of terrier's own Snowball
        # default -- matches real Terrier 5's own (classic Porter) stemmer.
        d = str(tmp_path / "terrier_porter")
        import os

        os.makedirs(d)
        b = impact_index.BOWIndexBuilder(
            d, dtype="int32", pipeline="terrier", stemmer="porter"
        )
        b.add_text(0, "the cat and the dog are running, don't stop")
        b.build(False)
        analyzer = impact_index.TextAnalyzer.from_index(d)
        # Terrier's list (unlike Lucene's short one) covers "and".
        assert analyzer.analyze_query("and") == {}
        # Terrier's tokenizer (not Lucene's) still applies: "don't" -> "don".
        assert len(analyzer.analyze_query("don")) == 1

    def test_pipeline_with_none_stop_words_uses_pipeline_default(self, tmp_path):
        # Python can't distinguish "stop_words omitted" from
        # "stop_words=None explicitly passed" -- with `pipeline` set, both
        # mean "use that pipeline's own default stop-word list". Pass [] to
        # opt out of stop words entirely while still using the pipeline's
        # tokenizer/stemmer -- regression test for a real bug where the
        # "no stop words" ablation config silently got Lucene's stopwords
        # applied anyway because it passed `stop_words=None` alongside
        # `pipeline="pyserini"`.
        d_none = str(tmp_path / "none")
        import os

        os.makedirs(d_none)
        b_none = impact_index.BOWIndexBuilder(
            d_none, dtype="int32", pipeline="pyserini", stop_words=None
        )
        b_none.add_text(0, "the cat and the dog are running")
        b_none.build(False)
        analyzer_none = impact_index.TextAnalyzer.from_index(d_none)
        assert analyzer_none.analyze_query("the") == {}, (
            "stop_words=None with a pipeline set must use that pipeline's "
            "default list (Lucene's, here), not skip stop-word filtering"
        )

        d_empty = str(tmp_path / "empty")
        os.makedirs(d_empty)
        b_empty = impact_index.BOWIndexBuilder(
            d_empty, dtype="int32", pipeline="pyserini", stop_words=[]
        )
        b_empty.add_text(0, "the cat and the dog are running")
        b_empty.build(False)
        analyzer_empty = impact_index.TextAnalyzer.from_index(d_empty)
        assert analyzer_empty.analyze_query("the") != {}, (
            "stop_words=[] must explicitly disable stop-word filtering "
            "even with a pipeline set"
        )

    def test_unknown_pipeline_rejected(self, tmp_path):
        d = str(tmp_path / "bad_pipeline")
        import os

        os.makedirs(d)
        with pytest.raises(ValueError):
            impact_index.BOWIndexBuilder(d, dtype="int32", pipeline="nope")

    def test_terrier_pisa_pipeline_indexes_stop_words_but_query_filters_them(
        self, tmp_path
    ):
        # PISA's own index (as built by the pyterrier_pisa wrapper commonly
        # compared against) never filters stop words from what it stores,
        # at index time -- unlike real Terrier 5, which does (see
        # "terrier"). But PISA's own query processing DOES exclude them
        # (verified: querying it with only stop words returns no results).
        # "terrier-pisa" reproduces that asymmetry: same tokenizer/stemmer
        # as "terrier", nothing filtered at index time, but Terrier's list
        # still filtered at query time.
        text = "the cat and the dog are however running, don't stop"
        import os

        d_tp = str(tmp_path / "terrier_pisa")
        os.makedirs(d_tp)
        b_tp = impact_index.BOWIndexBuilder(
            d_tp, dtype="int32", pipeline="terrier-pisa"
        )
        b_tp.add_text(0, text)
        index_tp = b_tp.build(True)

        d_t = str(tmp_path / "terrier")
        os.makedirs(d_t)
        b_t = impact_index.BOWIndexBuilder(d_t, dtype="int32", pipeline="terrier")
        b_t.add_text(0, text)
        index_t = b_t.build(True)

        # terrier-pisa's index keeps stop words ("the", "and", "however",
        # ...); terrier's index drops them -- so terrier-pisa's vocab is
        # strictly larger for the same text.
        assert index_tp.num_postings() > index_t.num_postings()

        # But querying terrier-pisa still filters stop words out of the
        # query, same as terrier does.
        analyzer = impact_index.TextAnalyzer.from_index(d_tp)
        assert analyzer.analyze_query("the") == {}
        assert analyzer.analyze_query("and") == {}
        assert analyzer.analyze_query("however") == {}
        assert analyzer.analyze_query("cat") != {}
        # PISA's tokenizer still applies: "don't" -> "don".
        assert len(analyzer.analyze_query("don")) == 1

    def test_terrier_pisa_pipeline_stop_words_override_is_index_time_only(
        self, tmp_path
    ):
        # An explicit stop_words= overrides the pipeline's own (empty)
        # INDEX-time default, same as any other pipeline. Query-time
        # filtering is a fixed characteristic of matching PISA's own query
        # behavior, not affected by this override -- see the `pipeline=`
        # docstring.
        d = str(tmp_path / "terrier_pisa_custom")
        import os

        os.makedirs(d)
        b = impact_index.BOWIndexBuilder(
            d, dtype="int32", pipeline="terrier-pisa", stop_words=["cat"]
        )
        b.add_text(0, "the cat and the dog are running")
        index = b.build(True)
        # "cat" was excluded from indexing by the custom list; "the"/"and"
        # weren't (terrier-pisa's own index-time default is empty).
        analyzer = index.analyzer()
        assert analyzer.analyze_query("cat") == {}
        # Query-time Terrier filtering still applies regardless.
        assert analyzer.analyze_query("the") == {}
        assert analyzer.analyze_query("and") == {}
        assert analyzer.analyze_query("dog") != {}


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
        # Terrier's family filters pre-stem only, against the RAW word list
        # (matching real Terrier 5's default `Stopwords,PorterStemmer`
        # termpipeline -- stop words are checked once, before stemming --
        # see `StopWordFilterMode::PreStem`), so all three are caught
        # regardless of how they stem.
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
        # All three are checked pre-stem, against the raw list, so all are
        # caught regardless of how they'd otherwise stem.
        assert analyzer_terrier.analyze_query("particular") == {}
        assert analyzer_terrier.analyze_query("however") == {}
        assert analyzer_terrier.analyze_query("several") == {}
        # Lucene's shorter list doesn't contain any of them.
        analyzer_lucene = impact_index.TextAnalyzer.from_index(d_lucene)
        assert len(analyzer_lucene.analyze_query("however")) > 0
        assert len(analyzer_lucene.analyze_query("particular")) > 0
        assert len(analyzer_lucene.analyze_query("several")) > 0

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
