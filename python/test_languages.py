"""Text analysis for non-English languages (MIRACL languages in particular)"""

import impact_index
import pytest


def build(tmp_path, texts, **kwargs):
    builder = impact_index.BOWIndexBuilder(str(tmp_path), dtype="int32", **kwargs)
    builder.add_texts(list(enumerate(texts)))
    builder.build(False)
    return impact_index.Index.load(str(tmp_path), True)


def search(index, query):
    analyzer = index.analyzer()
    scored = index.with_scoring(impact_index.BM25Scoring())
    hits = scored.search_maxscore_many([analyzer.analyze_query(query)], 10)[0]
    return [hit.docid for hit in hits]


@pytest.mark.parametrize(
    "language,stemmer,stop_words,text,query",
    [
        ("ar", "snowball", True, "ذهب الأولاد إلى المدارس", "المدارس"),
        ("bn", "none", True, "আমি বাংলায় গান গাই", "গান"),
        ("fa", "none", True, "من به مدرسه می‌روم", "مدرسه"),
        ("hi", "none", True, "मैं स्कूल जा रहा हूँ", "स्कूल"),
        ("id", "none", True, "Anak-anak bermain di sekolah", "sekolah"),
        ("ja", "none", False, "東京は日本の首都です", "日本"),
        ("ko", "none", False, "서울은 한국의 수도입니다", "한국의"),
        ("sw", "none", False, "Watoto walicheza shuleni", "shuleni"),
        ("te", "none", True, "నేను పాఠశాలకు వెళ్తున్నాను", "పాఠశాలకు"),
        ("th", "none", True, "กรุงเทพเป็นเมืองหลวงของประเทศไทย", "เมือง"),
        ("yo", "none", False, "Àwọn ọmọdé ń ṣeré", "ọmọdé"),
        ("zh", "none", False, "北京是中国的首都", "中国"),
    ],
)
def test_language(tmp_path, language, stemmer, stop_words, text, query):
    index = build(
        tmp_path,
        [text, "unrelated"],
        stemmer=stemmer,
        language=language,
        stop_words=stop_words,
    )
    assert search(index, query) == [0]


def test_icu_segmentation(tmp_path):
    """Chinese is segmented into words, not characters"""
    index = build(tmp_path, ["北京是中国的首都"], stemmer="none", language="zh")
    # 北京, 是, 中国, 的, 首都
    assert index.num_postings() == 5
    # A character-level query does not match a word
    assert search(index, "中") == []


def test_explicit_tokenizer(tmp_path):
    index = build(
        tmp_path,
        ["北京是中国的首都"],
        stemmer="none",
        language="zh",
        tokenizer="standard",
    )
    # One token per character
    assert index.num_postings() == 8
    with pytest.raises(ValueError):
        impact_index.BOWIndexBuilder(str(tmp_path / "x"), tokenizer="nope")


def test_no_stemmer_no_stop_words_keeps_analyzer(tmp_path):
    """An analyzer without stemmer nor stop words survives reloading"""
    index = build(
        tmp_path, ["the cats", "dogs"], stemmer="none", language="sw", stop_words=False
    )
    assert search(index, "the") == [0]
    assert search(index, "cats") == [0]
    assert search(index, "cat") == []


def test_stop_words_iso_code(tmp_path):
    index = build(
        tmp_path,
        ["les enfants", "chiens"],
        stemmer="snowball",
        language="fr",
        stop_words=True,
    )
    assert search(index, "les") == []
    assert search(index, "enfant") == [0]


# Pyserini pipeline: Lucene analyzers as used by Anserini (made-up sentences;
# expected terms checked against Lucene 10.5.0)
@pytest.mark.parametrize(
    "language,text,query,expected",
    [
        ("fr", "Les enfants jouaient à l'école", "écoles", 0),
        ("de", "Die Häuser der Straße", "haus", 0),
        ("es", "Las luces de la ciudad", "luz", 0),
        ("ar", "ذهب الأولاد إلى المدارس", "مدرسة", None),
        ("hi", "मैं स्कूल जा रहा हूँ", "स्कूल", 0),
        ("zh", "北京是中国的首都", "中国", 0),
        ("sw", "Watoto walicheza shuleni.", "shuleni.", 0),
        ("th", "กรุงเทพเป็นเมืองหลวงของประเทศไทย", "กรุงเทพ", 0),
    ],
)
def test_pyserini_language(tmp_path, language, text, query, expected):
    index = build(tmp_path, [text, "unrelated"], pipeline="pyserini", language=language)
    hits = search(index, query)
    if expected is not None:
        assert hits == [expected]


def test_pyserini_chinese_bigrams(tmp_path):
    index = build(tmp_path, ["北京是中国的首都"], pipeline="pyserini", language="zh")
    # 7 bigrams
    assert index.num_postings() == 7


def test_pyserini_whitespace_case_sensitive(tmp_path):
    """Anserini's WhitespaceAnalyzer (e.g. Swahili) does not lowercase"""
    index = build(tmp_path, ["Watoto walicheza"], pipeline="pyserini", language="sw")
    assert search(index, "Watoto") == [0]
    assert search(index, "watoto") == []


def test_pyserini_unported_language(tmp_path):
    with pytest.raises(ValueError):
        impact_index.BOWIndexBuilder(str(tmp_path), pipeline="pyserini", language="it")


def test_prepare_analyzer_without_resources():
    assert impact_index.prepare_analyzer("fr", pipeline="pyserini") == []
    assert impact_index.prepare_analyzer("ja") == []


needs_dictionaries = pytest.mark.skipif(
    not __import__("os").environ.get("IMPACT_INDEX_TEST_DICTIONARIES"),
    reason="set IMPACT_INDEX_TEST_DICTIONARIES=1 to test (downloads dictionaries)",
)


@needs_dictionaries
@pytest.mark.parametrize(
    "language,text,query",
    [
        ("ja", "東京は日本の首都です", "日本"),
        ("ko", "서울은 한국의 수도입니다", "한국"),
    ],
)
def test_pyserini_morphological(tmp_path, language, text, query):
    (path,) = impact_index.prepare_analyzer(language, pipeline="pyserini")
    index = build(tmp_path, [text, "unrelated"], pipeline="pyserini", language=language)
    assert search(index, query) == [0]
    # Explicit dictionary path
    index = build(
        tmp_path / "explicit",
        [text, "unrelated"],
        pipeline="pyserini",
        language=language,
        dictionary=path,
    )
    assert search(index, query) == [0]
