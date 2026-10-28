"""Tests for incremental indexing: filtered search, quantisation ranges and
segmented search (add / modify / delete over immutable segments)."""

import os

import numpy as np
import pytest

import impact_index

NUM_TERMS = 100


def random_doc(rng, max_value=2.0):
    n = rng.randint(3, 15)
    terms = np.sort(rng.choice(NUM_TERMS, n, replace=False)).astype(np.uint64)
    values = (0.01 + rng.rand(n) * (max_value - 0.01)).astype(np.float32)
    return terms, values


def random_query(rng):
    terms = rng.choice(NUM_TERMS, rng.randint(2, 7), replace=False)
    return {int(t): float(0.5 + rng.rand()) for t in terms}


def build(path, docs, nbits=8, quantization=None):
    """Builds and compresses an index over ``docs`` ({docid: (terms, values)})."""
    builder = impact_index.IndexBuilder(os.path.join(path, "raw"))
    for docid in sorted(docs):
        builder.add(docid, *docs[docid])
    raw = builder.build(True)
    return raw.compress(
        os.path.join(path, "compressed"), nbits=nbits, quantization=quantization
    )


def assert_same_top_k(got, expected):
    """Same score lists; same documents above the k-th score (ties at the
    k-th score may legitimately differ)."""
    assert [s for _, s in got] == [s for _, s in expected]
    if expected:
        kth = expected[-1][1]
        assert sorted(d for d, s in got if s > kth) == sorted(
            d for d, s in expected if s > kth
        )


@pytest.fixture
def rng():
    return np.random.RandomState(42)


class TestFilteredSearch:
    @pytest.fixture
    def index(self, tmp_path, rng):
        docs = {d: random_doc(rng) for d in range(300)}
        return build(str(tmp_path), docs)

    @pytest.mark.parametrize("method", ["maxscore", "wand"])
    def test_matches_post_filtering(self, index, rng, method):
        deleted = [d for d in range(300) if d % 3 == 0]
        mask = np.zeros(300, dtype=bool)
        mask[deleted] = True
        allowed = [d for d in range(300) if d % 4 != 1]

        search = getattr(index, f"search_{method}")
        filtered = getattr(index, f"search_{method}_filtered")
        for _ in range(20):
            q = random_query(rng)
            everything = [(r.docid, r.score) for r in search(q, 300)]
            for k in (1, 10, 30):
                expected = [r for r in everything if r[0] % 3 != 0][:k]
                for arg in (deleted, mask):
                    got = [(r.docid, r.score) for r in filtered(q, k, deleted=arg)]
                    assert_same_top_k(got, expected)

                expected = [r for r in everything if r[0] % 4 != 1][:k]
                got = [(r.docid, r.score) for r in filtered(q, k, allowed=allowed)]
                assert_same_top_k(got, expected)

                # No filter: same as the plain search
                got = [(r.docid, r.score) for r in filtered(q, k)]
                assert_same_top_k(got, everything[:k])

    def test_bad_filter(self, index):
        with pytest.raises(TypeError):
            index.search_maxscore_filtered({0: 1.0}, 10, deleted="nope")


class TestQuantization:
    DOCS = {
        0: (np.array([0, 1], dtype=np.uint64), np.array([0.5, 1.5], dtype=np.float32)),
        1: (np.array([0, 2], dtype=np.uint64), np.array([2.5, 1.0], dtype=np.float32)),
        2: (np.array([1], dtype=np.uint64), np.array([3.0], dtype=np.float32)),
    }

    def test_per_index(self, tmp_path):
        index = build(str(tmp_path), self.DOCS)
        params = index.quantizer()
        assert params.nbits == 8
        assert params.min <= 0.5
        assert params.max == pytest.approx(3.0)
        assert index.clamped_impacts == 0

    def test_fixed_range_clamps(self, tmp_path):
        index = build(str(tmp_path), self.DOCS, quantization=(0.0, 2.0))
        assert index.clamped_impacts == 2
        assert index.quantizer() == impact_index.QuantizerParams(8, 0.0, 2.0)
        top = index.search_maxscore({0: 1.0}, 1)[0]
        assert top.score == pytest.approx(2.0 - index.quantizer().step / 2, abs=1e-6)

    def test_reference(self, tmp_path):
        first = build(str(tmp_path / "a"), self.DOCS, nbits=16)
        # nbits comes from the reference
        other = build(
            str(tmp_path / "b"), self.DOCS, nbits=0, quantization=first.quantizer()
        )
        assert other.quantizer() == first.quantizer()
        assert other.quantizer().nbits == 16

    def test_errors(self, tmp_path):
        with pytest.raises(ValueError):
            build(str(tmp_path / "a"), self.DOCS, quantization=(2.0, 1.0))
        with pytest.raises(ValueError):
            build(str(tmp_path / "b"), self.DOCS, nbits=0, quantization=(0.0, 2.0))
        with pytest.raises(TypeError):
            build(str(tmp_path / "c"), self.DOCS, quantization="fixed")

    def test_lossless_has_no_quantizer(self, tmp_path):
        assert build(str(tmp_path), self.DOCS, nbits=0).quantizer() is None

    def test_compressor_class(self, tmp_path):
        builder = impact_index.IndexBuilder(str(tmp_path / "raw"))
        for docid in sorted(self.DOCS):
            builder.add(docid, *self.DOCS[docid])
        raw = builder.build(True)
        transform = impact_index.CompressionTransform(
            max_block_size=128,
            doc_ids_compressor=impact_index.EliasFanoCompressor(),
            impacts_compressor=impact_index.QuantizedBitPackedCompressor(
                8, quantization=(0.0, 4.0)
            ),
        )
        transform.process(str(tmp_path / "c"), raw)
        index = impact_index.Index.load(str(tmp_path / "c"), True)
        assert index.quantizer() == impact_index.QuantizerParams(8, 0.0, 4.0)


@pytest.mark.parametrize("mode", ["fixed", "reference"])
@pytest.mark.parametrize("algorithm", ["maxscore", "wand"])
def test_segments_match_from_scratch(tmp_path, mode, algorithm):
    """Random add / modify / delete over segments == one index over the
    live documents (exactly, with a common quantisation range)."""
    rng = np.random.RandomState(7)
    live = {}  # docid -> (terms, values)
    location = {}  # docid -> segment holding its live version
    segments, tombstones = [], []
    reference = None
    next_docid = 0

    for round in range(5):
        buffer = {}
        for _ in range(200 if round == 0 else rng.randint(30, 80)):
            op = rng.randint(10)
            if op < 6 or not live:
                next_docid += 1 + rng.randint(3)
                live[next_docid] = buffer[next_docid] = random_doc(rng)
            else:
                docid = int(rng.choice(list(live)))
                if docid in location:
                    tombstones[location.pop(docid)].append(docid)
                if op < 8:
                    live[docid] = buffer[docid] = random_doc(rng, 2.4)
                else:
                    del live[docid]
                    buffer.pop(docid, None)

        quantization = (0.0, 2.0) if mode == "fixed" else reference
        if buffer:
            seg = build(
                str(tmp_path / f"seg{round}"), buffer, quantization=quantization
            )
            reference = reference or seg.quantizer()
            for docid in buffer:
                location[docid] = len(segments)
            segments.append(seg)
            tombstones.append([])

        quantization = (0.0, 2.0) if mode == "fixed" else reference
        scratch = build(
            str(tmp_path / f"scratch{round}"), live, quantization=quantization
        )
        search_scratch = getattr(scratch, f"search_{algorithm}")
        segmented = impact_index.SegmentedIndex(segments)
        assert len(segmented) == len(segments)

        for _ in range(15):
            q = random_query(rng)
            for k in (1, 10, 50):
                expected = [(r.docid, r.score) for r in search_scratch(q, k)]
                for share in (True, False):
                    hits = segmented.search(
                        q,
                        k,
                        deleted=[t or None for t in tombstones],
                        algorithm=algorithm,
                        share_threshold=share,
                    )
                    for h in hits:
                        assert location[h.docid] == h.segment
                    assert_same_top_k([(h.docid, h.score) for h in hits], expected)


def test_segmented_errors(tmp_path, rng):
    index = build(str(tmp_path), {0: random_doc(rng)})
    segmented = impact_index.SegmentedIndex([index, index])
    with pytest.raises(ValueError):
        segmented.search({0: 1.0}, 10, deleted=[None])
    with pytest.raises(ValueError):
        segmented.search({0: 1.0}, 10, algorithm="bm25")
    with pytest.raises(TypeError):
        impact_index.SegmentedIndex([object()])
