"""Incremental indexing demo: a collection that changes continuously.

impact-index indices are immutable, so a changing collection is kept as a
set of segments (see the "Incremental indexing" section of the README):

- the raw sparse vectors are the source of truth, so any segment can be
  rebuilt without re-encoding documents;
- new and modified documents go to a small buffer, searched exhaustively;
  when it is full it becomes a new immutable segment;
- deleted / modified documents are tombstoned in the segment holding them,
  and skipped at search time;
- all segments are compressed with the first segment's quantiser, so their
  scores are exactly comparable, and searched together with
  ``SegmentedIndex``;
- a merge rewrites everything live into one segment;
- a manifest listing the live segments, swapped atomically, makes the
  state crash-safe.

Run with: python examples/incremental.py
(requires `maturin develop --release` first -- see python/README.md)
"""

import json
import os
import tempfile

import numpy as np

import impact_index


class IncrementalIndex:
    def __init__(self, folder, buffer_size=200, nbits=8):
        self.folder = folder
        self.buffer_size = buffer_size
        self.nbits = nbits
        self.vectors = {}  # docid -> (terms, values): the source of truth
        self.buffer = set()  # docids whose live version is not in a segment
        self.segments = []  # [(name, Index)]
        self.tombstones = []  # one set of docids per segment
        self.location = {}  # docid -> segment position
        self.reference = None  # QuantizerParams shared by all segments
        self.counter = 0

    # --- Updates

    def upsert(self, docid, terms, values):
        self._tombstone(docid)
        self.vectors[docid] = (terms, values)
        self.buffer.add(docid)
        if len(self.buffer) >= self.buffer_size:
            self.flush()

    def delete(self, docid):
        self._tombstone(docid)
        self.vectors.pop(docid, None)
        self.buffer.discard(docid)

    def _tombstone(self, docid):
        if docid in self.location:
            self.tombstones[self.location.pop(docid)].add(docid)

    # --- Segments

    def _build(self, docids):
        self.counter += 1
        name = f"segment-{self.counter}"
        builder = impact_index.IndexBuilder(os.path.join(self.folder, name, "raw"))
        for docid in sorted(docids):
            builder.add(docid, *self.vectors[docid])
        segment = builder.build(True).compress(
            os.path.join(self.folder, name),
            nbits=self.nbits,
            # First segment: its own range; later ones reuse its quantiser
            quantization=self.reference,
        )
        if self.reference is None:
            self.reference = segment.quantizer()
        elif segment.clamped_impacts:
            print(f"  {segment.clamped_impacts} impacts clamped (merge to re-quantise)")
        return name, segment

    def flush(self):
        if not self.buffer:
            return
        self.segments.append(self._build(self.buffer))
        self.tombstones.append(set())
        for docid in self.buffer:
            self.location[docid] = len(self.segments) - 1
        self.buffer.clear()
        self._write_manifest()

    def merge(self):
        """Rewrites all live documents into one segment, with a fresh
        quantisation range."""
        self.reference = None
        live = set(self.vectors)
        self.segments = [self._build(live)]
        self.tombstones = [set()]
        self.location = {docid: 0 for docid in live}
        self.buffer.clear()
        self._write_manifest()

    def _write_manifest(self):
        # Atomic swap: readers see either the old or the new segment list.
        # (A real application also persists vectors and tombstones, and
        # removes the directories of segments that left the manifest.)
        manifest = {
            "segments": [name for name, _ in self.segments],
            "tombstones": [sorted(t) for t in self.tombstones],
        }
        tmp = os.path.join(self.folder, "manifest.json.tmp")
        with open(tmp, "w") as f:
            json.dump(manifest, f)
        os.replace(tmp, os.path.join(self.folder, "manifest.json"))

    # --- Search

    def search(self, query, top_k):
        hits = []
        if self.segments:
            segmented = impact_index.SegmentedIndex([s for _, s in self.segments])
            for hit in segmented.search(
                query, top_k, deleted=[sorted(t) or None for t in self.tombstones]
            ):
                hits.append((hit.docid, hit.score))
        # The buffer is small: score it exhaustively
        for docid in self.buffer:
            terms, values = self.vectors[docid]
            score = sum(
                float(v) * query.get(int(t), 0.0) for t, v in zip(terms, values)
            )
            if score > 0:
                hits.append((docid, score))
        hits.sort(key=lambda h: -h[1])
        return hits[:top_k]


def random_vector(rng, num_terms=1000):
    n = rng.randint(5, 30)
    terms = np.sort(rng.choice(num_terms, n, replace=False)).astype(np.uint64)
    return terms, rng.rand(n).astype(np.float32) * 2


def main():
    rng = np.random.RandomState(0)
    with tempfile.TemporaryDirectory() as folder:
        index = IncrementalIndex(folder)
        for docid in range(1000):
            index.upsert(docid, *random_vector(rng))
        for docid in rng.choice(1000, 100, replace=False):
            index.upsert(int(docid), *random_vector(rng))  # modify
        for docid in rng.choice(1000, 50, replace=False):
            index.delete(int(docid))

        query = {int(t): 1.0 for t in rng.choice(1000, 5, replace=False)}
        print(f"{len(index.segments)} segments, {len(index.buffer)} buffered docs")
        before = index.search(query, 5)
        for docid, score in before:
            print(f"  doc {docid}: {score:.4f}")

        index.merge()
        print(f"After merge: {len(index.segments)} segment")
        after = index.search(query, 5)
        for docid, score in after:
            print(f"  doc {docid}: {score:.4f}")


if __name__ == "__main__":
    main()
