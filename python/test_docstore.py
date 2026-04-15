"""Tests for DocumentStore, DocumentStoreBuilder, and Document."""

import os

import pytest

import impact_index


class TestDocumentStore:
    def test_build_and_load(self, tmp_path):
        store_dir = str(tmp_path / "store")
        os.makedirs(store_dir)

        builder = impact_index.DocumentStoreBuilder(store_dir)
        builder.add({"id": "doc0", "title": "First"}, b"Hello world")
        builder.add({"id": "doc1", "title": "Second"}, b"Goodbye world")
        builder.add({"id": "doc2", "title": "Third"}, b"Third doc")
        builder.build()

        store = impact_index.DocumentStore.load(store_dir)
        assert store.num_documents() == 3

    def test_key_names(self, tmp_path):
        store_dir = str(tmp_path / "store")
        os.makedirs(store_dir)

        builder = impact_index.DocumentStoreBuilder(store_dir)
        builder.add({"id": "a", "author": "Alice"}, b"content")
        builder.build()

        store = impact_index.DocumentStore.load(store_dir)
        keys = store.key_names()
        assert "id" in keys
        assert "author" in keys

    def test_get_by_number(self, tmp_path):
        store_dir = str(tmp_path / "store")
        os.makedirs(store_dir)

        builder = impact_index.DocumentStoreBuilder(store_dir)
        builder.add({"id": "doc0"}, b"content0")
        builder.add({"id": "doc1"}, b"content1")
        builder.build()

        store = impact_index.DocumentStore.load(store_dir)
        docs = store.get_by_number([0, 1])
        assert len(docs) == 2
        assert docs[0].content == b"content0"
        assert docs[1].content == b"content1"

    def test_get_by_key(self, tmp_path):
        store_dir = str(tmp_path / "store")
        os.makedirs(store_dir)

        builder = impact_index.DocumentStoreBuilder(store_dir)
        builder.add({"id": "alpha"}, b"data_alpha")
        builder.add({"id": "beta"}, b"data_beta")
        builder.build()

        store = impact_index.DocumentStore.load(store_dir)
        docs = store.get_by_key("id", ["beta", "alpha", "missing"])
        assert len(docs) == 3
        assert docs[0] is not None
        assert docs[0].content == b"data_beta"
        assert docs[1] is not None
        assert docs[1].content == b"data_alpha"
        assert docs[2] is None

    def test_document_properties(self, tmp_path):
        store_dir = str(tmp_path / "store")
        os.makedirs(store_dir)

        builder = impact_index.DocumentStoreBuilder(store_dir)
        builder.add({"id": "x", "lang": "en"}, b"test content")
        builder.build()

        store = impact_index.DocumentStore.load(store_dir)
        docs = store.get_by_number([0])
        doc = docs[0]
        assert doc.internal_id == 0
        assert doc.keys["id"] == "x"
        assert doc.keys["lang"] == "en"
        assert doc.content == b"test content"

    def test_builder_double_build_raises(self, tmp_path):
        store_dir = str(tmp_path / "store")
        os.makedirs(store_dir)

        builder = impact_index.DocumentStoreBuilder(store_dir)
        builder.add({"id": "a"}, b"data")
        builder.build()
        with pytest.raises(RuntimeError):
            builder.build()

    def test_custom_block_size_and_zstd(self, tmp_path):
        store_dir = str(tmp_path / "store")
        os.makedirs(store_dir)

        builder = impact_index.DocumentStoreBuilder(
            store_dir, block_size=2048, zstd_level=1
        )
        builder.add({"id": "a"}, b"data")
        builder.build()

        store = impact_index.DocumentStore.load(store_dir)
        assert store.num_documents() == 1

    def test_checkpoint_and_resume(self, tmp_path):
        store_dir = str(tmp_path / "store")
        os.makedirs(store_dir)

        builder = impact_index.DocumentStoreBuilder(
            store_dir, block_size=64, checkpoint_frequency=5
        )
        for i in range(12):
            builder.add({"id": f"DOC-{i:03d}"}, f"c{i}".encode())
        builder.checkpoint()
        assert builder.num_documents() == 12
        del builder  # simulate exit without build()

        builder = impact_index.DocumentStoreBuilder(
            store_dir, block_size=64, checkpoint_frequency=5
        )
        assert builder.num_documents() == 12
        for i in range(12, 20):
            builder.add({"id": f"DOC-{i:03d}"}, f"c{i}".encode())
        builder.build()

        store = impact_index.DocumentStore.load(store_dir)
        assert store.num_documents() == 20
        docs = store.get_by_key("id", ["DOC-000", "DOC-010", "DOC-019"])
        assert all(d is not None for d in docs)
        assert docs[0].content == b"c0"
        assert docs[2].content == b"c19"


class TestAsyncDocumentStore:
    @pytest.mark.asyncio
    async def test_aio_get_by_number(self, tmp_path):
        store_dir = str(tmp_path / "store")
        os.makedirs(store_dir)

        builder = impact_index.DocumentStoreBuilder(store_dir)
        builder.add({"id": "d0"}, b"async_content")
        builder.build()

        store = impact_index.DocumentStore.load(store_dir)
        docs = await store.aio_get_by_number([0])
        assert len(docs) == 1
        assert docs[0].content == b"async_content"

    @pytest.mark.asyncio
    async def test_aio_get_by_key(self, tmp_path):
        store_dir = str(tmp_path / "store")
        os.makedirs(store_dir)

        builder = impact_index.DocumentStoreBuilder(store_dir)
        builder.add({"id": "k1"}, b"val1")
        builder.build()

        store = impact_index.DocumentStore.load(store_dir)
        docs = await store.aio_get_by_key("id", ["k1"])
        assert len(docs) == 1
        assert docs[0] is not None
