import pytest
from unittest.mock import MagicMock, patch, call
from langchain_core.documents import Document


@pytest.fixture
def mock_chroma():
    return MagicMock()


@pytest.fixture
def mock_embeddings():
    return MagicMock()


@pytest.fixture
def vector_store(mock_chroma, mock_embeddings):
    with patch("tools.vector_store.HuggingFaceEmbeddings", return_value=mock_embeddings), \
         patch("tools.vector_store.Chroma", return_value=mock_chroma):
        from tools.vector_store import VectorStore
        store = VectorStore(collection_name="test-collection")
        store._store = mock_chroma
        return store


class TestVectorStoreIngestData:
    def test_returns_list_of_string_ids(self, vector_store, mock_chroma):
        mock_chroma.add_documents.return_value = None
        ids = vector_store.ingest_data(
            url="https://example.com",
            title="Test Article",
            content="This is test content about artificial intelligence in healthcare.",
        )
        assert isinstance(ids, list)
        assert len(ids) > 0
        assert all(isinstance(i, str) for i in ids), "All IDs must be strings"

    def test_ids_are_valid_uuids(self, vector_store, mock_chroma):
        import uuid
        mock_chroma.add_documents.return_value = None
        ids = vector_store.ingest_data(
            url="https://example.com",
            title="Test",
            content="Content " * 100,
        )
        for id_ in ids:
            uuid.UUID(id_)  # raises if not a valid UUID

    def test_calls_add_documents(self, vector_store, mock_chroma):
        mock_chroma.add_documents.return_value = None
        vector_store.ingest_data(
            url="https://example.com",
            title="Test",
            content="Content about AI.",
        )
        mock_chroma.add_documents.assert_called_once()

    def test_documents_have_correct_metadata(self, vector_store, mock_chroma):
        captured = {}

        def capture(documents, ids):
            captured["docs"] = documents
            captured["ids"] = ids

        mock_chroma.add_documents.side_effect = capture
        vector_store.ingest_data(
            url="https://example.com/article",
            title="My Article",
            content="Enough content to make a chunk." * 10,
        )
        for doc in captured["docs"]:
            assert doc.metadata["url"] == "https://example.com/article"
            assert doc.metadata["title"] == "My Article"
            assert "chunk_index" in doc.metadata

    def test_chunk_count_matches_id_count(self, vector_store, mock_chroma):
        captured = {}

        def capture(documents, ids):
            captured["docs"] = documents
            captured["ids"] = ids

        mock_chroma.add_documents.side_effect = capture
        vector_store.ingest_data(
            url="https://example.com",
            title="Test",
            content="x" * 3000,
        )
        assert len(captured["docs"]) == len(captured["ids"])

    def test_empty_content_returns_empty_ids(self, vector_store, mock_chroma):
        mock_chroma.add_documents.return_value = None
        ids = vector_store.ingest_data(url="https://x.com", title="T", content="")
        assert ids == []


class TestVectorStoreSemanticSearch:
    def test_returns_documents(self, vector_store, mock_chroma):
        doc = Document(page_content="AI in healthcare context", metadata={})
        mock_chroma.similarity_search.return_value = [doc]
        results = vector_store.semantic_search("AI healthcare", k=3)
        assert len(results) == 1
        assert results[0].page_content == "AI in healthcare context"

    def test_calls_similarity_search_with_correct_args(self, vector_store, mock_chroma):
        mock_chroma.similarity_search.return_value = []
        vector_store.semantic_search("my query", k=5)
        mock_chroma.similarity_search.assert_called_once_with(query="my query", k=5)

    def test_returns_empty_list_on_exception(self, vector_store, mock_chroma):
        mock_chroma.similarity_search.side_effect = Exception("DB error")
        results = vector_store.semantic_search("query")
        assert results == []

    def test_default_k_is_6(self, vector_store, mock_chroma):
        mock_chroma.similarity_search.return_value = []
        vector_store.semantic_search("query")
        mock_chroma.similarity_search.assert_called_once_with(query="query", k=6)


class TestVectorStoreGetStore:
    def test_returns_chroma_store(self, vector_store, mock_chroma):
        assert vector_store.get_store() is mock_chroma


class TestModuleLevelShims:
    def test_ingest_store_delegates_to_default_store(self, mock_chroma, mock_embeddings):
        with patch("tools.vector_store.HuggingFaceEmbeddings", return_value=mock_embeddings), \
             patch("tools.vector_store.Chroma", return_value=mock_chroma):
            import importlib
            import tools.vector_store as vs_mod
            importlib.reload(vs_mod)

            mock_chroma.add_documents.return_value = None
            ids = vs_mod.ingest_store(
                url="https://x.com", title="T", content="Short content."
            )
            assert isinstance(ids, list)

    def test_semantic_search_delegates_to_default_store(self, mock_chroma, mock_embeddings):
        doc = Document(page_content="result", metadata={})
        mock_chroma.similarity_search.return_value = [doc]

        with patch("tools.vector_store.HuggingFaceEmbeddings", return_value=mock_embeddings), \
             patch("tools.vector_store.Chroma", return_value=mock_chroma):
            import importlib
            import tools.vector_store as vs_mod
            importlib.reload(vs_mod)

            results = vs_mod.semantic_search("query", k=3)
            assert isinstance(results, list)
