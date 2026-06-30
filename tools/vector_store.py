import uuid
from langchain_chroma import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.documents import Document


class VectorStore:
    def __init__(
            self,
            collection_name: str,
            embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2",
            persist_path: str = "./chroma_db",
    ) -> None:
        self._model_name = embedding_model
        self._persist_path = persist_path
        self._collection_name = collection_name
        self._store = None
        self._splitter = None
        self._load_configs()

    def _load_configs(self) -> None:
        embeddings = HuggingFaceEmbeddings(model_name=self._model_name)
        self._store = Chroma(
            collection_name=self._collection_name,
            embedding_function=embeddings,
            persist_directory=self._persist_path
        )
        self._splitter = RecursiveCharacterTextSplitter(
            chunk_size=800,
            chunk_overlap=100,
            separators=["\n\n", "\n", ". ", " "]
        )

    def ingest_data(self, url: str, title: str, content: str) -> list[str]:
        chunks = self._splitter.split_text(content)
        docs = [
            Document(
                page_content=chunk,
                metadata={"url": url, "title": title, "chunk_index": i}
            )
            for i, chunk in enumerate(chunks)
        ]
        ids = [str(uuid.uuid4()) for _ in chunks]
        self._store.add_documents(documents=docs, ids=ids)
        return ids

    def semantic_search(self, query: str, k: int = 6) -> list[Document]:
        try:
            return self._store.similarity_search(query=query, k=k)
        except Exception:
            return []

    def get_store(self) -> Chroma:
        return self._store


_default_store = VectorStore(collection_name="research_knowledge")


def ingest_store(url: str, title: str, content: str) -> list[str]:
    return _default_store.ingest_data(url=url, title=title, content=content)


def semantic_search(query: str, k: int = 6) -> list[Document]:
    return _default_store.semantic_search(query=query, k=k)
