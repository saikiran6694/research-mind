import uuid
from langchain_chroma import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_classic.schema import Document

embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

_store = Chroma(
    collection_name="research_knowledge",
    embedding_function=embeddings,
    persist_directory="./chroma_db"
)

splitter = RecursiveCharacterTextSplitter(
    chunk_size=800,
    chunk_overlap=100,
    separators=["\n\n", "\n", ". ", " "]
)


def ingest_store(url: str, title: str, content: str) -> list[str]:
    """Chunk and embed a source document. Returns chunk IDs."""
    chunks = splitter.split_text(content)
    docs = [
        Document(
            page_content=chunk,
            metadata={"url": url, "title": title, "chunk_index": i}
        )
        for i, chunk in enumerate(chunks)
    ]
    ids = [str(uuid.uuid4()) for _ in docs]
    _store.add_documents(docs, ids=ids)
    return ids


def semantic_search(query: str, k: int = 6) -> list[Document]:
    """Retrieve most relevant chunks for a query."""
    try:
        return _store.similarity_search(query=query, k=k)
    except Exception:
        return []
    
def get_store() -> Chroma:
    return _store