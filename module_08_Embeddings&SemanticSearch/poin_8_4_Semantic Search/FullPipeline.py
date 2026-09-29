import numpy as np
from dataclasses import dataclass, field
from typing import Optional, List
from openai import OpenAI
import os
from dotenv import load_dotenv

load_dotenv()

# =========================
# CLIENT (LLMsRelay)
# =========================
client = OpenAI(
    api_key=os.environ.get("OPENAI_API_KEY"),
    base_url="https://api.llmsrelay.com/v1"
)

# =========================
# DATA STRUCTURES
# =========================
@dataclass
class Document:
    id: str
    text: str
    metadata: dict = field(default_factory=dict)
    embedding: Optional[np.ndarray] = field(default=None, repr=False)


@dataclass
class SearchResult:
    document: Document
    score: float
    rank: int


# =========================
# EMBEDDING (fallback local kalau API gagal)
# =========================
def embed_batch(texts: List[str]) -> np.ndarray:
    try:
        response = client.embeddings.create(
            input=texts,
            model="text-embedding-3-small"
        )
        vectors = sorted(response.data, key=lambda e: e.index)
        return np.array([v.embedding for v in vectors], dtype=np.float32)

    except Exception as e:
        print("⚠️ API gagal, pakai local embeddings:", e)

        from sentence_transformers import SentenceTransformer
        model = SentenceTransformer("all-MiniLM-L6-v2")
        return model.encode(texts).astype(np.float32)


# =========================
# VECTOR STORE
# =========================
class VectorStore:
    def __init__(self):
        self._documents: List[Document] = []
        self._matrix: Optional[np.ndarray] = None

    def add_documents(self, documents: List[Document]) -> None:
        texts = [d.text for d in documents]
        vectors = embed_batch(texts)

        # normalize
        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        norms = np.where(norms == 0, 1, norms)
        normed = (vectors / norms).astype(np.float32)

        for doc, vec in zip(documents, normed):
            doc.embedding = vec
            self._documents.append(doc)

        self._matrix = np.array(
            [d.embedding for d in self._documents],
            dtype=np.float32
        )

        print(f"Index now contains {len(self._documents)} documents.")

    def search(self, query: str, k: int = 5) -> List[SearchResult]:
        if self._matrix is None or len(self._documents) == 0:
            raise RuntimeError("No documents indexed yet.")

        q_vec = embed_batch([query])[0]

        q_norm = np.linalg.norm(q_vec)
        if q_norm == 0:
            return []

        q_vec = (q_vec / q_norm).astype(np.float32)

        scores = self._matrix @ q_vec

        k = min(k, len(self._documents))
        top_idx = np.argsort(scores)[::-1][:k]

        results = []
        for rank, i in enumerate(top_idx):
            results.append(
                SearchResult(
                    document=self._documents[int(i)],
                    score=float(scores[i]),
                    rank=rank + 1,
                )
            )

        return results

    @property
    def size(self) -> int:
        return len(self._documents)


# =========================
# DEMO DATA
# =========================
CORPUS = [
    Document("d01", "Retrieval-Augmented Generation (RAG) combines retrieval with LLMs."),
    Document("d02", "Vector databases store embeddings and use HNSW or IVF."),
    Document("d03", "Fine-tuning adapts a model to specific tasks."),
    Document("d04", "Prompt engineering optimizes prompts."),
    Document("d05", "LangChain is a framework for LLM apps."),
    Document("d06", "Cosine similarity compares embeddings."),
    Document("d07", "RLHF aligns models with human feedback."),
    Document("d08", "Chunking is important in RAG pipelines."),
    Document("d09", "Transformers use self-attention."),
    Document("d10", "Agents use LLMs to plan and act."),
]

store = VectorStore()
store.add_documents(CORPUS)

# =========================
# QUERY TEST
# =========================
QUERIES = [
    "How does RAG work?",
    "What algorithms do vector databases use?",
    "How do I split documents for embedding?",
]

for query in QUERIES:
    print(f"\nQuery: {query}")
    results = store.search(query, k=3)

    for r in results:
        print(f"[{r.rank}] score={r.score:.4f} | {r.document.text[:80]}...")