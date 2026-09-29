import numpy as np
from dataclasses import dataclass, field
from typing import Optional
import os

# OPTIONAL (kalau mau coba API, tapi nanti fallback ke local)
USE_LOCAL = True

# =========================
# LOCAL EMBEDDING (AMAN)
# =========================
from sentence_transformers import SentenceTransformer

local_model = SentenceTransformer("all-MiniLM-L6-v2")

def embed_batch(texts: list[str]) -> np.ndarray:
    """Embedding pakai model lokal"""
    return local_model.encode(texts, normalize_embeddings=True)


# =========================
# DATA STRUCTURE
# =========================
@dataclass
class Document:
    id: str
    text: str
    embedding: Optional[np.ndarray] = field(default=None, repr=False)


@dataclass
class SearchResult:
    document: Document
    score: float
    rank: int


# =========================
# VECTOR STORE
# =========================
class VectorStore:
    def __init__(self):
        self._documents: list[Document] = []
        self._matrix: Optional[np.ndarray] = None

    def add_documents(self, documents: list[Document]):
        texts = [d.text for d in documents]
        vectors = embed_batch(texts)

        for doc, vec in zip(documents, vectors):
            doc.embedding = vec
            self._documents.append(doc)

        self._matrix = np.array(
            [d.embedding for d in self._documents],
            dtype=np.float32
        )

        print(f"Index now contains {len(self._documents)} documents.")

    def search(self, query: str, k: int = 5):
        q_vec = embed_batch([query])[0]

        scores = self._matrix @ q_vec
        top_idx = np.argsort(scores)[::-1][:k]

        return [
            SearchResult(
                document=self._documents[int(i)],
                score=float(scores[i]),
                rank=rank + 1,
            )
            for rank, i in enumerate(top_idx)
        ]


# =========================
# EVALUATION
# =========================
@dataclass
class RetrievalEvalCase:
    query: str
    relevant_doc_ids: list[str]


def precision_at_k(retrieved_ids, relevant_ids, k):
    top_k = retrieved_ids[:k]
    hits = sum(1 for d in top_k if d in relevant_ids)
    return hits / k if k > 0 else 0.0


def recall_at_k(retrieved_ids, relevant_ids, k):
    if not relevant_ids:
        return 0.0
    top_k = retrieved_ids[:k]
    hits = sum(1 for d in top_k if d in relevant_ids)
    return hits / len(relevant_ids)


def mean_reciprocal_rank(retrieved_ids, relevant_ids):
    for rank, d in enumerate(retrieved_ids, start=1):
        if d in relevant_ids:
            return 1.0 / rank
    return 0.0


def evaluate_retrieval(store, eval_cases, k=5):
    p_scores, r_scores, mrr_scores = [], [], []

    for case in eval_cases:
        results = store.search(case.query, k=k)
        retrieved_ids = [r.document.id for r in results]

        p_scores.append(precision_at_k(retrieved_ids, case.relevant_doc_ids, k))
        r_scores.append(recall_at_k(retrieved_ids, case.relevant_doc_ids, k))
        mrr_scores.append(mean_reciprocal_rank(retrieved_ids, case.relevant_doc_ids))

    return {
        f"precision@{k}": round(float(np.mean(p_scores)), 4),
        f"recall@{k}": round(float(np.mean(r_scores)), 4),
        "MRR": round(float(np.mean(mrr_scores)), 4),
    }


# =========================
# DATASET
# =========================
CORPUS = [
    Document("d01", "RAG combines retrieval with generation."),
    Document("d02", "Vector databases use HNSW or IVF."),
    Document("d03", "Fine-tuning trains a model on new data."),
    Document("d04", "Prompt engineering improves outputs."),
    Document("d05", "LangChain helps build LLM apps."),
    Document("d06", "Cosine similarity compares vectors."),
    Document("d07", "RLHF aligns models with humans."),
    Document("d08", "Chunking is important in RAG pipelines."),
    Document("d09", "Transformers use self-attention."),
    Document("d10", "Agents can plan and use tools."),
]


eval_cases = [
    RetrievalEvalCase("How does RAG work?", ["d01", "d08"]),
    RetrievalEvalCase("What are vector databases?", ["d02"]),
    RetrievalEvalCase("How do agents use language models?", ["d10"]),
    RetrievalEvalCase("What is fine-tuning?", ["d03"]),
    RetrievalEvalCase("How do transformers work?", ["d09"]),
]


# =========================
# RUN
# =========================
store = VectorStore()
store.add_documents(CORPUS)

print("\n=== SEARCH TEST ===")
results = store.search("How does RAG work?", k=3)
for r in results:
    print(f"[{r.rank}] {r.document.id} | score={r.score:.4f}")

print("\n=== EVALUATION ===")
metrics = evaluate_retrieval(store, eval_cases, k=3)

for k, v in metrics.items():
    print(f"{k}: {v}")