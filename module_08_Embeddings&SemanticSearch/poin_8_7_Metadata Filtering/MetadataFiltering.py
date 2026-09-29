from dataclasses import dataclass, field
from typing import Any, Callable, Optional
import numpy as np
import os
from dotenv import load_dotenv

load_dotenv()

# =========================
# TRY OPENAI (OPTIONAL)
# =========================
USE_LOCAL = True

if not USE_LOCAL:
    from openai import OpenAI
    openai_client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

# =========================
# LOCAL EMBEDDING (SAFE)
# =========================
from sentence_transformers import SentenceTransformer

local_model = SentenceTransformer("all-MiniLM-L6-v2")


def embed_texts(texts: list[str]) -> np.ndarray:
    """Embedding dengan fallback"""
    if USE_LOCAL:
        return local_model.encode(texts, normalize_embeddings=True)

    try:
        resp = openai_client.embeddings.create(
            input=texts,
            model="text-embedding-3-small"
        )

        vecs = np.array(
            [e.embedding for e in sorted(resp.data, key=lambda x: x.index)],
            dtype=np.float32
        )

        norms = np.linalg.norm(vecs, axis=1, keepdims=True)
        return vecs / np.where(norms == 0, 1, norms)

    except Exception as e:
        print("⚠️ API gagal, pakai local:", e)
        return local_model.encode(texts, normalize_embeddings=True)


# =========================
# DATA STRUCTURE
# =========================
@dataclass
class FilteredDocument:
    id: str
    text: str
    metadata: dict[str, Any] = field(default_factory=dict)
    embedding: Optional[np.ndarray] = field(default=None, repr=False)


# =========================
# VECTOR STORE + FILTER
# =========================
class FilteredVectorStore:
    def __init__(self):
        self._docs: list[FilteredDocument] = []

    def add(self, docs: list[FilteredDocument]) -> None:
        embeddings = embed_texts([d.text for d in docs])

        for doc, emb in zip(docs, embeddings):
            doc.embedding = emb
            self._docs.append(doc)

        print(f"Index size: {len(self._docs)} documents")

    def search(
        self,
        query: str,
        k: int = 5,
        filter_fn: Optional[Callable[[FilteredDocument], bool]] = None,
    ) -> list[tuple[FilteredDocument, float]]:
        """Search dengan optional filter metadata"""

        # Filter dulu
        candidates = (
            self._docs if filter_fn is None
            else [d for d in self._docs if filter_fn(d)]
        )

        if not candidates:
            return []

        # Embed query
        q_vec = embed_texts([query])[0]

        # Hitung similarity
        matrix = np.array([d.embedding for d in candidates], dtype=np.float32)
        scores = matrix @ q_vec

        # Top-k
        k = min(k, len(candidates))
        top_idx = np.argsort(scores)[::-1][:k]

        return [(candidates[i], float(scores[i])) for i in top_idx]


# =========================
# DEMO DATA
# =========================
docs = [
    FilteredDocument("a1", "GPT-4o supports vision and function calling.", {"category": "openai", "year": 2024}),
    FilteredDocument("a2", "Claude Sonnet excels at coding tasks.", {"category": "anthropic", "year": 2024}),
    FilteredDocument("a3", "GPT-4o-mini is a smaller, cheaper model.", {"category": "openai", "year": 2024}),
    FilteredDocument("a4", "Claude Opus is Anthropic's most capable model.", {"category": "anthropic", "year": 2025}),
    FilteredDocument("a5", "GPT-4 Turbo has a 128K context window.", {"category": "openai", "year": 2023}),
]

# =========================
# RUN
# =========================
fstore = FilteredVectorStore()
fstore.add(docs)

# Semua dokumen
print("\n=== All docs ===")
results = fstore.search("which model is good at coding?", k=3)
for doc, score in results:
    print(f"[{score:.4f}] {doc.id}: {doc.text}")

# Filter Anthropic
print("\n=== Anthropic only ===")
results = fstore.search(
    "which model is good at coding?",
    k=3,
    filter_fn=lambda d: d.metadata["category"] == "anthropic"
)

for doc, score in results:
    print(f"[{score:.4f}] {doc.id}: {doc.text}")