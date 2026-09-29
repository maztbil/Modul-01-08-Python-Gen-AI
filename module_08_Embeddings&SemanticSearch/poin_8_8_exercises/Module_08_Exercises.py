import numpy as np
import hashlib
import sqlite3
from dataclasses import dataclass, field
from typing import List, Tuple
from sentence_transformers import SentenceTransformer

# =========================
# LOCAL EMBEDDING MODEL
# =========================
model = SentenceTransformer("all-MiniLM-L6-v2")

# =========================
# CACHE (SQLite)
# =========================
class EmbeddingCache:
    def __init__(self, db_path="embeddings_cache.db"):
        self.conn = sqlite3.connect(db_path)
        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS embeddings (
                key TEXT PRIMARY KEY,
                vector BLOB
            )
        """)

    def _hash(self, text: str) -> str:
        return hashlib.sha256(text.encode()).hexdigest()

    def get(self, text: str):
        key = self._hash(text)
        cur = self.conn.execute("SELECT vector FROM embeddings WHERE key=?", (key,))
        row = cur.fetchone()
        if row:
            return np.frombuffer(row[0], dtype=np.float32)
        return None

    def set(self, text: str, vec: np.ndarray):
        key = self._hash(text)
        self.conn.execute(
            "INSERT OR REPLACE INTO embeddings VALUES (?, ?)",
            (key, vec.astype(np.float32).tobytes())
        )
        self.conn.commit()


cache = EmbeddingCache()


def embed_with_cache(texts: List[str]) -> np.ndarray:
    vectors = []
    missing = []
    missing_idx = []

    for i, t in enumerate(texts):
        cached = cache.get(t)
        if cached is not None:
            vectors.append(cached)
        else:
            vectors.append(None)
            missing.append(t)
            missing_idx.append(i)

    if missing:
        new_vecs = model.encode(missing, normalize_embeddings=True)
        for i, vec in zip(missing_idx, new_vecs):
            cache.set(texts[i], vec)
            vectors[i] = vec

    return np.array(vectors, dtype=np.float32)


# =========================
# DATA STRUCTURE
# =========================
@dataclass
class Document:
    id: str
    text: str
    embedding: np.ndarray = field(default=None)


# =========================
# VECTOR STORE (EXTENDED)
# =========================
class VectorStore:
    def __init__(self):
        self.docs: List[Document] = []

    def add(self, docs: List[Document]):
        texts = [d.text for d in docs]
        vecs = embed_with_cache(texts)

        for d, v in zip(docs, vecs):
            d.embedding = v
            self.docs.append(d)

    def delete(self, doc_id: str):
        self.docs = [d for d in self.docs if d.id != doc_id]

    def update(self, doc_id: str, new_text: str):
        for d in self.docs:
            if d.id == doc_id:
                d.text = new_text
                d.embedding = embed_with_cache([new_text])[0]

    def search(self, query: str, k=5):
        q = embed_with_cache([query])[0]
        scores = [(d, float(np.dot(d.embedding, q))) for d in self.docs]
        scores.sort(key=lambda x: x[1], reverse=True)
        return scores[:k]


# =========================
# DUPLICATE DETECTOR
# =========================
class DuplicateDetector:
    def __init__(self, threshold=0.95):
        self.threshold = threshold

    def find_duplicates(self, docs: List[Document]) -> List[Tuple[str, str, float]]:
        texts = [d.text for d in docs]
        vecs = embed_with_cache(texts)

        results = []
        for i in range(len(vecs)):
            for j in range(i + 1, len(vecs)):
                sim = float(np.dot(vecs[i], vecs[j]))
                if sim >= self.threshold:
                    results.append((docs[i].id, docs[j].id, sim))
        return results


# =========================
# HYBRID SEARCH
# =========================
class HybridSearch:
    def __init__(self, docs: List[Document]):
        self.docs = docs
        self.texts = [d.text.lower() for d in docs]
        self.embeddings = embed_with_cache(self.texts)

    def keyword_score(self, query: str, text: str):
        q_words = query.lower().split()
        return sum(text.count(w) for w in q_words)

    def search(self, query: str, alpha=0.7, k=5):
        q_vec = embed_with_cache([query])[0]

        scores = []
        for i, d in enumerate(self.docs):
            semantic = float(np.dot(self.embeddings[i], q_vec))
            keyword = self.keyword_score(query, d.text)

            final = alpha * semantic + (1 - alpha) * keyword
            scores.append((d, final))

        scores.sort(key=lambda x: x[1], reverse=True)
        return scores[:k]


# =========================
# DEMO
# =========================
docs = [
    Document("d1", "RAG combines retrieval and generation"),
    Document("d2", "RAG combines retrieval and generation"),  # duplicate
    Document("d3", "Vector databases use HNSW and IVF"),
    Document("d4", "Transformers use attention mechanism"),
    Document("d5", "Embedding maps text to vectors"),
]

store = VectorStore()
store.add(docs)

print("\n=== SEARCH ===")
for d, s in store.search("What is RAG?", k=3):
    print(d.id, round(s, 4))

print("\n=== DUPLICATES ===")
detector = DuplicateDetector(threshold=0.95)
dups = detector.find_duplicates(docs)
for a, b, sim in dups:
    print(a, b, round(sim, 4))

print("\n=== HYBRID SEARCH ===")
hybrid = HybridSearch(docs)
for d, s in hybrid.search("RAG retrieval", alpha=0.7):
    print(d.id, round(s, 4))

print("\n=== UPDATE TEST ===")
store.update("d5", "Embeddings represent meaning of text")
print(store.search("embedding meaning")[0][0].text)

print("\n=== DELETE TEST ===")
store.delete("d2")
print("Total docs:", len(store.docs))