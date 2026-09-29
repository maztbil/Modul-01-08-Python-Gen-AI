from openai import OpenAI
import os
import numpy as np
from dotenv import load_dotenv

# 🔥 tambahan buat lokal
from sentence_transformers import SentenceTransformer

load_dotenv()

# =========================
# CLIENT (LLMsRelay / OpenAI)
# =========================
client = OpenAI(
    api_key=os.environ.get("OPENAI_API_KEY"),
    base_url="https://api.llmsrelay.com/v1"  # relay
)

# =========================
# LOCAL MODEL (fallback)
# =========================
local_model = SentenceTransformer("all-MiniLM-L6-v2")

# =========================
# EMBEDDING FUNCTION
# =========================
def embed(texts: list[str], model: str = "text-embedding-3-small") -> np.ndarray:
    """Embed texts with API first, fallback to local if failed."""

    try:
        print("🌐 Trying API embeddings...")

        response = client.embeddings.create(
            input=texts,
            model=model
        )

        vectors = sorted(response.data, key=lambda e: e.index)

        embeddings = np.array([v.embedding for v in vectors], dtype=np.float32)

        print("✅ API embeddings success")
        return embeddings

    except Exception as e:
        print("⚠️ API failed, fallback ke lokal:", str(e))
        print("💻 Using local embeddings...")

        embeddings = local_model.encode(
            texts,
            convert_to_numpy=True,
            normalize_embeddings=True
        )

        print("✅ Local embeddings success")
        return embeddings


# =========================
# TEST DATA
# =========================
texts = [
    "Retrieval-Augmented Generation combines search with LLMs.",
    "RAG retrieves documents then generates an answer from them.",
    "The Eiffel Tower is in Paris.",
    "Python is a popular programming language.",
    "Fine-tuning trains a model on new data.",
]

# =========================
# RUN
# =========================
embeddings = embed(texts)

print(f"\nShape: {embeddings.shape}")
print(f"Norm of first vector: {np.linalg.norm(embeddings[0]):.4f}")