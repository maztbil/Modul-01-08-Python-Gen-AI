# pip install voyageai
import voyageai
import os
import numpy as np
from dotenv import load_dotenv

load_dotenv()

# =========================
# CLIENT
# =========================
api_key = os.environ.get("VOYAGE_API_KEY")

if not api_key:
    raise ValueError("❌ VOYAGE_API_KEY belum diset di .env")

vo = voyageai.Client(api_key=api_key)

# =========================
# EMBEDDING FUNCTION
# =========================
def embed(texts):
    try:
        print("🌐 Using VoyageAI embeddings...")

        result = vo.embed(
            texts,
            model="voyage-3",          # model terbaru
            input_type="document"      # "document" atau "query"
        )

        embeddings = np.array(result.embeddings, dtype=np.float32)

        print("✅ VoyageAI success")
        print(f"Token usage: {result.total_tokens}")

        return embeddings

    except Exception as e:
        print("⚠️ VoyageAI error:", str(e))
        raise


# =========================
# TEST
# =========================
texts = [
    "What is RAG?",
    "Explain vector databases."
]

embeddings = embed(texts)

print(f"Shape: {embeddings.shape}")   # biasanya (2, 1024)
print(f"Norm: {np.linalg.norm(embeddings[0]):.4f}")