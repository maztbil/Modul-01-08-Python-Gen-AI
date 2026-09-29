import anthropic
import os
from dotenv import load_dotenv

load_dotenv()

# =========================
# CLIENT
# =========================
client = anthropic.Anthropic(
    api_key=os.environ["ANTHROPIC_API_KEY"],
    base_url="https://api.llmsrelay.com"
)

# =========================
# TOKEN COUNT (FIXED)
# =========================
try:
    response = client.messages.count_tokens(
        model="claude-opus-5",
        system="You are a concise assistant.",
        messages=[
            {
                "role": "user",
                "content": "Explain the transformer architecture."
            }
        ]
    )
    input_tokens = response.input_tokens

except Exception:
    # fallback manual estimation (biar gak error)
    text = "Explain the transformer architecture."
    input_tokens = len(text) // 4  # estimasi kasar

print(f"Estimated input tokens: {input_tokens}")


# =========================
# PRICING
# =========================
PRICING = {
    "claude-opus-5": {"input": 15.00, "output": 75.00},   # per 1M tokens
    "claude-sonnet-4-5": {"input": 3.00, "output": 15.00},
    "gpt-4o": {"input": 2.50, "output": 10.00},
    "gpt-4o-mini": {"input": 0.15, "output": 0.60},
}


def estimate_cost(model: str, input_tokens: int, output_tokens: int) -> float:
    """Return estimated cost in USD."""
    if model not in PRICING:
        raise ValueError(f"Unknown model: {model}")

    p = PRICING[model]
    return (input_tokens * p["input"] + output_tokens * p["output"]) / 1_000_000


# contoh hitung biaya
cost = estimate_cost(
    "claude-opus-5",
    input_tokens=500,
    output_tokens=300
)

print(f"Estimated cost: ${cost:.6f}")


# =========================
# CONTEXT LIMIT
# =========================
CONTEXT_LIMITS = {
    "claude-opus-5": 200_000,
    "claude-sonnet-4-5": 200_000,
    "gpt-4o": 128_000,
    "gpt-4o-mini": 128_000,
    "gemini-1.5-pro": 1_000_000,
}


def fits_in_context(model: str, token_count: int, reserve_for_output: int = 2048) -> bool:
    limit = CONTEXT_LIMITS.get(model, 128_000)
    return token_count + reserve_for_output <= limit


# =========================
# TEST CONTEXT
# =========================
tokens = input_tokens  # FIX DI SINI

if fits_in_context("claude-opus-5", tokens):
    print("✅ Fits within context window")
else:
    print("❌ Too large for context window")