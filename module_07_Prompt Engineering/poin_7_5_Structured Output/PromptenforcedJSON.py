import anthropic
import os
import json
from dotenv import load_dotenv

load_dotenv()

# =========================
# CLIENT
# =========================
client = anthropic.Anthropic(
    api_key=os.environ.get("ANTHROPIC_API_KEY"),  # kalau pakai relay
    base_url="https://api.llmsrelay.com"
)

# kalau pakai API resmi:
# client = anthropic.Anthropic(
#     api_key=os.environ.get("ANTHROPIC_API_KEY")
# )

# =========================
# SYSTEM PROMPT
# =========================
SYSTEM = """You are a data extractor. Extract information and return ONLY a JSON object.

No markdown, no explanation, no code fences. Raw JSON only.

Schema:
{
  "company": string,
  "founded": integer or null,
  "products": [string],
  "headquarters": string or null,
  "is_public": boolean
}
"""

# =========================
# INPUT TEXTS
# =========================
texts = [
    """Anthropic was founded in 2021 by Dario Amodei and others.
It makes Claude AI models and is headquartered in San Francisco.
It is a private company.""",

    """OpenAI, founded in 2015, created ChatGPT and GPT-4.
Based in San Francisco, it remains private despite a major Microsoft investment."""
]

# =========================
# FUNCTION
# =========================
def extract_company_info(text: str) -> dict:
    resp = client.messages.create(
        model="claude-opus-5",
        max_tokens=256,
        system=SYSTEM,
        messages=[{"role": "user", "content": text}],
    )

    raw = resp.content[0].text.strip()

    # 🔥 CLEANING (biar gak error parsing)
    raw = raw.replace("```json", "").replace("```", "").strip()

    # 🔥 FAILSAFE kalau JSON rusak
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        print("⚠️ JSON parse failed, raw output:")
        print(raw)
        return {"error": "invalid_json", "raw": raw}


# =========================
# RUN
# =========================
for text in texts:
    info = extract_company_info(text)
    print(json.dumps(info, indent=2))
    print()