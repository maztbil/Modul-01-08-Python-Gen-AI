from openai import OpenAI
import os
import json
from dotenv import load_dotenv

load_dotenv()

# =========================
# CLIENT
# =========================
client = OpenAI(
    api_key=os.environ.get("OPENAI_API_KEY"),
    base_url="https://api.llmsrelay.com/v1"  # kalau pakai relay
)

# =========================
# REQUEST
# =========================
response = client.chat.completions.create(
    model="claude-opus-5",
    response_format={"type": "json_object"},  # ✅ enforce JSON output
    messages=[
        {
            "role": "system",
            "content": """Extract entities. Return JSON with this schema:
{"people": [string], "organizations": [string], "locations": [string]}"""
        },
        {
            "role": "user",
            "content": "Elon Musk founded SpaceX in Hawthorne, California. He also leads Tesla."
        }
    ]
)

# =========================
# PARSE RESULT
# =========================
raw = response.choices[0].message.content

try:
    result = json.loads(raw)
    print(result)
except json.JSONDecodeError:
    print("⚠️ JSON parsing failed")
    print(raw)