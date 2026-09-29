import anthropic
import os
import re
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
SYSTEM = """Solve problems using this exact format:

<thinking>
Step-by-step reasoning here.
</thinking>

<answer>
The final answer only, no reasoning.
</answer>
"""

# =========================
# REQUEST
# =========================
resp = client.messages.create(
    model="claude-opus-5",
    max_tokens=512,
    system=SYSTEM,
    messages=[
        {
            "role": "user",
            "content": """A RAG pipeline retrieves 5 documents, each 400 tokens.
The query is 50 tokens. The model has a 4096 token limit for context.
How many tokens remain for the response?"""
        }
    ],
)

text = resp.content[0].text

# =========================
# PARSING
# =========================
thinking_match = re.search(r"<thinking>(.*?)</thinking>", text, re.DOTALL)
answer_match = re.search(r"<answer>(.*?)</answer>", text, re.DOTALL)

thinking = thinking_match.group(1).strip() if thinking_match else "not found"
answer = answer_match.group(1).strip() if answer_match else "not found"

# =========================
# OUTPUT
# =========================
print("=== RAW RESPONSE ===")
print(text)

print("\n=== PARSED ===")
print("Reasoning:", thinking)
print("Answer:", answer)