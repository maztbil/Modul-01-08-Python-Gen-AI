import anthropic
import os
import json
from dataclasses import dataclass
from typing import List, Dict, Any
from dotenv import load_dotenv

load_dotenv()

# =========================
# CLIENT
# =========================
client = anthropic.Anthropic(
    api_key=os.environ.get("ANTHROPIC_API_KEY"),  # pakai ini kalau relay
    base_url="https://api.llmsrelay.com"
)

# =========================
# DATA STRUCTURE
# =========================
@dataclass
class EvalCase:
    input_text: str
    expected_keywords: List[str]  # at least one must appear in response
    must_be_json: bool = False


# =========================
# EVALUATION FUNCTION
# =========================
def evaluate_prompt(system: str, cases: List[EvalCase]) -> Dict[str, Any]:
    """Run a prompt against test cases and return pass rate + details."""
    results = []

    for case in cases:
        try:
            resp = client.messages.create(
                model="claude-opus-5",
                max_tokens=256,
                system=system,
                messages=[{"role": "user", "content": case.input_text}],
            )

            text = resp.content[0].text.strip()

            # 🔥 keyword check
            keyword_hit = any(
                kw.lower() in text.lower() for kw in case.expected_keywords
            )

            # 🔥 JSON validation (optional)
            json_valid = True
            if case.must_be_json:
                try:
                    cleaned = text.replace("```json", "").replace("```", "").strip()
                    json.loads(cleaned)
                except json.JSONDecodeError:
                    json_valid = False

            passed = keyword_hit and json_valid

            results.append({
                "input": case.input_text[:60],
                "passed": passed,
                "response_preview": text[:80],
            })

        except Exception as e:
            results.append({
                "input": case.input_text[:60],
                "passed": False,
                "response_preview": f"ERROR: {str(e)}",
            })

    pass_rate = sum(r["passed"] for r in results) / len(results)

    return {
        "pass_rate": pass_rate,
        "results": results
    }


# =========================
# TEST CASES
# =========================
CLASSIFY_SYSTEM = """Classify the AI task as one of:
CLASSIFICATION, GENERATION, RETRIEVAL, EMBEDDING.
Return ONLY the category word."""

test_cases = [
    EvalCase("Predict whether an email is spam.", ["CLASSIFICATION"]),
    EvalCase("Write a product description for headphones.", ["GENERATION"]),
    EvalCase("Find the most relevant documents for a query.", ["RETRIEVAL"]),
    EvalCase("Convert this sentence to a vector.", ["EMBEDDING"]),
    EvalCase("Label customer reviews as positive or negative.", ["CLASSIFICATION"]),
]

# =========================
# RUN
# =========================
report = evaluate_prompt(CLASSIFY_SYSTEM, test_cases)

print(f"Pass rate: {report['pass_rate']:.0%}")

for r in report["results"]:
    status = "PASS" if r["passed"] else "FAIL"
    print(f"[{status}] {r['input']!r} → {r['response_preview']!r}")