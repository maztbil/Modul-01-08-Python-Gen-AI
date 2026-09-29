import anthropic
import os
import json
import re
from dataclasses import dataclass, asdict
from typing import List, Dict
from dotenv import load_dotenv

load_dotenv()

client = anthropic.Anthropic(
    api_key=os.environ.get("ANTHROPIC_API_KEY"),
    base_url="https://api.llmsrelay.com"
)

# =====================================================
# 1. SYSTEM PROMPTS (BASIC → EXPERT)
# =====================================================

BASIC_PROMPT = "You are a code review assistant."

INTERMEDIATE_PROMPT = """You are a code reviewer.
- Find bugs
- Suggest improvements
- Keep answers short"""

EXPERT_PROMPT = """You are a senior software engineer reviewing production code.

Your job:
- Identify bugs, security issues, and performance problems
- Suggest concrete fixes with code examples
- Explain WHY each issue matters

Rules:
- Be direct
- No fluff
- Always give improved code

Format:
Issue → Impact → Fix"""

CODE_SNIPPETS = [
    "def add(a,b):return a+b",
    "password='123456'",
    "for i in range(len(arr)): print(arr[i])",
    "requests.get('http://api.com/data')",
    "def func(x): return x*x*x*x*x*x*x"
]


def evaluate_prompts():
    prompts = {
        "basic": BASIC_PROMPT,
        "intermediate": INTERMEDIATE_PROMPT,
        "expert": EXPERT_PROMPT
    }

    results = {}

    for level, system in prompts.items():
        outputs = []

        for code in CODE_SNIPPETS:
            resp = client.messages.create(
                model="claude-opus-5",
                max_tokens=300,
                system=system,
                messages=[{"role": "user", "content": code}]
            )
            text = resp.content[0].text.strip()
            outputs.append(text)

        results[level] = outputs

    return results


# =====================================================
# 2. PROMPT LIBRARY
# =====================================================

@dataclass
class PromptTemplate:
    name: str
    system: str
    user: str
    version: str


class PromptLibrary:
    def __init__(self):
        self.templates: Dict[str, PromptTemplate] = {}
        self.last_used_version: Dict[str, str] = {}

    def add(self, template: PromptTemplate):
        self.templates[template.name] = template

    def save(self, path: str):
        data = {
            name: asdict(tpl) for name, tpl in self.templates.items()
        }
        with open(path, "w") as f:
            json.dump(data, f, indent=2)

    def load(self, path: str):
        with open(path) as f:
            data = json.load(f)
        for name, tpl in data.items():
            self.templates[name] = PromptTemplate(**tpl)

    def mark_used(self, name: str):
        self.last_used_version[name] = self.templates[name].version


# =====================================================
# 3. SAFE JSON PARSE
# =====================================================

def safe_json_parse(text: str) -> dict:
    # 1. Try direct
    try:
        return json.loads(text)
    except:
        pass

    # 2. Remove markdown
    cleaned = re.sub(r"```json|```", "", text).strip()

    try:
        return json.loads(cleaned)
    except:
        pass

    # 3. Ask model to fix
    repair_prompt = f"""Fix this JSON and return ONLY valid JSON:
{text}"""

    resp = client.messages.create(
        model="claude-opus-5",
        max_tokens=200,
        messages=[{"role": "user", "content": repair_prompt}]
    )

    fixed = resp.content[0].text.strip()

    try:
        return json.loads(fixed)
    except:
        return {"error": "failed_to_parse", "raw": text}


# =====================================================
# 4. CoT PROMPT (RANKING MODELS)
# =====================================================

COT_SYSTEM = """You are an AI evaluator.

Steps:
1. Compare scores across tasks
2. Rank models overall
3. Give a short recommendation

Think step by step.

Finally output:
RANKING: ...
RECOMMENDATION: (2 sentences max)"""


def evaluate_models(scores_text: str):
    resp = client.messages.create(
        model="claude-opus-5",
        max_tokens=300,
        system=COT_SYSTEM,
        messages=[{"role": "user", "content": scores_text}]
    )
    return resp.content[0].text


# =====================================================
# TEST RUN
# =====================================================

if __name__ == "__main__":

    print("\n=== 1. PROMPT COMPARISON ===")
    results = evaluate_prompts()

    for level, outputs in results.items():
        print(f"\n--- {level.upper()} ---")
        for i, out in enumerate(outputs):
            print(f"\nCode {i+1}:\n{out[:150]}...")

    print("\n=== 2. PROMPT LIBRARY ===")
    lib = PromptLibrary()
    lib.add(PromptTemplate("qa", "system text", "user text", "1.0"))
    lib.save("prompts.json")
    lib.load("prompts.json")
    lib.mark_used("qa")
    print("Loaded templates:", lib.templates.keys())

    print("\n=== 3. JSON REPAIR ===")
    bad_json = """```json
    {"name": "test",}
    ```"""
    print(safe_json_parse(bad_json))

    print("\n=== 4. MODEL RANKING ===")

    test_inputs = [
        "ModelA: 80,85,90 | ModelB: 88,82,87 | ModelC: 70,75,78",
        "GPT: 90,91,92 | Claude: 88,89,87 | Gemini: 85,84,83",
        "A: 60,70,80 | B: 85,80,78 | C: 90,88,91"
    ]

    for t in test_inputs:
        print("\nInput:", t)
        print(evaluate_models(t))