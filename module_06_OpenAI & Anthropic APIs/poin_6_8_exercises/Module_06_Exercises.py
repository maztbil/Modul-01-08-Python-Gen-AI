# =========================================
# MODULE 06 - ALL IN ONE
# =========================================

import os
import time
import random
import asyncio
from dataclasses import dataclass
from typing import List
from dotenv import load_dotenv

import anthropic
from openai import OpenAI, RateLimitError as OpenAIRateLimitError
import pandas as pd

load_dotenv()


# =========================================
# DATA STRUCTURES
# =========================================
@dataclass
class ChatMessage:
    role: str
    content: str


@dataclass
class ChatResponse:
    text: str
    input_tokens: int
    output_tokens: int
    model: str


# =========================================
# BASE CLIENT
# =========================================
class BaseLLMClient:
    def chat(self, messages: List[ChatMessage]) -> ChatResponse:
        raise NotImplementedError


# =========================================
# ANTHROPIC CLIENT
# =========================================
class AnthropicClient(BaseLLMClient):
    def __init__(self, model="claude-opus-5"):
        self.model = model
        self.client = anthropic.Anthropic(
            api_key=os.environ.get("ANTHROPIC_API_KEY"),
            base_url="https://api.llmsrelay.com"
        )

    def chat(self, messages: List[ChatMessage]) -> ChatResponse:
        resp = self.client.messages.create(
            model=self.model,
            max_tokens=512,
            messages=[{"role": m.role, "content": m.content} for m in messages],
        )

        return ChatResponse(
            text=resp.content[0].text,
            input_tokens=resp.usage.input_tokens,
            output_tokens=resp.usage.output_tokens,
            model=self.model
        )


# =========================================
# OPENAI CLIENT
# =========================================
class OpenAIClient(BaseLLMClient):
    def __init__(self, model="claude-opus-5"):
        self.model = model
        self.client = OpenAI(
            api_key=os.environ.get("OPENAI_API_KEY"),
            base_url="https://api.llmsrelay.com/v1"
        )

    def chat(self, messages: List[ChatMessage]) -> ChatResponse:
        resp = self.client.chat.completions.create(
            model=self.model,
            max_tokens=512,
            messages=[{"role": m.role, "content": m.content} for m in messages],
        )

        return ChatResponse(
            text=resp.choices[0].message.content,
            input_tokens=resp.usage.prompt_tokens,
            output_tokens=resp.usage.completion_tokens,
            model=self.model
        )


# =========================================
# 1. RETRY ON RATE LIMIT
# =========================================
def retry_on_rate_limit(client, messages, max_retries=5):
    for attempt in range(max_retries):
        try:
            return client.chat(messages)

        except (anthropic.RateLimitError, OpenAIRateLimitError):
            sleep_time = (2 ** attempt) + random.random()
            print(f"Retrying in {sleep_time:.2f}s...")
            time.sleep(sleep_time)

    raise Exception("Max retries exceeded")


# =========================================
# 2. TOKEN BUDGET MANAGER
# =========================================
class BudgetExceeded(Exception):
    pass


class TokenBudgetManager:
    def __init__(self, max_tokens: int):
        self.max_tokens = max_tokens
        self.used_tokens = 0

    def add_usage(self, input_tokens: int, output_tokens: int):
        total = input_tokens + output_tokens
        self.used_tokens += total

        if self.used_tokens > self.max_tokens:
            raise BudgetExceeded(
                f"Budget exceeded: {self.used_tokens}/{self.max_tokens}"
            )

    def remaining(self):
        return self.max_tokens - self.used_tokens


# =========================================
# 3. COMPARE MODELS (ASYNC)
# =========================================
async def call_model(client, model_name, prompt):
    start = time.time()

    msgs = [ChatMessage(role="user", content=prompt)]

    try:
        response = client.chat(msgs)

        latency = (time.time() - start) * 1000

        return {
            "model": model_name,
            "response_text": response.text,
            "input_tokens": response.input_tokens,
            "output_tokens": response.output_tokens,
            "latency_ms": latency,
        }

    except Exception as e:
        return {
            "model": model_name,
            "response_text": str(e),
            "input_tokens": 0,
            "output_tokens": 0,
            "latency_ms": 0,
        }


async def compare_models(prompt: str, models: list, client_type="openai"):
    tasks = []

    for model in models:
        if client_type == "openai":
            client = OpenAIClient(model)
        else:
            client = AnthropicClient(model)

        tasks.append(call_model(client, model, prompt))

    results = await asyncio.gather(*tasks)
    return pd.DataFrame(results)


# =========================================
# 4. STREAM TO FILE (ANTHROPIC)
# =========================================
def stream_to_file(prompt: str, output_path: str):
    client = anthropic.Anthropic(
        api_key=os.environ.get("ANTHROPIC_API_KEY"),
        base_url="https://api.llmsrelay.com"
    )

    with open(output_path, "w", encoding="utf-8") as f:
        with client.messages.stream(
            model="claude-opus-5",
            max_tokens=512,
            messages=[{"role": "user", "content": prompt}],
        ) as stream:

            for text in stream.text_stream:
                print(text, end="", flush=True)
                f.write(text)
                f.flush()

        final = stream.get_final_message()

    print(f"\nSaved to {output_path}")
    print(f"Tokens: {final.usage.input_tokens + final.usage.output_tokens}")


# =========================================
# MAIN TEST
# =========================================
if __name__ == "__main__":
    # pilih client
    client = OpenAIClient("claude-opus-5")

    messages = [ChatMessage("user", "Explain RAG in simple terms")]

    # 1. retry
    result = retry_on_rate_limit(client, messages)
    print("\n=== RESPONSE ===")
    print(result.text)

    # 2. budget
    budget = TokenBudgetManager(max_tokens=2000)
    budget.add_usage(result.input_tokens, result.output_tokens)
    print(f"\nRemaining tokens: {budget.remaining()}")

    # 3. compare models
    print("\n=== COMPARING MODELS ===")
    df = asyncio.run(compare_models(
        "Explain vector databases",
        ["gpt-4o", "gpt-4o-mini"]
    ))
    print(df)

    # 4. streaming
    print("\n=== STREAMING ===")
    stream_to_file(
        "Write a short explanation of embeddings.",
        "output.txt"
    )