from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import List
import anthropic
import os
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()


# =========================
# DATA STRUCTURES
# =========================
@dataclass
class ChatMessage:
    role: str  # "user" or "assistant"
    content: str


@dataclass
class ChatResponse:
    text: str
    input_tokens: int
    output_tokens: int
    model: str


# =========================
# BASE CLASS
# =========================
class BaseLLMClient(ABC):
    @abstractmethod
    def chat(
        self,
        messages: List[ChatMessage],
        system: str = "",
        max_tokens: int = 1024,
        temperature: float = 0.7,
    ) -> ChatResponse:
        pass


# =========================
# ANTHROPIC CLIENT
# =========================
class AnthropicClient(BaseLLMClient):
    def __init__(self, model: str = "claude-opus-5"):
        self.model = model
        self._client = anthropic.Anthropic(
            api_key=os.environ.get("ANTHROPIC_API_KEY"),  # pakai ini kalau relay
            base_url="https://api.llmsrelay.com"
        )
        # kalau pakai API asli:
        # api_key=os.environ.get("ANTHROPIC_API_KEY")

    def chat(self, messages, system="", max_tokens=1024, temperature=0.7) -> ChatResponse:
        resp = self._client.messages.create(
            model=self.model,
            max_tokens=max_tokens,
            system=system,
            messages=[{"role": m.role, "content": m.content} for m in messages],
        )

        return ChatResponse(
            text=resp.content[0].text,
            input_tokens=resp.usage.input_tokens,
            output_tokens=resp.usage.output_tokens,
            model=self.model,
        )


# =========================
# OPENAI CLIENT
# =========================
class OpenAIClient(BaseLLMClient):
    def __init__(self, model: str = "claude-opus-5"):
        self.model = model
        self._client = OpenAI(
            api_key=os.environ.get("OPENAI_API_KEY"),
            base_url="https://api.llmsrelay.com/v1"  # kalau relay
        )
        # kalau OpenAI asli: hapus base_url

    def chat(self, messages, system="", max_tokens=1024, temperature=0.7) -> ChatResponse:
        api_messages = []

        if system:
            api_messages.append({"role": "system", "content": system})

        api_messages += [
            {"role": m.role, "content": m.content}
            for m in messages
        ]

        resp = self._client.chat.completions.create(
            model=self.model,
            max_tokens=max_tokens,
            temperature=temperature,
            messages=api_messages,
        )

        return ChatResponse(
            text=resp.choices[0].message.content,
            input_tokens=resp.usage.prompt_tokens,
            output_tokens=resp.usage.completion_tokens,
            model=self.model,
        )


# =========================
# USAGE
# =========================
if __name__ == "__main__":
    # pilih backend
    # client: BaseLLMClient = AnthropicClient()
    client: BaseLLMClient = OpenAIClient()

    msgs = [
        ChatMessage(role="user", content="What is a vector database?")
    ]

    result = client.chat(msgs, system="Be concise.")

    print(result.text)
    print(f"Tokens: {result.input_tokens} in, {result.output_tokens} out")