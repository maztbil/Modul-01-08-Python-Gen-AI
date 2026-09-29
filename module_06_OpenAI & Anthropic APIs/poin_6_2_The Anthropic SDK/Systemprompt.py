import anthropic
import os
from dotenv import load_dotenv

load_dotenv()

client = anthropic.Anthropic(
    api_key=os.environ["ANTHROPIC_API_KEY"],
    base_url="https://api.llmsrelay.com"  # WAJIB untuk relay
)

message = client.messages.create(
    model="claude-opus-5",  # bisa juga "claude-opus-5"
    max_tokens=512,
    system="You are a concise technical writer. Answer in plain English, no jargon.",
    messages=[
        {
            "role": "user",
            "content": "Explain what a vector database does."
        }
    ]
)

print(message.content[0].text)