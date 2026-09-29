from openai import OpenAI
import os
from dotenv import load_dotenv

load_dotenv()

client = OpenAI(
    api_key=os.environ["OPENAI_API_KEY"],
    base_url="https://api.llmsrelay.com/v1"  # karena lu pakai relay
)

stream = client.chat.completions.create(
    model="claude-opus-5",
    max_tokens=512,
    stream=True,
    messages=[
        {
            "role": "user",
            "content": "Explain embeddings in 3 bullet points."
        }
    ]
)

# streaming output
for chunk in stream:
    delta = chunk.choices[0].delta.content
    if delta:
        print(delta, end="", flush=True)

print()  # newline setelah selesai