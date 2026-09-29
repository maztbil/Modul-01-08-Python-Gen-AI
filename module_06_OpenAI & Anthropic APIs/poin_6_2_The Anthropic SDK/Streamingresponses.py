import anthropic
import os
from dotenv import load_dotenv

load_dotenv()

client = anthropic.Anthropic(
    api_key=os.environ["ANTHROPIC_API_KEY"],
    base_url="https://api.llmsrelay.com"
)

with client.messages.stream(
    model="claude-opus-5",  # bisa ganti ke opus-5 kalau error
    max_tokens=512,
    messages=[
        {
            "role": "user",
            "content": "List 5 use cases for vector databases."
        }
    ],
) as stream:

    # streaming text (real-time)
    for text in stream.text_stream:
        print(text, end="", flush=True)

    print()  # newline setelah selesai

    # ambil final message + usage
    final = stream.get_final_message()

    print(f"\nTotal tokens: {final.usage.input_tokens + final.usage.output_tokens}")