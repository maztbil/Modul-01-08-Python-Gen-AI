import anthropic
import os
from dotenv import load_dotenv

load_dotenv()

client = anthropic.Anthropic(
    api_key=os.environ["ANTHROPIC_API_KEY"],
    base_url="https://api.llmsrelay.com"
)

def chat(system: str) -> None:
    """Simple interactive multi-turn chat loop."""
    
    history = []

    while True:
        user_input = input("You: ").strip()

        if user_input.lower() in ("exit", "quit"):
            break

        # simpan input user
        history.append({
            "role": "user",
            "content": user_input
        })

        # request ke model
        response = client.messages.create(
            model="claude-opus-5",
            max_tokens=1024,
            system=system,
            messages=history,
        )

        assistant_text = response.content[0].text

        # simpan jawaban AI
        history.append({
            "role": "assistant",
            "content": assistant_text
        })

        print(f"\nClaude: {assistant_text}\n")


# JALANKAN CHAT
chat(system="You are a helpful Python tutor.")