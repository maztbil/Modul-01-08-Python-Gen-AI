import anthropic
import os
import base64
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

client = anthropic.Anthropic(
    api_key=os.environ["ANTHROPIC_API_KEY"],  # pakai ini kalau lu pakai relay
    base_url="https://api.llmsrelay.com"
)

def describe_image_file(path: str) -> str:
    data = Path(path).read_bytes()
    b64 = base64.b64encode(data).decode()

    ext = Path(path).suffix.lstrip(".").lower()
    if ext == "jpg":
        ext = "jpeg"

    media_type = f"image/{ext}"

    response = client.messages.create(
        model="claude-opus-5",
        max_tokens=512,
        messages=[
            {
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "source": {
                            "type": "base64",
                            "media_type": media_type,
                            "data": b64
                        }
                    },
                    {
                        "type": "text",
                        "text": "Describe this image in detail."
                    }
                ]
            }
        ]
    )

    return response.content[0].text


# =========================
# AUTO RUN
# =========================
if __name__ == "__main__":
    result = describe_image_file("7b623356ae296b2c8c13f563f50f2373.jpg")
    print(result)