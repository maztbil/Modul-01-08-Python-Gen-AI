import os
from dotenv import load_dotenv
import anthropic

# WAJIB INI
load_dotenv()

client = anthropic.Anthropic(
    api_key=os.environ["ANTHROPIC_API_KEY"],
    base_url="https://api.llmsrelay.com",
)

print("API KEY:", os.environ.get("XKIRO_API_KEY"))