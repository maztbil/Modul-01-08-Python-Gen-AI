from openai import OpenAI
import os
import json
from dotenv import load_dotenv

load_dotenv()

client = OpenAI(
    api_key=os.environ["OPENAI_API_KEY"],
    base_url="https://api.llmsrelay.com/v1"
)

# Tool schema
tools = [
    {
        "type": "function",
        "function": {
            "name": "get_model_info",
            "description": "Returns context window and pricing for a given LLM.",
            "parameters": {
                "type": "object",
                "properties": {
                    "model_name": {
                        "type": "string",
                        "description": "Model identifier."
                    }
                },
                "required": ["model_name"]
            }
        }
    }
]

# Function implementation
def get_model_info(model_name: str) -> dict:
    db = {
        "gpt-4o": {"context_k": 128, "cost_input": 2.50},
        "claude-opus-5": {"context_k": 200, "cost_input": 3.00},
    }
    return db.get(model_name, {"error": f"unknown model: {model_name}"})


messages = [
    {"role": "user", "content": "What is claude-opus-5's context window?"}
]

# First call
response = client.chat.completions.create(
    model="claude-opus-5",
    tools=tools,
    messages=messages
)

message = response.choices[0].message

# Check if tool is called
if message.tool_calls:
    tool_call = message.tool_calls[0]

    name = tool_call.function.name
    args = json.loads(tool_call.function.arguments)

    result = get_model_info(**args)

    print(f"Tool called: {name}({args})")
    print(f"Tool result: {result}")

    # Append messages
    messages.append(message)
    messages.append({
        "role": "tool",
        "tool_call_id": tool_call.id,
        "content": json.dumps(result)
    })

    # Second call
    final = client.chat.completions.create(
        model="claude-opus-5",
        messages=messages
    )

    print("\nFinal answer:")
    print(final.choices[0].message.content)

else:
    # kalau model langsung jawab tanpa tool
    print(message.content)