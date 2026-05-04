"""
Exercise 03: Stop Sequences and Response Prefill
Demonstrates two techniques:
- stop_sequences: halt generation when a specific token is reached
- prefill: inject an assistant turn prefix to steer the response format
"""
from dotenv import load_dotenv
from anthropic import Anthropic


load_dotenv()

client = Anthropic()
model = "claude-haiku-4-5-20251001"

messages = []
system_prompt = """
    Generate command examples showing how to call an API with curl.
"""


def add_user_message(content: str):
    messages.append({"role": "user", "content": content})


def add_assistant_message(content: str):
    messages.append({"role": "assistant", "content": content})


def chat(messages: list, system=system_prompt, temperature=0.0):
    with client.messages.stream(
        model=model,
        messages=messages,
        max_tokens=256,
        system=system,
        temperature=temperature,
        stop_sequences=["```"],
    ) as stream:
        for text in stream.text_stream:
            print(text, end="", flush=True)

    final_message = stream.get_final_message()
    return final_message.content[0].text


while True:
    user_input = input("\n You: ")
    add_user_message(user_input)

    # Prefill: force Claude to start inside a bash code block
    add_assistant_message("```bash")
    answer = chat(messages)
