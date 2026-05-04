"""
Exercise 01: Multi-Turn Conversation
Demonstrates how to maintain conversation history across multiple API calls
to create a stateful back-and-forth exchange with Claude.
"""
from dotenv import load_dotenv
from anthropic import Anthropic


load_dotenv()

client = Anthropic()
model = "claude-sonnet-4-6"

messages = []


def add_user_message(content: str):
    messages.append({"role": "user", "content": content})


def add_assistant_message(content: str):
    messages.append({"role": "assistant", "content": content})


def chat(messages: list):
    message = client.messages.create(
        model=model,
        messages=messages,
        max_tokens=256,
    )
    return message.content[0].text


add_user_message("What is Quantum Computing?, Answer in 100 words or less.")

answer = chat(messages)
print(answer)
add_assistant_message(answer)

add_user_message("Write another sentence.")

follow_up_answer = chat(messages)
print("<<<<<<<<=============>>>>>>>>")
print(follow_up_answer)
