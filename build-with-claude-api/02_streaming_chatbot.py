"""
Exercise 02: Streaming Chatbot with System Prompt
Demonstrates how to build an interactive chatbot using streaming responses
and a system prompt to give Claude a specific persona and context.
"""
from dotenv import load_dotenv
from anthropic import Anthropic


load_dotenv()

client = Anthropic()
model = "claude-sonnet-4-6"

messages = []
system_prompt = """
    You are a helpful assistant for insurance company in DR considering the following information:
    - The company is called "DR Insurance"
    - We base on insurance policies of superintendencia de seguros de la republica dominicana (SISAR) https://sis.gob.do/
    - We want to provide service in simple manner to our customers, services like new insurance policy, claim insurance, check insurance policy, and renew insurance etc.
    - For now we just focus on car insurance and we want to acquire car information and customer to know about car insurance policies and services.
    - Before we suggest a plan for customer we review the car information and customer information to suggest the best plan for them.
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
    ) as stream:
        for text in stream.text_stream:
            print(text, end="", flush=True)

    final_message = stream.get_final_message()
    return final_message.content[0].text


while True:
    user_input = input("\n You: ")
    add_user_message(user_input)

    answer = chat(messages)
    add_assistant_message(answer)
