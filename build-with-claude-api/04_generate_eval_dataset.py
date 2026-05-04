"""
Exercise 04: Generate Evaluation Dataset
Uses Claude to generate a structured JSON dataset of tasks
that will be used to benchmark prompt quality in exercise 05.
"""
from dotenv import load_dotenv
from anthropic import Anthropic
import json


load_dotenv()

client = Anthropic()
model = "claude-haiku-4-5"

messages = []


def add_user_message(messages, text):
    messages.append({"role": "user", "content": text})


def add_assistant_message(messages, text):
    messages.append({"role": "assistant", "content": text})


def chat(messages, system=None, temperature=1.0, stop_sequences=[]):
    params = {
        "model": model,
        "max_tokens": 1000,
        "messages": messages,
        "temperature": temperature,
        "stop_sequences": stop_sequences,
    }

    if system:
        params["system"] = system

    message = client.messages.create(**params)
    return message.content[0].text


def generate_dataset():
    prompt = """
Generate a evaluation dataset for a prompt evaluation. The dataset will be used to evaluate prompts
that generate Python, JSON, or Regex specifically for AWS-related tasks. Generate an array of JSON objects,
each representing task that requires Python, JSON, or a Regex to complete.

Example output:
```json
[
    {
        "task": "Description of task",
        "format": "python" or "json" or "regex",
        "solution_criteria": "Criteria for evaluating the solution, e.g. correct output, valid syntax, etc."
    },
    ...additional
]
```

* Focus on tasks that can be solved by writing a single Python function, a single JSON object, or a regular expression.
* Focus on tasks that do not require writing much code

Please generate 3 objects.
"""
    add_user_message(messages, prompt)
    # Prefill to ensure output starts as valid JSON
    add_assistant_message(messages, "```json")
    response = chat(messages, stop_sequences=["```"])

    return json.loads(response)


dataset = generate_dataset()

with open("dataset.json", "w") as f:
    json.dump(dataset, f, indent=4)

print("Dataset saved to dataset.json")
print(json.dumps(dataset, indent=2))
