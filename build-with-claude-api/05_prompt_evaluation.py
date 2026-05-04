"""
Exercise 05: Prompt Evaluation with Model-as-Judge
Runs a set of tasks (from dataset.json) through a prompt, then grades each
output using two methods: syntax validation and a model-as-judge scorer.
"""
from dotenv import load_dotenv
from anthropic import Anthropic
import json
import ast
import re

load_dotenv()

client = Anthropic()
model = "claude-haiku-4-5-20251001"


# --- Validators ---

def validate_json(text):
    try:
        json.loads(text.strip())
        return 10
    except json.JSONDecodeError:
        return 0


def validate_python(text):
    try:
        ast.parse(text.strip())
        return 10
    except SyntaxError:
        return 0


def validate_regex(text):
    try:
        re.compile(text.strip())
        return 10
    except re.error:
        return 0


def grade_syntax(test_case, output):
    if test_case["format"] == "json":
        return validate_json(output)
    elif test_case["format"] == "python":
        return validate_python(output)
    elif test_case["format"] == "regex":
        return validate_regex(output)
    else:
        raise ValueError(f"Unknown format: {test_case['format']}")


# --- API helpers ---

def add_user_message(messages, content: str):
    messages.append({"role": "user", "content": content})


def add_assistant_message(messages, content: str):
    messages.append({"role": "assistant", "content": content})


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

    response = client.messages.create(**params)
    return response.content[0].text


# --- Evaluation pipeline ---

def run_prompt(test_case):
    """Send the task to Claude and return its raw output."""
    prompt = f"""
    Please resolve the following task: {test_case['task']}

    * Respond only with Python, JSON or a plain Regex
    * Do not add any comments or explanations, only the solution
    """
    messages = []
    add_user_message(messages, prompt)
    add_assistant_message(messages, "```code")
    output = chat(messages, stop_sequences=["```"])
    return output


def grade_by_model(test_case, output):
    """Use Claude as a judge to score the output against the task criteria."""
    eval_prompt = f"""
        You are an expert AWS code reviewer. Your task is to evaluate the following AI-generated solution.

        Original Task:
        <task>
        {test_case["task"]}
        </task>

        Solution to Evaluate:
        <solution>
        {output}
        </solution>

        Criteria should be used to evaluate the solution:
        <criteria>
        {test_case["solution_criteria"]}
        </criteria>

        Output Format
        Provide your evaluation as a structured JSON object with the following fields, in this specific order:
        - "strengths": An array of 1-3 key strengths
        - "weaknesses": An array of 1-3 key areas for improvement
        - "reasoning": A concise explanation of your overall assessment
        - "score": A number between 1-10

        Respond with JSON. Keep your response concise and direct.
        Example response shape:
        {{
            "strengths": string[],
            "weaknesses": string[],
            "reasoning": string,
            "score": number
        }}
    """

    messages = []
    add_user_message(messages, eval_prompt)
    add_assistant_message(messages, "```json")
    eval_response = chat(messages, stop_sequences=["```"])
    return json.loads(eval_response)


def run_test_case(test_case):
    """Run a single test case and return combined scores and reasoning."""
    output = run_prompt(test_case)

    model_grade = grade_by_model(test_case=test_case, output=output)
    syntax_score = grade_syntax(test_case, output)

    combined_score = (model_grade["score"] + syntax_score) / 2

    return {
        "output": output,
        "score": combined_score,
        "reasoning": model_grade["reasoning"],
        "strengths": model_grade["strengths"],
        "weaknesses": model_grade["weaknesses"],
        "test_case": test_case,
    }


def run_evaluation(cases):
    """Run all test cases and print a summary."""
    results = []
    for test_case in cases:
        result = run_test_case(test_case)
        results.append(result)

    average_score = sum(r["score"] for r in results) / len(results)
    print(f"Average Score: {average_score:.2f}")

    return results


with open("dataset.json", "r", encoding="utf-8") as f:
    test_cases = json.load(f)

evaluation_results = run_evaluation(test_cases)

print(json.dumps(evaluation_results, indent=4))
