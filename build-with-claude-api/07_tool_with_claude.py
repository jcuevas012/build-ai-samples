# Load env variables and create client
from dotenv import load_dotenv
from anthropic import Anthropic
from tool_schema import add_duration_to_datetime, set_reminder, get_current_time, add_duration_to_datetime_schema, set_reminder_schema, batch_tool_schema, get_current_time_schema
from anthropic.types import Message
import json

load_dotenv()

client = Anthropic()
model = "claude-haiku-4-5"


# Helper functions
def add_user_message(messages, message):
    user_message = {
        "role": "user", 
        "content": message.content if isinstance(message, Message) else message, 
    }
    messages.append(user_message)


def add_assistant_message(messages, message):
    assistant_message = {
        "role": "assistant",
        "content": message.content if isinstance(message, Message) else message
    }
    messages.append(assistant_message)



def text_from_message(message):
    return "\n".join([block.text for block in message.content if block.type == "text"])

def chat(messages, system=None, temperature=1.0, stop_sequences=[], tools=[]):
    params = {
        "model": model,
        "max_tokens": 1000,
        "messages": messages,
        "temperature": temperature,
        "stop_sequences": stop_sequences,
        "tools": tools,
    }

    if system:
        params["system"] = system

    message = client.messages.create(**params)
    return message

def run_tool(tool_name, tool_args):
    if tool_name == "add_duration_to_datetime":
        result = add_duration_to_datetime(**tool_args)
    elif tool_name == "set_reminder":
        result = set_reminder(**tool_args)
    elif tool_name == "get_current_time":
        result = get_current_time()
    else:
        raise ValueError(f"Unknown tool: {tool_name}")

    return result

def run_tools(message):
    print("Running tools...")

    tool_requests = [ block for block in message.content if block.type == "tool_use" ]

    tool_result_bocks = []
    for tool in tool_requests:
        result = run_tool(tool.name, tool.input)
        tool_result_bock = {
                     "type": "tool_result",
                     "tool_use_id": tool.id,
                     "content": json.dumps(result),
                     "is_error": False,
        }
        tool_result_bocks.append(tool_result_bock)

    return tool_result_bocks


def run_conversation(messages):
    while True:
        response = chat(messages, tools=[add_duration_to_datetime_schema, set_reminder_schema, get_current_time_schema, batch_tool_schema])

        add_assistant_message(messages, response)
        print(text_from_message(response))

        if response.stop_reason != "tool_use":
            break

        tool_results = run_tools(response)
        add_user_message(messages, tool_results)

    print("Conversation ended.", messages)
    return messages


messages=[]

add_user_message(messages, "What is currant day? and add two days to it. Then set a reminder for that day with content 'Meeting with team'")
run_conversation(messages)
