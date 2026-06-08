"""
Exercise 08 — Tools
===================
Define callable tools that LLMs can invoke to interact with the outside world.

Concepts introduced:
- @tool decorator — the simplest way to define a tool
- StructuredTool.from_function() — explicit schema definition
- bind_tools() — attach tools to an LLM for tool calling
- Tool calling flow — LLM decides which tool to call and with what arguments
- ToolMessage — the result returned to the LLM after tool execution
"""

import math
import json
from datetime import datetime
from langchain_core.tools import tool, StructuredTool
from langchain_core.messages import HumanMessage, ToolMessage
from shared.llm import get_llm


# --- Define tools using @tool decorator ---


@tool
def get_current_time() -> str:
    """Get the current date and time."""
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


@tool
def calculator(expression: str) -> str:
    """Evaluate a mathematical expression. Supports +, -, *, /, **, sqrt, sin, cos, etc.

    Args:
        expression: A mathematical expression like '2 + 3 * 4' or 'sqrt(16)'
    """
    try:
        # Safe eval with math functions
        allowed_names = {
            k: v
            for k, v in math.__dict__.items()
            if not k.startswith("_")
        }
        allowed_names["__builtins__"] = {}
        result = eval(expression, allowed_names, {})
        return f"Result: {result}"
    except Exception as e:
        return f"Error: {e}"


@tool
def word_counter(text: str) -> str:
    """Count words, characters, and sentences in a text.

    Args:
        text: The text to analyze
    """
    words = text.split()
    chars = len(text)
    sentences = text.count(".") + text.count("!") + text.count("?")
    return (
        f"Words: {len(words)}, Characters: {chars}, "
        f"Estimated sentences: {max(sentences, 1)}"
    )


# --- Define a tool using StructuredTool.from_function ---


def reverse_string(text: str) -> str:
    """Reverse a string."""
    return text[::-1]


reverse_tool = StructuredTool.from_function(
    func=reverse_string,
    name="reverse_string",
    description="Reverse a string. Input: the text to reverse.",
)


def demo_tool_definitions():
    """Inspect tool schemas — this is what the LLM sees."""
    print("\n" + "=" * 60)
    print("1. Tool Definitions")
    print("=" * 60)

    all_tools = [get_current_time, calculator, word_counter, reverse_tool]

    for t in all_tools:
        print(f"\nTool: {t.name}")
        print(f"  Description: {t.description}")
        if hasattr(t, "args_schema") and t.args_schema:
            schema = t.args_schema.model_json_schema()
            print(f"  Args: {json.dumps(schema.get('properties', {}), indent=4)}")


def demo_bind_tools():
    """Bind tools to an LLM — the LLM can now decide to call them."""
    print("\n" + "=" * 60)
    print("2. bind_tools() — Manual Tool Calling")
    print("=" * 60)

    llm = get_llm()
    tools = [get_current_time, calculator, word_counter]
    tools_by_name = {t.name: t for t in tools}

    llm_with_tools = llm.bind_tools(tools)

    # The LLM decides whether to call a tool or respond directly
    queries = [
        "What time is it?",
        "Calculate 15 * 7 + 3",
        "Count the words in: The quick brown fox jumps over the lazy dog",
    ]

    for query in queries:
        print(f"\n👤 Query: {query}")
        response = llm_with_tools.invoke([HumanMessage(content=query)])

        if response.tool_calls:
            for tc in response.tool_calls:
                print(f"  🔧 LLM wants to call: {tc['name']}({tc['args']})")
                tool = tools_by_name[tc["name"]]
                result = tool.invoke(tc["args"])
                print(f"  📋 Tool result: {result}")
        else:
            print(f"  🤖 Direct response: {response.content}")


def demo_tool_call_loop():
    """Full tool-calling loop: LLM calls tool → execute → feed back → LLM responds."""
    print("\n" + "=" * 60)
    print("3. Full Tool-Calling Loop")
    print("=" * 60)

    llm = get_llm()
    tools = [get_current_time, calculator, word_counter]
    tools_by_name = {t.name: t for t in tools}

    llm_with_tools = llm.bind_tools(tools)

    query = "What time is it, and what is 100 divided by 7?"
    print(f"\n👤 Query: {query}")

    messages = [HumanMessage(content=query)]

    # First call — LLM may request tool calls
    response = llm_with_tools.invoke(messages)
    messages.append(response)

    # Execute tool calls and add results
    for tc in response.tool_calls:
        print(f"  🔧 Calling: {tc['name']}({tc['args']})")
        tool = tools_by_name[tc["name"]]
        tool_result = tool.invoke(tc["args"])
        print(f"  📋 Result: {tool_result}")
        messages.append(ToolMessage(content=tool_result, tool_call_id=tc["id"]))

    # Second call — LLM synthesizes tool results into a final answer
    final_response = llm_with_tools.invoke(messages)
    print(f"\n🤖 Final answer: {final_response.content}")


def demo_error_handling():
    """How tools handle errors gracefully."""
    print("\n" + "=" * 60)
    print("4. Tool Error Handling")
    print("=" * 60)

    # Calculator already handles errors, but let's test edge cases
    bad_expressions = [
        "10 / 0",
        "sqrt(-1)",
        "2 +* 3",
    ]

    for expr in bad_expressions:
        result = calculator.invoke({"expression": expr})
        print(f"  calculator('{expr}') → {result}")


def main():
    print("=" * 60)
    print("Exercise 08: Tools")
    print("=" * 60)

    demo_tool_definitions()
    demo_bind_tools()
    demo_tool_call_loop()
    demo_error_handling()


if __name__ == "__main__":
    main()
