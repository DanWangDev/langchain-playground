"""
Exercise 09 — Agents
====================
Build autonomous agents that plan and execute multi-step tasks using tools.

Concepts introduced:
- create_agent — LangGraph's prebuilt ReAct agent
- Agent loop — think → act → observe → repeat
- Tool-calling agent — modern approach using native function calling
- Agent scratchpad — internal working memory during an agent run
- Recursion limit — prevent infinite loops
- Agent state — messages, intermediate steps

Agent Pattern:
    User: "What's the weather in Paris?"
    Agent: [Thinks] I need to call get_weather("Paris")
    Agent: [Calls] get_weather("Paris") → "Sunny, 22°C"
    Agent: [Responds] "The weather in Paris is sunny and 22°C."
"""

import math
from datetime import datetime
from langchain_core.tools import tool
from langchain.agents import create_agent
from shared.llm import get_llm


# --- Define tools for the agent ---


@tool
def get_current_time() -> str:
    """Get the current date and time."""
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


@tool
def calculator(expression: str) -> str:
    """Evaluate a mathematical expression. Use for any math calculation.

    Args:
        expression: A mathematical expression like '2+3' or '100/7'
    """
    try:
        allowed = {k: v for k, v in math.__dict__.items() if not k.startswith("_")}
        allowed["__builtins__"] = {}
        return str(eval(expression, allowed, {}))
    except Exception as e:
        return f"Error: {e}"


@tool
def word_length(word: str) -> str:
    """Get the length of a word and whether it's short, medium, or long.

    Args:
        word: A single word to analyze
    """
    length = len(word)
    if length <= 4:
        category = "short"
    elif length <= 8:
        category = "medium"
    else:
        category = "long"
    return f"'{word}' has {length} characters — {category}"


def demo_basic_agent():
    """Create a simple ReAct agent and run a single query."""
    print("\n" + "=" * 60)
    print("1. Basic Agent — Single Tool Call")
    print("=" * 60)

    llm = get_llm()
    tools = [get_current_time, calculator, word_length]

    agent = create_agent(llm, tools)

    result = agent.invoke(
        {"messages": [("user", "What time is it right now?")]}
    )

    # Print the conversation
    for msg in result["messages"]:
        role = msg.__class__.__name__.replace("Message", "")
        content = getattr(msg, "content", "")
        tool_calls = getattr(msg, "tool_calls", [])
        if content:
            print(f"  [{role}] {str(content)[:200]}")
        for tc in tool_calls:
            print(f"  [{role} → tool] {tc['name']}({tc['args']})")


def demo_multi_step_agent():
    """Agent that needs multiple tool calls to answer."""
    print("\n" + "=" * 60)
    print("2. Multi-Step Agent — Multiple Tool Calls")
    print("=" * 60)

    llm = get_llm()
    tools = [get_current_time, calculator, word_length]

    agent = create_agent(llm, tools)

    # This requires: calculator → word_length → synthesize
    query = (
        "First, calculate 15 * 7. Then, take the word 'LangChain' and tell me "
        "its length. Finally, tell me which is larger: the result of 15*7 or "
        "the length of 'LangChain'?"
    )

    print(f"\n👤 Query: {query}\n")

    result = agent.invoke({"messages": [("user", query)]})

    for msg in result["messages"]:
        role = msg.__class__.__name__.replace("Message", "")
        content = getattr(msg, "content", "")
        tool_calls = getattr(msg, "tool_calls", [])
        if content and role not in ("Tool",):
            print(f"[{role}] {str(content)[:300]}")
        for tc in tool_calls:
            print(f"[{role} → 🔧] {tc['name']}({tc['args']})")


def demo_agent_with_system_prompt():
    """Customize agent behavior with a system prompt."""
    print("\n" + "=" * 60)
    print("3. Agent with System Prompt")
    print("=" * 60)

    llm = get_llm()
    tools = [calculator, word_length]

    system_prompt = (
        "You are a MATH TUTOR who explains every step clearly. "
        "Always show your work before giving the final answer. "
        "Use the calculator tool for calculations and explain what you did."
    )

    agent = create_agent(llm, tools, system_prompt=system_prompt)

    query = "What is 256 divided by 8, and then multiplied by 3?"
    print(f"\n👤 Query: {query}\n")

    result = agent.invoke({"messages": [("user", query)]})

    # Print just the assistant's final response
    for msg in reversed(result["messages"]):
        if hasattr(msg, "content") and msg.content and not hasattr(msg, "tool_calls"):
            print(f"🤖 Tutor: {msg.content}")
            break


def demo_recursion_limit():
    """Demonstrate the recursion limit safety mechanism."""
    print("\n" + "=" * 60)
    print("4. Recursion Limit — Safety Mechanism")
    print("=" * 60)

    llm = get_llm()
    tools = [calculator]

    # Create agent with a low recursion limit
    agent = create_agent(llm, tools)

    # Set a low recursion limit to demonstrate the safety net
    config = {"recursion_limit": 3}
    query = (
        "Calculate 1+1, then 2+2, then 3+3, then 4+4, then 5+5. "
        "Show each result."
    )

    print(f"\nRecursion limit: {config['recursion_limit']}")
    print(f"Query: {query}\n")

    try:
        result = agent.invoke({"messages": [("user", query)]}, config=config)
        for msg in result["messages"]:
            if hasattr(msg, "content") and msg.content and not hasattr(msg, "tool_calls"):
                print(f"[{msg.__class__.__name__}] {msg.content[:200]}")
    except Exception as e:
        print(f"Agent stopped: {e}")


def main():
    print("=" * 60)
    print("Exercise 09: Agents")
    print("=" * 60)

    demo_basic_agent()
    demo_multi_step_agent()
    demo_agent_with_system_prompt()
    demo_recursion_limit()


if __name__ == "__main__":
    main()
