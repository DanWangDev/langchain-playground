# Exercise 09: Agents

## What You'll Learn

- **create_agent** — LangGraph's prebuilt ReAct agent factory
- **Agent loop** — think → act → observe → repeat until done
- **Tool-calling agent** — modern approach using native function calling
- **System prompts** — customize agent behavior and personality
- **Recursion limit** — safety mechanism to prevent infinite loops
- **Agent state** — messages, intermediate steps, scratchpad

## Why Agents Matter

A chain follows a fixed path. An agent **decides** its own path.

```
Chain:   A → B → C → D   (predetermined)

Agent:   A → [think] → B? C? Done?
              ↑___________|   (dynamic, tool-driven)
```

Agents are the foundation of autonomous AI systems. They can:
1. **Plan** — break down complex tasks into steps
2. **Use tools** — call APIs, run calculations, search databases
3. **Adapt** — change strategy based on intermediate results
4. **Iterate** — retry with different approaches if something fails

## The Agent Loop

```
┌────────────────────────────────────────────────┐
│                 Agent Loop                       │
│                                                  │
│  ┌──────────┐     ┌──────────┐     ┌──────────┐ │
│  │  THINK   │────▶│   ACT    │────▶│ OBSERVE  │ │
│  │ (LLM)    │     │ (Tool)   │     │ (Result) │ │
│  └──────────┘     └──────────┘     └──────────┘ │
│       ▲                                  │       │
│       └──────────────────────────────────┘       │
│               Repeat until DONE                  │
└────────────────────────────────────────────────┘
```

### Example: Multi-Step Agent

```
User: "Calculate 15*7, then tell me the length of 'LangChain',
      and which is larger."

→ THINK: I need calculator("15*7") and word_length("LangChain")
→ ACT: calculator("15*7") → "105"
→ ACT: word_length("LangChain") → "'LangChain' has 9 characters"
→ OBSERVE: Results = {calc: 105, word: 9}
→ THINK: 105 > 9, so the calculation result is larger
→ RESPOND: "15*7 = 105, 'LangChain' has 9 characters. 105 is larger."
```

### Creating an Agent

```python
from langchain.agents import create_agent

agent = create_agent(
    llm,
    tools=[calculator, get_current_time, word_length],
    system_prompt="You are a helpful math tutor. Explain each step.",
)

result = agent.invoke({"messages": [("user", "What is 256 / 8 * 3?")]})
```

### With System Prompt

The system prompt defines the agent's **persona** and **constraints**:

```python
system_prompt = (
    "You are a MATH TUTOR who explains every step clearly. "
    "Always show your work before giving the final answer. "
    "Use the calculator tool for calculations."
)
```

This transforms a generic tool-user into a specialized tutor. The system prompt is the most powerful lever for controlling agent behavior.

### Recursion Limit

Agents can get stuck in loops if a tool keeps returning unexpected results:

```
THINK → call tool → unexpected result → THINK → call same tool → ...
```

The recursion limit stops this:

```python
config = {"recursion_limit": 3}
agent.invoke({"messages": [...]}, config=config)
```

After 3 LLM calls, the agent stops — even if it hasn't produced a final answer. This is a critical safety net.

## Key Concepts

### create_agent vs Manual Tool-Calling Loop

| Approach | When to Use |
|----------|-------------|
| `create_agent` (LangGraph) | Multi-step tasks, autonomous decision-making, complex workflows |
| Manual bind_tools() loop (Ex 08) | Single tool call, simple orchestration, maximum control |

`create_agent` wraps the full ReAct loop. You don't write the loop yourself — LangGraph handles tool execution, message management, and stopping conditions.

### Agent State

The agent maintains internal state as it works:

```python
{
    "messages": [
        HumanMessage("What is 15*7?"),
        AIMessage(tool_calls=[...]),
        ToolMessage("105"),
        AIMessage("15*7 = 105"),
    ],
    # Intermediate steps, scratchpad, etc.
}
```

You can inspect `result["messages"]` to see the full reasoning trace.

## Gotchas

1. **Recursion limit too low**: Complex tasks need more steps. Default is usually 25. Setting it to 3 may stop agents mid-task.
2. **Stuck in tool loops**: If a tool returns unexpected format, the agent may retry endlessly. Good tool descriptions and error returns prevent this.
3. **System prompt is key**: A poorly written system prompt makes the agent wander. Be specific about what the agent should and shouldn't do.
4. **create_agent is from langchain.agents**: Not `langgraph.prebuilt` (deprecated `create_react_agent`). Use the new import path.
5. **Agent messages include all intermediate steps**: The result contains every tool call and intermediate thought. Filter for final content when displaying to users.
