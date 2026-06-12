# Exercise 09: Agents / 练习 09：智能体

## What You'll Learn / 你将学到

- **create_agent** — LangGraph's prebuilt ReAct agent factory
- **Agent loop** — think → act → observe → repeat until done
- **Tool-calling agent** — modern approach using native function calling
- **System prompts** — customize agent behavior and personality
- **Recursion limit** — safety mechanism to prevent infinite loops
- **Agent state** — messages, intermediate steps, scratchpad

## Why Agents Matter / 为什么智能体很重要

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

## The Agent Loop / 智能体循环

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

## Key Concepts / 核心概念

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

## Gotchas / 常见陷阱

1. **Recursion limit too low**: Complex tasks need more steps. Default is usually 25. Setting it to 3 may stop agents mid-task.
2. **Stuck in tool loops**: If a tool returns unexpected format, the agent may retry endlessly. Good tool descriptions and error returns prevent this.
3. **System prompt is key**: A poorly written system prompt makes the agent wander. Be specific about what the agent should and shouldn't do.
4. **create_agent is from langchain.agents**: Not `langgraph.prebuilt` (deprecated `create_react_agent`). Use the new import path.
5. **Agent messages include all intermediate steps**: The result contains every tool call and intermediate thought. Filter for final content when displaying to users.

---

# 练习 09：智能体

## 你将学到

- **create_agent** — LangGraph 预构建的 ReAct 智能体工厂
- **智能体循环** — 思考 → 行动 → 观察 → 重复直到完成
- **工具调用智能体** — 使用原生函数调用的现代方法
- **系统提示词** — 自定义智能体行为和个性
- **递归限制** — 防止无限循环的安全机制
- **智能体状态** — 消息、中间步骤、草稿板

## 为什么智能体很重要

链遵循固定路径。智能体**自主决定**自己的路径。智能体是自主 AI 系统的基础。它们可以规划、使用工具、根据中间结果调整策略、失败时以不同方法重试。

## 智能体循环

```
思考（LLM）→ 行动（工具）→ 观察（结果）→ 重复直到完成
```

## create_agent vs 手动工具调用循环

| 方法 | 何时使用 |
|------|----------|
| `create_agent`（LangGraph） | 多步骤任务、自主决策、复杂工作流 |
| 手动 bind_tools() 循环（练习 08） | 单次工具调用、简单编排、最大控制权 |

`create_agent` 封装了完整的 ReAct 循环。你不需要自己写循环——LangGraph 处理工具执行、消息管理和停止条件。

## 常见陷阱

1. **递归限制太低**：复杂任务需要更多步骤。默认通常是 25。设为 3 可能会在任务中途停止智能体。
2. **陷入工具循环**：如果工具返回意外格式，智能体可能无限重试。良好的工具描述和错误返回可以防止这种情况。
3. **系统提示词是关键**：写得不清晰的系统提示词会让智能体偏离目标。明确说明智能体应该做什么和不应该做什么。
4. **create_agent 来自 langchain.agents**：不是 `langgraph.prebuilt`（已弃用的 `create_react_agent`）。使用新的导入路径。
