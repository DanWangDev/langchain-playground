# Exercise 08: Tools / 练习 08：工具

## What You'll Learn / 你将学到

- **@tool decorator** — the simplest way to define a callable tool
- **StructuredTool.from_function()** — explicit schema definition for tools
- **bind_tools()** — attach tools to an LLM for tool calling
- **Tool calling flow** — LLM decides which tool to call, with what arguments
- **ToolMessage** — the result returned to the LLM after tool execution
- **Error handling** — graceful failure when tools encounter invalid input

## Why Tools Matter / 为什么工具很重要

LLMs are powerful reasoning engines, but they're trapped in their training data. They can't:

- Look up the current time
- Query a database
- Call an API
- Run calculations (reliably)
- Read files from disk

**Tools break the LLM out of its sandbox.** They connect the reasoning capability of an LLM to the outside world.

```
User: "What's 15 * 7 + current temperature in Tokyo?"
LLM: [Thinks] I need: calculator("15*7"), get_weather("Tokyo")
LLM: [Calls tools] → Result: 105, "22°C"
LLM: [Responds] "15*7 = 105, and it's currently 22°C in Tokyo."
```

## How Tools Work / 工具的工作原理

```
┌──────────────────────────────────────────────────────┐
│                  Tool Calling Flow                    │
│                                                       │
│  1. User sends query                                  │
│  2. LLM decides: respond directly OR call tool(s)     │
│  3. If tool call: execute tool, get result            │
│  4. Feed ToolMessage back to LLM                      │
│  5. LLM synthesizes final answer                      │
└──────────────────────────────────────────────────────┘
```

### Defining a Tool with @tool

```python
from langchain_core.tools import tool

@tool
def calculator(expression: str) -> str:
    """Evaluate a mathematical expression. Supports +, -, *, /, sqrt, etc.

    Args:
        expression: A mathematical expression like '2 + 3 * 4'
    """
    try:
        result = eval(expression, safe_namespace)
        return f"Result: {result}"
    except Exception as e:
        return f"Error: {e}"
```

The **docstring** is critical — it's what the LLM reads to understand what the tool does and when to use it.

### What the LLM Sees

When you `bind_tools([calculator, get_current_time])`, the LLM receives tool definitions like:

```json
{
  "name": "calculator",
  "description": "Evaluate a mathematical expression...",
  "parameters": {
    "expression": {"type": "string", "description": "A mathematical expression like '2+3'"}
  }
}
```

The LLM uses these schemas to decide: _should I call a tool, and if so, which one, with what arguments?_

### Manual Tool-Calling Loop

```python
llm_with_tools = llm.bind_tools(tools)
messages = [HumanMessage(content="What is 100 / 7?")]

# Step 1: LLM decides to call tool
response = llm_with_tools.invoke(messages)
# response.tool_calls = [{"name": "calculator", "args": {"expression": "100/7"}}]

# Step 2: Execute the tool
for tc in response.tool_calls:
    tool = tools_by_name[tc["name"]]
    tool_result = tool.invoke(tc["args"])
    messages.append(ToolMessage(content=tool_result, tool_call_id=tc["id"]))

# Step 3: LLM synthesizes final answer
final = llm_with_tools.invoke(messages)
# final.content = "100 divided by 7 is approximately 14.2857"
```

### StructuredTool.from_function()

For more control over the tool schema:

```python
def reverse_string(text: str) -> str:
    return text[::-1]

reverse_tool = StructuredTool.from_function(
    func=reverse_string,
    name="reverse_string",
    description="Reverse a string. Input: the text to reverse.",
)
```

Use this when you need a custom name, description, or args_schema that differs from the function signature.

## Key Concepts / 核心概念

### @tool vs StructuredTool

| Feature | @tool | StructuredTool.from_function() |
|---------|-------|-------------------------------|
| Schema inference | Automatic from type hints | Explicit control |
| Name | Function name | Custom name |
| Docstring usage | Used as description | Separate description + args description |
| Best for | Simple tools | Tools needing custom schemas |

### ToolMessage

```python
ToolMessage(content="Result: 14.2857", tool_call_id="call_abc123")
```

The `tool_call_id` links the result back to the specific tool call that produced it. This matters when the LLM makes multiple parallel tool calls — each result must be matched to the correct call.

### Safe eval()

When implementing `calculator`, NEVER use `eval()` on raw user input:

```python
# DANGEROUS: eval("__import__('os').system('rm -rf /')")
# SAFE: Restrict available names and remove builtins
allowed_names = {k: v for k, v in math.__dict__.items() if not k.startswith("_")}
allowed_names["__builtins__"] = {}
result = eval(expression, allowed_names, {})
```

## Gotchas / 常见陷阱

1. **Tool descriptions drive behavior**: The LLM decides to call a tool based on its description. Vague descriptions → wrong tool chosen or tool not called at all.
2. **bind_tools() requires supported models**: Not all models support native tool-calling. Older models need prompt-based tool descriptions.
3. **ToolMessage must be in messages list**: The LLM needs to see the tool result. Forgetting to append `ToolMessage` means the LLM won't know the result.
4. **tool_call_id must match**: When constructing `ToolMessage`, the `tool_call_id` must match the ID from the LLM's `tool_calls`. Mismatched IDs may cause the LLM to ignore the result.
5. **Tool errors should return strings, not raise exceptions**: If a tool raises an exception, the agent crashes. Return error strings so the LLM can handle the failure gracefully.

---

# 练习 08：工具

## 你将学到

- **@tool 装饰器** — 定义可调用工具的最简方式
- **StructuredTool.from_function()** — 工具的显式 schema 定义
- **bind_tools()** — 将工具绑定到 LLM 以启用工具调用
- **工具调用流程** — LLM 决定调用哪个工具、带什么参数
- **ToolMessage** — 工具执行后返回给 LLM 的结果
- **错误处理** — 工具遇到无效输入时的优雅失败

## 为什么工具很重要

LLM 是强大的推理引擎，但它们受限于训练数据。它们无法查当前时间、查询数据库、调用 API、可靠地计算或读取文件。**工具打破了 LLM 的沙盒限制**，将推理能力与外部世界连接起来。

## 工具调用流程

1. 用户发送查询
2. LLM 决定：直接回复 OR 调用工具
3. 如果调用工具：执行工具，获取结果
4. 将 ToolMessage 反馈给 LLM
5. LLM 综合最终答案

## @tool vs StructuredTool

| 特性 | @tool | StructuredTool.from_function() |
|------|-------|-------------------------------|
| Schema 推断 | 从类型注解自动推断 | 显式控制 |
| 名称 | 使用函数名 | 自定义名称 |
| 文档字符串用途 | 用作描述 | 分离的描述 + 参数说明 |
| 最适合 | 简单工具 | 需要自定义 schema 的工具 |

## 常见陷阱

1. **工具描述驱动行为**：LLM 根据描述决定是否调用工具。模糊的描述 → 选错工具或不调用工具。
2. **bind_tools() 需要模型支持**：并非所有模型都支持原生工具调用。旧模型需要基于提示词的工具描述。
3. **ToolMessage 必须在消息列表中**：LLM 需要看到工具结果。忘记添加 `ToolMessage` 意味着 LLM 不知道结果。
4. **tool_call_id 必须匹配**：构造 `ToolMessage` 时，`tool_call_id` 必须与 LLM 的 `tool_calls` 中的 ID 匹配。不匹配的 ID 可能导致 LLM 忽略结果。
5. **工具错误应返回字符串而非抛异常**：如果工具抛异常，Agent 会崩溃。返回错误字符串让 LLM 优雅地处理失败。
