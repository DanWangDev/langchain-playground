# Exercise 02: Prompt Templates / 练习 02：提示模板

## What You'll Learn / 你将学到

- **ChatPromptTemplate** — define a reusable message structure with placeholders
- **SystemMessage** vs **HumanMessage** — role-based message composition
- **MessagesPlaceholder** — a dynamic slot for conversation history
- **Few-shot prompting** — provide examples to guide the model's output format
- `.format_messages()` — render templates into concrete messages ready for the LLM

## Why Prompt Templates Matter / 为什么提示模板很重要

Writing prompts as raw strings doesn't scale. When you have variables (user input, conversation history, context documents), you need a structured way to compose messages. Prompt templates provide:

1. **Reusability** — define once, use with many different inputs
2. **Type safety** — templates validate that required variables are provided
3. **Role awareness** — system, human, and AI messages are handled differently by the model
4. **History integration** — `MessagesPlaceholder` makes conversation memory seamless

## How Prompt Templates Work / 提示模板的工作原理

```
Template Definition
┌──────────────────────────────────────┐
│ ChatPromptTemplate.from_messages([   │
│   ("system", "You are a {role}."),   │  ← Placeholders: {role}, {question}
│   ("human", "{question}"),           │
│ ])                                   │
└──────────────────────────────────────┘
         │
         │ .format_messages(role="Python expert", question="What is...")
         ▼
┌──────────────────────────────────────┐
│ [SystemMessage("You are a Python     │
│   expert."),                         │  ← Concrete messages
│  HumanMessage("What is...")]         │
└──────────────────────────────────────┘
         │
         │ llm.invoke(messages)
         ▼
     AIMessage (response)
```

### Message Roles

| Role | Class | Purpose |
|------|-------|---------|
| System | `SystemMessage` | Sets the AI's behavior, tone, and constraints |
| Human | `HumanMessage` | The user's input or question |
| AI | `AIMessage` | The model's previous responses (used in history) |
| Tool | `ToolMessage` | Results from tool calls (see Exercise 08) |

### MessagesPlaceholder

```python
MessagesPlaceholder(variable_name="history")
```

This is a slot that expands into zero or more messages at render time. It's essential for:
- **Conversation history**: The full chat log is inserted here
- **Few-shot examples**: Dynamically insert example exchanges
- **Multi-turn context**: Accumulate messages across turns without changing the template

### Few-Shot Prompting

Instead of describing what you want, **show examples**:

```
System: "Classify sentiment as positive, negative, or neutral.

Examples:
Input: I love this! → positive
Input: Terrible. → negative
Input: It's blue. → neutral"

Human: "Input: This is amazing! →"
```

The model learns the pattern from examples and follows it for new inputs. This is more reliable than describing the format in words alone.

## Key Concepts / 核心概念

### Template Syntax

- `{variable_name}` — simple string substitution
- `MessagesPlaceholder(variable_name="...")` — expands to a list of messages
- Tuples `("role", "content")` — shorthand for creating messages

### Composition Patterns

```python
# Basic: system + human
prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a {role}."),
    ("human", "{question}"),
])

# With history slot
prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful assistant."),
    MessagesPlaceholder(variable_name="history"),
    ("human", "{input}"),
])

# Few-shot: examples inline in system prompt
prompt = ChatPromptTemplate.from_messages([
    ("system", "Classify sentiment.\n\nExamples:\n{each_example}\n"),
    ("human", "{text}"),
])
```

## Gotchas / 常见陷阱

1. **Missing variables**: If a template expects `{name}` but you don't pass it to `.format_messages()`, the call raises a `KeyError`.
2. **MessagesPlaceholder must be a list**: The variable passed for a placeholder must be a list of `BaseMessage` objects, not a string.
3. **Few-shot examples count against context**: Every example consumes tokens. For long prompts, limit to 2-4 high-quality examples.
4. **Template is NOT the prompt**: `ChatPromptTemplate` is a factory — it produces messages only after `.format_messages()`. You can't invoke it directly.

---

# 练习 02：提示模板

## 你将学到

- **ChatPromptTemplate** — 定义带有占位符的可复用消息结构
- **SystemMessage** 与 **HumanMessage** — 基于角色的消息组合
- **MessagesPlaceholder** — 用于对话历史的动态插槽
- **少样本提示** — 提供示例来引导模型输出格式
- `.format_messages()` — 将模板渲染为可供 LLM 使用的具体消息

## 为什么提示模板很重要

手写原始字符串的提示词无法规模化。当你有变量（用户输入、对话历史、上下文文档）时，你需要一种结构化的方式来组合消息。提示模板提供：

1. **可复用性** — 定义一次，可用于多种不同的输入
2. **类型安全** — 模板验证是否提供了必需的变量
3. **角色感知** — 系统、用户和 AI 消息在模型中被区别对待
4. **历史集成** — `MessagesPlaceholder` 使对话记忆无缝衔接

## 消息角色

| 角色 | 类名 | 用途 |
|------|------|------|
| System | `SystemMessage` | 设定 AI 的行为、语气和约束 |
| Human | `HumanMessage` | 用户的输入或问题 |
| AI | `AIMessage` | 模型之前的回复（用于历史） |
| Tool | `ToolMessage` | 工具调用的结果（参见练习 08） |

## 少样本提示

不是用语言描述你想要的，而是**提供示例**：

```
System: "判断情感倾向：正面、负面或中性。

示例：
输入：太棒了！→ 正面
输入：太糟糕了。→ 负面
输入：它是蓝色的。→ 中性"

Human: "输入：太神奇了！→"
```

模型从示例中学习模式并将其应用于新输入。这比仅用文字描述格式更可靠。

## 常见陷阱

1. **缺少变量**：如果模板期望 `{name}` 但你没有传给它，调用将抛出 `KeyError`。
2. **MessagesPlaceholder 必须是列表**：为占位符传入的变量必须是 `BaseMessage` 对象列表，不能是字符串。
3. **少样本示例也占用上下文**：每个示例都消耗 Token。对于较长的提示词，限制在 2-4 个高质量示例。
4. **模板不是提示词**：`ChatPromptTemplate` 是一个工厂——只有在 `.format_messages()` 之后才会生成消息。你不能直接调用它。
