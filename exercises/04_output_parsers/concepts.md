# Exercise 04: Output Parsers / 练习 04：输出解析器

## What You'll Learn / 你将学到

- **StrOutputParser** — simplest parser: extract string from AIMessage
- **PydanticOutputParser** — parse LLM output into Pydantic models via format instructions
- **with_structured_output()** — modern approach using native tool-calling (preferred)
- **CommaSeparatedListOutputParser** — parse comma-separated values into a list
- **Error handling** — what happens when parsing fails and how to recover

## Why Output Parsers Matter / 为什么输出解析器很重要

LLMs return **text**. Your application needs **structured data**. Output parsers bridge this gap:

```
LLM Output: "{\"name\": \"Alice\", \"age\": 30}"
                │
                ▼
         Output Parser
                │
                ▼
Structured Data: Person(name="Alice", age=30)
```

Without parsers, you'd write fragile regex or `json.loads()` in every application. Parsers provide:
1. **Type safety** — Pydantic models validate types at runtime
2. **Error messages** — clear feedback when the LLM produces malformed output
3. **Retry logic** — some parsers can ask the LLM to fix its output
4. **Schema communication** — `get_format_instructions()` tells the LLM exactly what format to use

## Two Approaches / 两种方法

### 1. PydanticOutputParser (Classic — Format Instructions)

```
Prompt (with format_instructions) → LLM → JSON string → PydanticOutputParser → Pydantic model
```

The parser injects format instructions into the prompt telling the LLM to output JSON matching a schema. The LLM must follow these instructions precisely — if it adds extra text or malforms the JSON, parsing fails.

### 2. with_structured_output() (Modern — Tool-Calling)

```
Prompt → LLM (with structured output mode) → Pydantic model (directly)
```

Uses the model's native **tool-calling** or **JSON mode** capability. The LLM is constrained to produce valid JSON matching the schema. This is much more reliable because:
- The model's token sampling is constrained to valid outputs
- No format instructions needed in the prompt
- No fragile JSON parsing step

**Rule of thumb**: Always prefer `with_structured_output()`. Use `PydanticOutputParser` only for models that don't support tool-calling.

## How with_structured_output() Works / 工作原理

```python
from pydantic import BaseModel, Field

class Person(BaseModel):
    name: str = Field(description="Person's full name")
    age: int = Field(description="Person's age in years")

# The LLM now returns Person objects directly
structured_llm = llm.with_structured_output(Person)
person = structured_llm.invoke("My name is Sarah and I am 34.")
# person = Person(name="Sarah", age=34)
```

### In a Chain

```python
class SentimentResult(BaseModel):
    sentiment: str = Field(description="positive, negative, or neutral")
    confidence: float = Field(description="0.0 to 1.0")

chain = prompt | llm.with_structured_output(SentimentResult)
result = chain.invoke({"text": "I love this!"})
# result.sentiment == "positive", result.confidence == 0.95
```

## Key Concepts / 核心概念

### Pydantic Field Descriptions

```python
class MovieReview(BaseModel):
    rating: float = Field(description="Rating out of 10")  # ← This IS the prompt for the LLM
```

Field descriptions serve double duty: they document your code AND tell the LLM what to output. Write them clearly — the LLM reads them as instructions.

### Parser Chain Pattern

```
Input → PromptTemplate → LLM → OutputParser → Typed Data
```

The parser is always the LAST step in the chain. It transforms the raw LLM output into application-ready data.

### Error Handling

```python
try:
    result = chain.invoke(input)
except Exception as e:
    # Parser failed — log and handle gracefully
    print(f"Parse error: {e}")
```

Common failures:
- LLM wraps JSON in markdown code blocks (```json...```)
- LLM adds explanatory text before/after JSON
- LLM uses single quotes instead of double quotes
- Missing required fields
- Wrong types (string instead of number)

## Gotchas / 常见陷阱

1. **with_structured_output() requires tool-calling support**: Not all models support it. FakeLLM and older models fall back to JSON mode or fail.
2. **Field descriptions ARE your prompt**: The LLM sees `Field(description="...")` as instructions. Vague descriptions → vague outputs.
3. **Temperature matters**: Use `temperature=0.0` or low values for structured output — you want determinism, not creativity.
4. **Nested models work but add complexity**: Deeply nested Pydantic models can confuse the LLM. Keep schemas flat when possible.
5. **Streaming and structured output don't mix**: `with_structured_output()` returns the complete object, not tokens. You can't stream partial objects.

---

# 练习 04：输出解析器

## 你将学到

- **StrOutputParser** — 从 AIMessage 提取纯文本字符串的最简单解析器
- **PydanticOutputParser** — 通过格式指令将 LLM 输出解析为 Pydantic 模型（经典方法）
- **with_structured_output()** — 使用原生工具调用的现代方法（推荐）
- **CommaSeparatedListOutputParser** — 将逗号分隔的值解析为列表
- **错误处理** — 解析失败时会发生什么以及如何恢复

## 为什么输出解析器很重要

LLM 返回**文本**。你的应用程序需要**结构化数据**。输出解析器弥合了这一差距。没有解析器，你会在每个应用程序中写脆弱的正则表达式或 `json.loads()`。

## 两种方法

### PydanticOutputParser（经典方法）

```
Prompt（含格式指令）→ LLM → JSON 字符串 → PydanticOutputParser → Pydantic 模型
```

### with_structured_output()（现代方法 — 推荐）

```
Prompt → LLM（结构化输出模式）→ Pydantic 模型（直接）
```

使用模型原生的**工具调用**能力。模型被约束为生成匹配 schema 的有效 JSON。这可靠得多。

**经验法则**：始终优先使用 `with_structured_output()`。仅在不支持工具调用的模型上使用 `PydanticOutputParser`。

## 核心概念

### Pydantic Field 描述

```python
class MovieReview(BaseModel):
    rating: float = Field(description="满分 10 分的评分")  # ← 这本身就是给 LLM 的提示
```

Field 描述起双重作用：记录你的代码 AND 告诉 LLM 输出什么。写清楚——LLM 把它们当作指令来读。

## 常见陷阱

1. **with_structured_output() 需要模型支持工具调用**：并非所有模型都支持。FakeLLM 和旧模型会回退到 JSON 模式或直接失败。
2. **Field 描述就是你的提示词**：LLM 把 `Field(description="...")` 当作指令。模糊的描述 → 模糊的输出。
3. **Temperature 很重要**：结构化输出使用低值或 `temperature=0.0`——你需要的是确定性，不是创造性。
4. **流式输出和结构化输出不兼容**：`with_structured_output()` 返回完整对象，不是 Token。你不能流式输出部分对象。
