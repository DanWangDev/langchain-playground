
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

