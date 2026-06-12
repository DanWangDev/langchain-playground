# Exercise 03: Chains & LCEL / 练习 03：链与 LCEL 表达式

## What You'll Learn / 你将学到

- **LCEL pipe operator** (`|`) — compose components into pipelines
- **StrOutputParser** — extract plain string from AIMessage
- **RunnableParallel** — run multiple chains concurrently
- **RunnableLambda** — wrap any Python function as a chain-compatible step
- **RunnablePassthrough** — pass input through unchanged
- `.assign()` — enrich input dicts with computed fields
- `.bind()` — pre-set parameters on a runnable (e.g., temperature)

## Why LCEL Matters / 为什么 LCEL 很重要

LangChain Expression Language (LCEL) is the **universal composition language** of LangChain. Everything that is "Runnable" (LLMs, prompts, parsers, custom functions) can be connected with `|`.

```
component_a | component_b | component_c
```

This is declarative, readable, and automatically handles:
- **Streaming**: Each component streams its output to the next
- **Async**: `.ainvoke()` works end-to-end
- **Parallelism**: `RunnableParallel` fans out automatically
- **Batching**: `.batch()` optimizes across all steps

Without LCEL, you'd write imperative glue code for each of these concerns. With LCEL, you compose once and get all execution modes for free.

## How Chains Work / 链的工作原理

```
Input Dict
    │
    ▼
┌──────────┐     ┌──────────┐     ┌──────────┐
│  Prompt  │────▶│   LLM    │────▶│  Parser  │────▶ Output String
│ Template │     │          │     │          │
└──────────┘     └──────────┘     └──────────┘
  {topic} →       AIMessage  →    StrOutputParser
  messages         .content        extracts string
```

### The Pipe Operator

```python
chain = prompt | llm | StrOutputParser()
result = chain.invoke({"topic": "vector databases"})
```

Each `|` connects an output to an input. Data flows left to right. The output of one step becomes the input of the next.

### RunnableParallel

```python
multi = RunnableParallel(
    summary=summary_chain,   # These three chains
    pros=pros_chain,         # run at the same time
    cons=cons_chain,         # on the same input
)
result = multi.invoke({"topic": "microservices"})
# result = {"summary": "...", "pros": "...", "cons": "..."}
```

This is the key to efficient multi-perspective analysis — all branches execute concurrently.

### RunnableLambda

```python
RunnableLambda(lambda text: f"Response ({len(text.split())} words): {text}")
```

Wraps any function so it can participate in a chain. Use when you need custom logic between steps: transform, validate, log, or format data.

### RunnablePassthrough

```python
RunnablePassthrough.assign(
    word_count=lambda x: len(x["text"].split()),
)
```

Passes the input through unchanged while computing additional fields. This is the **immutable enrichment** pattern — you never mutate the original input, you always return a new dict with added keys.

## Key Concepts / 核心概念

### The Runnable Interface

Every LCEL component implements the `Runnable` protocol:

| Method | Description |
|--------|-------------|
| `.invoke(input)` | Single synchronous call |
| `.ainvoke(input)` | Single async call |
| `.stream(input)` | Synchronous streaming |
| `.astream(input)` | Async streaming |
| `.batch(inputs)` | Parallel batch processing |

When you chain with `|`, you create a new `Runnable` that supports all of these.

### Data Flow

```
dict → PromptTemplate → [SystemMessage, HumanMessage] → ChatOpenAI → AIMessage → StrOutputParser → str
```

Each step transforms the data type. Understanding the types at each stage helps you debug chains.

### .bind() vs .assign()

| Method | What It Does |
|--------|-------------|
| `.bind(temperature=0.0)` | Pre-sets **LLM parameters** for a specific runnable instance |
| `.assign(field=func)` | Adds **computed fields** to the input dict as it passes through |

`.bind()` configures HOW a component runs. `.assign()` enriches WHAT data flows through.

## Gotchas / 常见陷阱

1. **Input dict keys must match template variables**: If your prompt has `{topic}` but you pass `{"subject": "AI"}`, you'll get a `KeyError`.
2. **RunnableParallel preserves all keys**: Every branch's output becomes a key in the result dict. Key names must be unique.
3. **RunnableLambda must return valid input for next step**: If the next component expects a dict, your lambda must return a dict.
4. **Order matters in .assign()**: Earlier assignments run first. An assignment can reference values from earlier ones in the same chain.

---

# 练习 03：链与 LCEL 表达式

## 你将学到

- **LCEL 管道操作符**（`|`）— 将组件组合成流水线
- **StrOutputParser** — 从 AIMessage 提取纯文本
- **RunnableParallel** — 并发运行多条链
- **RunnableLambda** — 将任何 Python 函数包装为可链接的步骤
- **RunnablePassthrough** — 原样传递输入
- `.assign()` — 用计算字段丰富输入字典
- `.bind()` — 预设运行参数（如 temperature）

## 为什么 LCEL 很重要

LangChain 表达式语言（LCEL）是 LangChain 的**通用组合语言**。所有 "Runnable" 的东西（LLM、提示词、解析器、自定义函数）都可以用 `|` 连接起来。这是声明式的、可读的，并且自动处理流式输出、异步、并行和批处理。

## Runnable 接口

每个 LCEL 组件都实现了 `Runnable` 协议：

| 方法 | 说明 |
|--------|------|
| `.invoke(input)` | 单次同步调用 |
| `.ainvoke(input)` | 单次异步调用 |
| `.stream(input)` | 同步流式输出 |
| `.astream(input)` | 异步流式输出 |
| `.batch(inputs)` | 并行批处理 |

## 数据流

```
dict → PromptTemplate → [SystemMessage, HumanMessage] → ChatOpenAI → AIMessage → StrOutputParser → str
```

每一步都转换数据类型。理解每个阶段的数据类型有助于调试链。

## .bind() vs .assign()

| 方法 | 作用 |
|--------|------|
| `.bind(temperature=0.0)` | 为特定实例**预设 LLM 参数** |
| `.assign(field=func)` | 在传递过程中为输入字典**添加计算字段** |

`.bind()` 配置组件**如何**运行。`.assign()` 丰富**什么**数据在流动。

## 常见陷阱

1. **输入字典的键必须匹配模板变量**：如果提示词有 `{topic}` 但你传入 `{"subject": "AI"}`，将得到 `KeyError`。
2. **RunnableParallel 保留所有键**：每个分支的输出成为结果字典中的一个键。键名必须唯一。
3. **RunnableLambda 必须返回下一步的有效输入**：如果下一个组件期望一个 dict，你的 lambda 必须返回一个 dict。
4. **.assign() 中的顺序很重要**：先声明的赋值先运行。后面的赋值可以引用前面的结果。
