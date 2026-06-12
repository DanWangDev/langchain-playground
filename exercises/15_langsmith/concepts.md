# Exercise 15: LangSmith / 练习 15：LangSmith 追踪与评估

## What You'll Learn / 你将学到

- **Automatic tracing** — zero code changes, just set environment variables
- **@traceable decorator** — granular span-level tracing for custom functions
- **Run names, tags, and metadata** — organize traces via `RunnableConfig`
- **Dataset creation** — build evaluation datasets from example inputs/outputs
- **Evaluation** — score chain outputs against expected answers
- **Comparative evaluation** — compare providers (DeepSeek vs Qwen) side by side

## Why LangSmith Matters / 为什么 LangSmith 很重要

LLM applications are **non-deterministic** — the same input can produce different outputs. Traditional logging and debugging don't work well. You need to see:

1. **Every step** — prompt → LLM → tool → LLM → output
2. **Timing** — where is time spent? Which step is slow?
3. **Token usage** — how much does each call cost?
4. **Errors** — exactly which step failed and why?
5. **Outputs over time** — is quality improving or degrading?

LangSmith is the observability platform built specifically for LLM applications. It's like Datadog, but designed for chains and agents.

## LangSmith Architecture / 架构

```
Your Code                    LangSmith Cloud
┌──────────────┐           ┌─────────────────────┐
│ chain.invoke │──────────▶│ smith.langchain.com │
│ @traceable   │  traces   │                     │
│ agent.run    │           │ ┌─────────────────┐ │
└──────────────┘           │ │  Trace Viewer   │ │
                            │ │  Datasets       │ │
                            │ │  Experiments    │ │
                            │ │  Annotation     │ │
                            │ └─────────────────┘ │
                            └─────────────────────┘
```

## How Tracing Works / 追踪原理

### 1. Automatic Tracing

Just set environment variables — no code changes:

```bash
LANGCHAIN_TRACING_V2=true
LANGCHAIN_API_KEY=ls_...
LANGCHAIN_PROJECT=langchain-playground
```

Every `.invoke()` call automatically sends a trace to LangSmith. Every chain step, LLM call, and tool execution is captured.

### 2. @traceable Decorator

For custom functions that aren't LangChain components:

```python
from langsmith import traceable

@traceable(run_type="chain", name="preprocess-input")
def preprocess(text: str) -> str:
    return text.strip().lower()

@traceable(run_type="chain", name="postprocess-output")
def postprocess(text: str) -> str:
    return text.capitalize()
```

These appear as named spans in the trace. When LangSmith is not configured, `@traceable` is a **no-op** — it passes through without any effect. This means you can decorate your code and it works identically with or without LangSmith.

### 3. Run Metadata

```python
from langchain_core.runnables import RunnableConfig

config = RunnableConfig(
    run_name="fact-dolphins",
    tags=["production", "science", "exercise-15"],
    metadata={
        "topic": "dolphins",
        "experiment": "langsmith-demo",
    },
)
result = chain.invoke({"topic": "dolphins"}, config=config)
```

Tags and metadata make traces searchable and filterable in the LangSmith UI.

## Evaluation / 评估

### Creating Datasets

```python
from langsmith import Client

client = Client()
dataset = client.create_dataset(
    dataset_name="langchain-basics-qa",
    description="Basic LangChain Q&A evaluation dataset",
)

client.create_examples(
    inputs=[{"question": "What is LangChain?"}],
    outputs=[{"answer": "A framework for building LLM-powered applications."}],
    dataset_id=dataset.id,
)
```

A dataset is a collection of **(input, expected_output)** pairs. You run your chain against each input and compare the output to the expected output.

### Evaluation Functions

```python
def evaluator(output: str, expected: str) -> dict:
    """Score output against expected answer."""
    key_terms = expected.lower().split()
    matches = sum(1 for t in key_terms if t in output.lower())
    score = matches / max(len(key_terms), 1)
    return {"score": round(score, 2), "key_terms_matched": matches}
```

LangSmith supports custom evaluators (like above) and built-in ones (correctness, helpfulness, toxicity).

### Comparative Evaluation

Run the same dataset against different models/configurations and compare:

```
DeepSeek (temp=0)  →  Score: 0.85 avg
Qwen (temp=0)      →  Score: 0.78 avg
DeepSeek (temp=0.3) → Score: 0.82 avg
```

This is how you make **data-driven model selection decisions** — not based on benchmarks, but on YOUR actual use case.

## Key Concepts / 核心概念

### Traces and Spans

```
Trace: "exercise-15-run"
├── Span: "preprocess-input" (@traceable)
├── Span: "ChatOpenAI" (automatic)
│   ├── prompt_tokens: 45
│   └── completion_tokens: 12
└── Span: "postprocess-output" (@traceable)
```

A **trace** is the full execution of a chain. **Spans** are individual steps within it.

### No-Op Mode

When `LANGCHAIN_TRACING_V2` is not set:
- `@traceable` is a pass-through (no overhead)
- No data leaves your machine
- Chains work identically

This makes it safe to add `@traceable` everywhere — it only activates when you want it.

### Free Tier

LangSmith offers a free tier: 3,000 traces/month. This is enough for development and light production use.

## Gotchas / 常见陷阱

1. **@traceable is a no-op without config**: If you're not seeing traces, check `LANGCHAIN_TRACING_V2=true` and `LANGCHAIN_API_KEY`.
2. **Don't trace sensitive data**: Traces include prompt content and LLM outputs. Don't trace PII, passwords, or secrets.
3. **Dataset names are unique per workspace**: Creating a dataset with an existing name returns the existing one (idempotent).
4. **Evaluation is async in LangSmith**: `client.evaluate()` runs experiments asynchronously. Check results in the UI, not immediately in code.
5. **Comparative eval has API costs**: Each evaluation example calls the LLM — budget accordingly.

---

# 练习 15：LangSmith 追踪与评估

## 你将学到

- **自动追踪** — 零代码更改，只需设置环境变量
- **@traceable 装饰器** — 为自定义函数添加精细的 Span 级追踪
- **运行名称、标签和元数据** — 通过 `RunnableConfig` 组织追踪
- **数据集创建** — 从示例输入/输出构建评估数据集
- **评估** — 根据预期答案对链的输出进行评分
- **对比评估** — 并排比较不同服务商（DeepSeek vs Qwen）

## 为什么 LangSmith 很重要

LLM 应用是**非确定性的**——相同的输入可能产生不同的输出。传统日志和调试效果不佳。LangSmith 是专为 LLM 应用构建的可观测性平台——就像 Datadog，但为链和智能体而设计。

## 追踪架构

只需设置环境变量——无需代码更改。每次 `.invoke()` 调用自动向 LangSmith 发送追踪数据。当未配置 LangSmith 时，`@traceable` 是**空操作**——无任何影响地透传。这意味着你可以装饰代码，在有或没有 LangSmith 的情况下都能正常工作。

## 评估

数据集是 **(input, expected_output)** 对的集合。你针对每个输入运行链，并将输出与预期输出进行比较。LangSmith 支持自定义评估器和内置评估器（正确性、有用性、毒性）。

## 常见陷阱

1. **没有配置时 @traceable 是空操作**：如果看不到追踪数据，检查 `LANGCHAIN_TRACING_V2=true` 和 `LANGCHAIN_API_KEY`。
2. **不要追踪敏感数据**：追踪包含提示词内容和 LLM 输出。不要追踪 PII、密码或密钥。
3. **数据集名称在工作区内唯一**：创建同名数据集返回已有数据集（幂等）。
4. **评估在 LangSmith 中是异步的**：`client.evaluate()` 异步运行实验。在 UI 中查看结果，而不是立即在代码中获取。
