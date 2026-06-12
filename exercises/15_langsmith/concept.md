# Exercise 15: LangSmith

## What You'll Learn

- **Automatic tracing** — zero code changes, just set environment variables
- **@traceable decorator** — granular span-level tracing for custom functions
- **Run names, tags, and metadata** — organize traces via `RunnableConfig`
- **Dataset creation** — build evaluation datasets from example inputs/outputs
- **Evaluation** — score chain outputs against expected answers
- **Comparative evaluation** — compare providers (DeepSeek vs Qwen) side by side

## Why LangSmith Matters

LLM applications are **non-deterministic** — the same input can produce different outputs. Traditional logging and debugging don't work well. You need to see:

1. **Every step** — prompt → LLM → tool → LLM → output
2. **Timing** — where is time spent? Which step is slow?
3. **Token usage** — how much does each call cost?
4. **Errors** — exactly which step failed and why?
5. **Outputs over time** — is quality improving or degrading?

LangSmith is the observability platform built specifically for LLM applications. It's like Datadog, but designed for chains and agents.

## LangSmith Architecture

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

## How Tracing Works

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

## Evaluation

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

## Key Concepts

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

## Gotchas

1. **@traceable is a no-op without config**: If you're not seeing traces, check `LANGCHAIN_TRACING_V2=true` and `LANGCHAIN_API_KEY`.
2. **Don't trace sensitive data**: Traces include prompt content and LLM outputs. Don't trace PII, passwords, or secrets.
3. **Dataset names are unique per workspace**: Creating a dataset with an existing name returns the existing one (idempotent).
4. **Evaluation is async in LangSmith**: `client.evaluate()` runs experiments asynchronously. Check results in the UI, not immediately in code.
5. **Comparative eval has API costs**: Each evaluation example calls the LLM — budget accordingly.
