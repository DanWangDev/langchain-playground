# Exercise 03: Chains & LCEL

## What You'll Learn

- **LCEL pipe operator** (`|`) — compose components into pipelines
- **StrOutputParser** — extract plain string from AIMessage
- **RunnableParallel** — run multiple chains concurrently
- **RunnableLambda** — wrap any Python function as a chain-compatible step
- **RunnablePassthrough** — pass input through unchanged
- `.assign()` — enrich input dicts with computed fields
- `.bind()` — pre-set parameters on a runnable (e.g., temperature)

## Why LCEL Matters

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

## How Chains Work

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

## Key Concepts

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

## Gotchas

1. **Input dict keys must match template variables**: If your prompt has `{topic}` but you pass `{"subject": "AI"}`, you'll get a `KeyError`.
2. **RunnableParallel preserves all keys**: Every branch's output becomes a key in the result dict. Key names must be unique.
3. **RunnableLambda must return valid input for next step**: If the next component expects a dict, your lambda must return a dict.
4. **Order matters in .assign()**: Earlier assignments run first. An assignment can reference values from earlier ones in the same chain.
