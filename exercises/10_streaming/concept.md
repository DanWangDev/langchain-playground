# Exercise 10: Streaming

## What You'll Learn

- **`.stream()`** — synchronous token-by-token streaming
- **`.astream()`** — async streaming for async applications
- **`.astream_events()`** — detailed event-level streaming (v2 API)
- **Event types** — `on_chat_model_stream`, `on_chat_model_start`, `on_chat_model_end`
- **Parallel streaming** — stream from multiple chains simultaneously
- **Token-level vs chunk-level** streaming granularity

## Why Streaming Matters

Without streaming, the user sees nothing until the LLM finishes generating the full response. For a 30-second generation, that's 30 seconds of blank screen.

```
Non-streaming:  [..........30 seconds..........] → Full response appears
Streaming:      H→e→l→l→o→ →W→o→r→l→d→! → Tokens appear as generated
```

Streaming provides:
1. **Perceived performance** — users see progress immediately (TTFB: Time To First Byte)
2. **Interruptibility** — users can stop generation if the output is wrong
3. **Better UX** — feels more conversational, like a human typing

## Streaming Methods

### .stream() — Token by Token

```python
for chunk in chain.stream({"topic": "programming"}):
    print(chunk, end="", flush=True)
```

Each `chunk` is a partial string (usually a few characters or a word). The `flush=True` ensures characters appear immediately.

### .astream() — Async Token Streaming

```python
async for chunk in chain.astream({"technology": "async programming"}):
    print(chunk, end="", flush=True)
```

Same as `.stream()` but non-blocking. Essential for web servers (FastAPI, Flask async) where you can't block the event loop.

### .astream_events() — Full Event Visibility

```python
async for event in chain.astream_events({"concept": "recursion"}, version="v2"):
    kind = event["event"]
    if kind == "on_chat_model_start":
        print("[LLM START]")
    elif kind == "on_chat_model_stream":
        print(event["data"]["chunk"].content, end="")
    elif kind == "on_chat_model_end":
        print("[LLM END]")
```

This is the most detailed API. It emits events for every lifecycle step:
- `on_chain_start` / `on_chain_end` — chain boundaries
- `on_chat_model_start` / `on_chat_model_stream` / `on_chat_model_end` — LLM lifecycle
- `on_tool_start` / `on_tool_end` — tool execution
- `on_retriever_start` / `on_retriever_end` — retrieval steps

Use this when building UIs that show **what the system is doing at each step** (not just what it's saying).

### Parallel Streaming

```python
parallel = RunnableParallel(
    haiku=haiku_chain,
    limerick=limerick_chain,
)

for chunk in parallel.stream({"topic": "coffee"}):
    for key, value in chunk.items():
        if value:
            print(f"[{key}] {value}", end="")
```

Both chains stream simultaneously. Each chunk dict contains partial results from whichever chain produced output. The keys let you route output to different UI elements.

## Key Concepts

### Streaming Granularity

| Method | Granularity | Use Case |
|--------|-------------|----------|
| `.stream()` | Token chunks | Simple chat UI |
| `.astream()` | Token chunks (async) | Async web apps |
| `.astream_events()` | Lifecycle events | Complex UI with status indicators |

### Event Filtering

`astream_events()` emits many events. Filter by `event["event"]` to focus on what you care about:

```python
# Only stream LLM tokens, ignore everything else
async for event in chain.astream_events(input, version="v2"):
    if event["event"] == "on_chat_model_stream":
        content = event["data"]["chunk"].content
        if content:
            yield content
```

### TTFB (Time To First Byte)

Streaming doesn't make generation faster — it makes it **feel** faster. The first token arrives in the same time regardless, but showing it immediately transforms the user experience.

## Gotchas

1. **astream_events requires version="v2"**: The v1 API is deprecated. Always pass `version="v2"`.
2. **Streaming doesn't work with all parsers**: `with_structured_output()` returns the complete object — it can't stream partial Pydantic models. Use `StrOutputParser` for streaming.
3. **Parallel streaming can interleave awkwardly**: Two streams arriving simultaneously may mix mid-word. Design your UI to handle partial outputs gracefully.
4. **Don't await inside sync .stream()**: `.stream()` is synchronous. Use `async for` only with `.astream()`.
5. **Flush is important**: Without `flush=True`, output may buffer until a newline — defeating the purpose of streaming.
