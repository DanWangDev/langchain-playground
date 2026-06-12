# Exercise 11: Callbacks

## What You'll Learn

- **BaseCallbackHandler** — sync callback interface for lifecycle hooks
- **Lifecycle hooks** — `on_llm_start`, `on_llm_end`, `on_chain_start`, `on_chain_end`
- **Tool monitoring** — `on_tool_start`, `on_tool_end`
- **Token counting** — track prompt + completion tokens across calls
- **Timing** — measure how long each step takes
- **Combined callbacks** — multiple handlers on the same run

## Why Callbacks Matter

Callbacks are the **observability primitive** of LangChain. They let you hook into every lifecycle event without modifying your chain code.

```
Your Chain:  prompt | llm | parser

Callbacks hook into:
  ┌─ on_chain_start
  │  ┌─ on_llm_start
  │  │  (LLM processing)
  │  └─ on_llm_end
  └─ on_chain_end
```

With callbacks, you can:
1. **Log** every step for debugging
2. **Time** operations to find bottlenecks
3. **Count tokens** to track costs
4. **Stream to UIs** — push updates to frontend
5. **Trigger alerts** on errors or slow responses

## How Callbacks Work

### Creating a Custom Handler

```python
from langchain_core.callbacks import BaseCallbackHandler

class TimingHandler(BaseCallbackHandler):
    def __init__(self):
        self.timings = {}
        self._start_times = {}

    def on_llm_start(self, serialized, prompts, **kwargs):
        run_id = str(kwargs.get("run_id", ""))[:8]
        self._start_times[run_id] = time.time()

    def on_llm_end(self, response, **kwargs):
        run_id = str(kwargs.get("run_id", ""))[:8]
        elapsed = time.time() - self._start_times.pop(run_id, 0)
        self.timings[run_id] = elapsed
```

### Attaching Callbacks to a Run

```python
timer = TimingHandler()
tracker = TokenTracker()

result = chain.invoke(
    {"topic": "quantum computing"},
    config={"callbacks": [timer, tracker]},
)
```

Multiple handlers can be attached simultaneously — each receives every event.

### Lifecycle Events

| Event | When It Fires | Key Data |
|-------|--------------|----------|
| `on_llm_start` | Before LLM call | `prompts` (list of messages) |
| `on_llm_end` | After LLM call | `response` (AIMessage with token_usage) |
| `on_chain_start` | Before chain step | `inputs` (what the step receives) |
| `on_chain_end` | After chain step | `outputs` (what the step produces) |
| `on_tool_start` | Before tool execution | Tool name + input |
| `on_tool_end` | After tool execution | Tool output |

### Token Tracking

```python
class TokenTracker(BaseCallbackHandler):
    def on_llm_end(self, response, **kwargs):
        usage = response.llm_output.get("token_usage", {})
        self.total_prompt_tokens += usage.get("prompt_tokens", 0)
        self.total_completion_tokens += usage.get("completion_tokens", 0)
```

Track cumulative usage across all LLM calls in a run. Multiply by model pricing to estimate cost.

## Key Concepts

### Callbacks vs astream_events()

| Feature | Callbacks | astream_events() |
|---------|-----------|-----------------|
| Scope | Per-invoke (config) | Per-invoke (async for) |
| Async | Separate `AsyncCallbackHandler` | Native async |
| Data access | Method parameters | Event dict |
| Best for | Logging, metrics, cost tracking | Real-time UI updates |

They serve different purposes. Callbacks are for **observability** (logging, metrics, cost). `astream_events()` is for **interactivity** (streaming to UIs).

### Handler Lifecycle

```
on_chain_start
  on_llm_start
  on_llm_end
  on_chain_start     ← nested chain
    on_llm_start
    on_llm_end
  on_chain_end
on_chain_end
```

Events nest — chains can contain sub-chains, and each fires its own start/end events. Use `run_id` to correlate events from the same run.

## Gotchas

1. **Don't mutate state in handlers**: Multiple handlers may run concurrently in async mode. Use thread-safe data structures or avoid shared mutable state.
2. **serialized can be None**: The `serialized` parameter in `on_llm_start` and `on_chain_start` may be `None` for some runnables. Always check before accessing.
3. **Callbacks add overhead**: Every event fires Python function calls. For high-throughput systems, keep handler logic minimal.
4. **Token usage may not be available**: Not all providers return token counts. Check `response.llm_output.get("token_usage")` for `None`.
5. **run_id is a UUID**: The full ID is long. Common practice is to truncate for display (first 8 chars).
