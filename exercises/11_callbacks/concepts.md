# Exercise 11: Callbacks / 练习 11：回调

## What You'll Learn / 你将学到

- **BaseCallbackHandler** — sync callback interface for lifecycle hooks
- **Lifecycle hooks** — `on_llm_start`, `on_llm_end`, `on_chain_start`, `on_chain_end`
- **Tool monitoring** — `on_tool_start`, `on_tool_end`
- **Token counting** — track prompt + completion tokens across calls
- **Timing** — measure how long each step takes
- **Combined callbacks** — multiple handlers on the same run

## Why Callbacks Matter / 为什么回调很重要

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

## How Callbacks Work / 回调的工作原理

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

## Key Concepts / 核心概念

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

## Gotchas / 常见陷阱

1. **Don't mutate state in handlers**: Multiple handlers may run concurrently in async mode. Use thread-safe data structures or avoid shared mutable state.
2. **serialized can be None**: The `serialized` parameter in `on_llm_start` and `on_chain_start` may be `None` for some runnables. Always check before accessing.
3. **Callbacks add overhead**: Every event fires Python function calls. For high-throughput systems, keep handler logic minimal.
4. **Token usage may not be available**: Not all providers return token counts. Check `response.llm_output.get("token_usage")` for `None`.
5. **run_id is a UUID**: The full ID is long. Common practice is to truncate for display (first 8 chars).

---

# 练习 11：回调

## 你将学到

- **BaseCallbackHandler** — 生命周期钩子的同步回调接口
- **生命周期钩子** — `on_llm_start`, `on_llm_end`, `on_chain_start`, `on_chain_end`
- **工具监控** — `on_tool_start`, `on_tool_end`
- **Token 计数** — 跨调用跟踪提示词和补全 Token
- **计时** — 测量每个步骤的耗时
- **组合回调** — 在同一次运行中使用多个处理器

## 为什么回调很重要

回调是 LangChain 的**可观测性原语**。它们让你能在不修改链代码的情况下钩入每个生命周期事件。通过回调你可以：记录调试日志、计时查找瓶颈、统计 Token 跟踪成本、向 UI 推送更新、在错误或响应慢时触发告警。

## 生命周期事件

| 事件 | 触发时机 | 关键数据 |
|------|----------|----------|
| `on_llm_start` | LLM 调用之前 | `prompts`（消息列表） |
| `on_llm_end` | LLM 调用之后 | `response`（含 token_usage 的 AIMessage） |
| `on_chain_start` | 链步骤之前 | `inputs` |
| `on_chain_end` | 链步骤之后 | `outputs` |
| `on_tool_start` | 工具执行之前 | 工具名称 + 输入 |
| `on_tool_end` | 工具执行之后 | 工具输出 |

## Callbacks vs astream_events()

| 特性 | Callbacks | astream_events() |
|------|-----------|-----------------|
| 范围 | 每次调用（config） | 每次调用（async for） |
| 异步 | 独立的 `AsyncCallbackHandler` | 原生 async |
| 数据访问 | 方法参数 | 事件字典 |
| 最适合 | 日志、指标、成本跟踪 | 实时 UI 更新 |

## 常见陷阱

1. **不要在处理器中修改状态**：多个处理器可能在异步模式下并发运行。使用线程安全的数据结构或避免共享可变状态。
2. **serialized 可能为 None**：某些 runnable 的 `on_llm_start` 和 `on_chain_start` 中的 `serialized` 参数可能为 `None`。访问前始终检查。
3. **Token 用量可能不可用**：并非所有服务商都返回 Token 计数。检查 `response.llm_output.get("token_usage")` 是否为 `None`。
