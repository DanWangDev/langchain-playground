# Exercise 14: Production Patterns / 练习 14：生产模式

## What You'll Learn / 你将学到

- **.with_fallbacks()** — degrade gracefully when the primary model fails
- **.with_retry()** — auto-retry on transient errors with exponential backoff
- **LLM Caching** — InMemoryCache, SQLiteCache to avoid redundant API calls
- **Error handling** — try/except, validation, and graceful degradation
- **Async parallel** — `asyncio.gather()` for concurrent chain execution
- **Rate limiting** — control request frequency

## Why Production Patterns Matter / 为什么生产模式很重要

The gap between a working prototype and a production system is wide:

| Prototype | Production |
|-----------|------------|
| "It works on my machine" | Works under load, 24/7 |
| No error handling | Graceful degradation |
| Every call hits the API | Caching saves cost + latency |
| Synchronous, sequential | Async, parallel, fast |
| No retry logic | Auto-retry with backoff |

These patterns close that gap.

## Caching / 缓存

```
Without cache:
  "What's the capital of France?" → API call → "Paris" (2.1s)
  "What's the capital of France?" → API call → "Paris" (2.0s) ← Redundant!
  "What's the capital of France?" → API call → "Paris" (2.1s) ← Redundant!

With InMemoryCache:
  "What's the capital of France?" → API call → "Paris" (2.1s) → cached
  "What's the capital of France?" → cache hit → "Paris" (0.001s) ✓
  "What's the capital of France?" → cache hit → "Paris" (0.001s) ✓
```

### Enabling Cache

```python
from langchain_core.caches import InMemoryCache
from langchain_core.globals import set_llm_cache

set_llm_cache(InMemoryCache())  # Fast, but lost on restart

# For persistence:
# from langchain_community.caches import SQLiteCache
# set_llm_cache(SQLiteCache(database_path=".langchain.db"))
```

Cache keys are based on: prompt text + model name + parameters (temperature, etc.). Identical prompt + identical settings = cache hit.

## Fallbacks / 回退

```python
primary_chain = prompt | deepseek_llm | StrOutputParser()
fallback_chain = prompt | qwen_llm | StrOutputParser()

robust_chain = primary_chain.with_fallbacks([fallback_chain])
```

If `primary_chain` fails (timeout, rate limit, API error), `robust_chain` automatically tries `fallback_chain`. Multiple fallbacks are tried in order.

```
Primary (DeepSeek) → FAILED
  → Fallback 1 (Qwen) → SUCCESS → Return result
```

### Fallback Chain

```python
robust_chain = chain.with_fallbacks([
    fallback_1,
    fallback_2,
    fallback_3,  # Last resort
])
```

## Retry / 重试

```python
resilient_chain = chain.with_retry(
    stop_after_attempt=3,         # Try up to 3 times
    wait_exponential_multiplier=1, # Wait: 1s, 2s, 4s...
    wait_exponential_max=10,      # Max wait: 10s
)
```

Only retries on **transient** errors (network timeouts, rate limits). Does NOT retry on permanent errors (bad API key, invalid parameters).

## Async Parallel / 异步并行

```python
async def demo():
    things = ["Python", "JavaScript", "Rust", "Go", "Kotlin"]

    # All 5 chains run concurrently
    results = await asyncio.gather(
        *[chain.ainvoke({"thing": t}) for t in things]
    )
```

Serial execution: 5 × 2s = 10s. Parallel execution: ~2s (all run simultaneously).

## Key Concepts / 核心概念

### Defense in Depth

Combine patterns for maximum resilience:

```python
robust_chain = (
    prompt
    | llm
    | StrOutputParser()
).with_fallbacks([fallback_chain])\
 .with_retry(stop_after_attempt=3)
```

Order matters: retry on the primary first, then fallback.

### Error Handling Patterns

| Pattern | When to Use | Example |
|---------|-------------|---------|
| try/except | Expected failure modes | Invalid input, API down |
| .with_fallbacks() | Provider-level failure | DeepSeek down → Qwen |
| .with_retry() | Transient errors | Rate limit, timeout |
| Validation | Prevent bad input | Required fields check |
| Graceful degradation | Non-critical failures | Return cached/default |

### Cache Strategy

| Cache Type | Persistence | Speed | Use Case |
|------------|-------------|-------|----------|
| InMemoryCache | Process only | Fastest | Development, single server |
| SQLiteCache | Disk | Fast | Single server production |
| RedisCache | External | Fast | Multi-server, shared cache |

## Gotchas / 常见陷阱

1. **Don't cache everything**: Only cache deterministic outputs. Don't cache `temperature=1.0` creative responses or `get_current_time()` tool calls.
2. **set_llm_cache is global**: It affects all LLM calls in the process. Use `set_llm_cache(None)` to disable when needed.
3. **Fallbacks hide errors**: A failing primary + succeeding fallback means you might not notice your primary is broken. Add logging/monitoring.
4. **Retry × Fallback interaction**: Be careful not to retry a failing primary 3 times BEFORE falling back — that's 3× the wait time. Consider retry ON the fallback instead.
5. **asyncio.gather fails fast**: If one chain raises, all other chains are cancelled. Use `return_exceptions=True` to collect errors gracefully.

---

# 练习 14：生产模式

## 你将学到

- **.with_fallbacks()** — 主模型失败时优雅回退
- **.with_retry()** — 指数退避的自动重试
- **LLM 缓存** — InMemoryCache、SQLiteCache 避免冗余 API 调用
- **错误处理** — try/except、验证和优雅降级
- **异步并行** — `asyncio.gather()` 并发执行链

## 为什么生产模式很重要

| 原型 | 生产 |
|------|------|
| "在我机器上能跑" | 24/7 负载下运行 |
| 无错误处理 | 优雅降级 |
| 每次调用都访问 API | 缓存节省成本 + 延迟 |
| 同步、顺序执行 | 异步、并行、快速 |
| 无重试逻辑 | 带退避的自动重试 |

## 纵深防御

组合多种模式以获得最大的韧性：

```python
robust_chain = (
    prompt | llm | StrOutputParser()
).with_fallbacks([fallback_chain]).with_retry(stop_after_attempt=3)
```

## 缓存策略

| 缓存类型 | 持久性 | 速度 | 适用场景 |
|----------|--------|------|----------|
| InMemoryCache | 仅进程内 | 最快 | 开发、单服务器 |
| SQLiteCache | 磁盘 | 快 | 单服务器生产 |
| RedisCache | 外部 | 快 | 多服务器共享 |

## 常见陷阱

1. **不要缓存所有东西**：只缓存确定性输出。不要缓存 `temperature=1.0` 的创意回答或 `get_current_time()` 工具调用。
2. **set_llm_cache 是全局的**：它影响进程内所有 LLM 调用。需要时用 `set_llm_cache(None)` 禁用。
3. **回退隐藏了错误**：主模型失败 + 回退成功意味着你可能没注意到主模型坏了。添加日志/监控。
4. **asyncio.gather 快速失败**：如果一条链抛出异常，所有其他链都被取消。使用 `return_exceptions=True` 优雅地收集错误。
