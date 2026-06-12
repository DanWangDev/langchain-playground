# Exercise 14: Production Patterns

## What You'll Learn

- **.with_fallbacks()** — degrade gracefully when the primary model fails
- **.with_retry()** — auto-retry on transient errors with exponential backoff
- **LLM Caching** — InMemoryCache, SQLiteCache to avoid redundant API calls
- **Error handling** — try/except, validation, and graceful degradation
- **Async parallel** — `asyncio.gather()` for concurrent chain execution
- **Rate limiting** — control request frequency

## Why Production Patterns Matter

The gap between a working prototype and a production system is wide:

| Prototype | Production |
|-----------|------------|
| "It works on my machine" | Works under load, 24/7 |
| No error handling | Graceful degradation |
| Every call hits the API | Caching saves cost + latency |
| Synchronous, sequential | Async, parallel, fast |
| No retry logic | Auto-retry with backoff |

These patterns close that gap.

## Caching

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

## Fallbacks

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

## Retry

```python
resilient_chain = chain.with_retry(
    stop_after_attempt=3,         # Try up to 3 times
    wait_exponential_multiplier=1, # Wait: 1s, 2s, 4s...
    wait_exponential_max=10,      # Max wait: 10s
)
```

Only retries on **transient** errors (network timeouts, rate limits). Does NOT retry on permanent errors (bad API key, invalid parameters).

## Async Parallel

```python
async def demo():
    things = ["Python", "JavaScript", "Rust", "Go", "Kotlin"]

    # All 5 chains run concurrently
    results = await asyncio.gather(
        *[chain.ainvoke({"thing": t}) for t in things]
    )
```

Serial execution: 5 × 2s = 10s. Parallel execution: ~2s (all run simultaneously).

## Key Concepts

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

## Gotchas

1. **Don't cache everything**: Only cache deterministic outputs. Don't cache `temperature=1.0` creative responses or `get_current_time()` tool calls.
2. **set_llm_cache is global**: It affects all LLM calls in the process. Use `set_llm_cache(None)` to disable when needed.
3. **Fallbacks hide errors**: A failing primary + succeeding fallback means you might not notice your primary is broken. Add logging/monitoring.
4. **Retry × Fallback interaction**: Be careful not to retry a failing primary 3 times BEFORE falling back — that's 3× the wait time. Consider retry ON the fallback instead.
5. **asyncio.gather fails fast**: If one chain raises, all other chains are cancelled. Use `return_exceptions=True` to collect errors gracefully.
