
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

