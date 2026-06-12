
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

