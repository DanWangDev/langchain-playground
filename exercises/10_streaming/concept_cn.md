
# 练习 10：流式输出

## 你将学到

- **`.stream()`** — 同步逐 Token 流式输出
- **`.astream()`** — 异步应用的异步流式输出
- **`.astream_events()`** — 详细事件级流式输出（v2 API）
- **事件类型** — `on_chat_model_stream`, `on_chat_model_start`, `on_chat_model_end`
- **并行流式** — 同时从多条链流式输出

## 为什么流式输出很重要

没有流式输出，用户在 LLM 生成完整响应之前什么也看不到。流式输出提供感知性能（用户立即看到进展）、可中断性和更好的用户体验。

## 流式方法

| 方法 | 粒度 | 适用场景 |
|------|------|----------|
| `.stream()` | Token 片段 | 简单聊天 UI |
| `.astream()` | Token 片段（异步） | 异步 Web 应用 |
| `.astream_events()` | 生命周期事件 | 带状态指示的复杂 UI |

## 常见陷阱

1. **astream_events 需要 version="v2"**：v1 API 已弃用。始终传递 `version="v2"`。
2. **流式输出不适用于所有解析器**：`with_structured_output()` 返回完整对象——无法流式输出部分 Pydantic 模型。
3. **并行流式可能交错输出**：两个同时到达的流可能混合并打断词语。设计 UI 以优雅处理部分输出。
4. **flush 很重要**：不设置 `flush=True`，输出可能缓冲到换行符才显示——违背了流式输出的目的。

