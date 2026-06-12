# Exercise 07: Memory / 练习 07：记忆

## What You'll Learn / 你将学到

- **ChatMessageHistory** — in-memory storage for conversation messages
- **RunnableWithMessageHistory** — wrap any chain with automatic history injection
- **get_session_history** — factory function for session-scoped history
- **Session management** — multiple concurrent conversations with isolated memory
- **Sliding window** — limit history to the last N messages to control context size

## Why Memory Matters / 为什么记忆很重要

LLMs are **stateless** by default. Each `.invoke()` is an independent call — the model has no memory of previous exchanges.

```
Without Memory:
  User: "My name is Alice."
  LLM: "Nice to meet you, Alice!"
  User: "What's my name?"
  LLM: "I don't know — you haven't told me." ❌

With Memory:
  User: "My name is Alice."
  LLM: "Nice to meet you, Alice!"
  User: "What's my name?"
  LLM: "Your name is Alice!" ✓
```

Memory makes conversations possible. Without it, every interaction starts from scratch.

## How Memory Works / 记忆的工作原理

```
┌──────────────────────────────────────────────────┐
│              RunnableWithMessageHistory           │
│                                                   │
│  1. Load history for session_id                   │
│  2. Inject into MessagesPlaceholder in prompt     │
│  3. Execute the chain                             │
│  4. Save user input + LLM output to history       │
│  5. Return result                                 │
└──────────────────────────────────────────────────┘
```

### The Architecture

```python
# 1. Define the prompt with a history slot
prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful assistant."),
    MessagesPlaceholder(variable_name="history"),  # ← History goes here
    ("human", "{input}"),
])

# 2. Build the base chain
chain = prompt | llm | StrOutputParser()

# 3. Wrap with history management
chain_with_history = RunnableWithMessageHistory(
    chain,
    get_session_history,          # Factory function
    input_messages_key="input",   # Which key holds user input
    history_messages_key="history", # Which key receives history
)

# 4. Invoke with a session_id
result = chain_with_history.invoke(
    {"input": "Hello!"},
    config={"configurable": {"session_id": "user-123"}},
)
```

### Session Store

```python
store: dict[str, InMemoryChatMessageHistory] = {}

def get_session_history(session_id: str) -> BaseChatMessageHistory:
    if session_id not in store:
        store[session_id] = InMemoryChatMessageHistory()
    return store[session_id]
```

Each session has its own `ChatMessageHistory` — conversations don't leak between users.

## Key Concepts / 核心概念

### Session Isolation

```
session_id = "alice"    →   History: [Alice's messages]
session_id = "bob"      →   History: [Bob's messages]
```

Same chain, different session IDs → completely isolated conversations. In production, replace the in-memory `store` dict with a database (Redis, Postgres).

### Sliding Window Memory

LLM context windows are finite. Long conversations eventually exceed the limit. The sliding window pattern keeps only the last N messages:

```python
MAX_HISTORY = 4  # Keep only last 4 messages (2 turns)
if len(history) > MAX_HISTORY:
    history = history[-MAX_HISTORY:]
```

The oldest messages are dropped. The LLM "forgets" early conversation to make room for recent context.

### Manual vs Automatic History

| Approach | Pros | Cons |
|----------|------|------|
| `RunnableWithMessageHistory` | Automatic, session-aware | Less control over what's stored |
| Manual history list | Full control, custom trimming | More code, no session isolation |

## Gotchas / 常见陷阱

1. **InMemoryChatMessageHistory is NOT persistent**: Restart the process, lose all conversations. For production, use a database-backed store.
2. **History grows unbounded without trimming**: Every turn adds 2 messages (user + assistant). At 1000 turns, that's 2000 messages — likely exceeding context limits.
3. **Session ID must be unique per conversation**: Two users sharing a session_id will see each other's history.
4. **MessagesPlaceholder must match the key name**: If `history_messages_key="history"` but your template uses `MessagesPlaceholder(variable_name="chat_history")`, history won't be injected.
5. **Config nesting**: Session ID goes in `config["configurable"]["session_id"]` — double nesting is easy to get wrong.

---

# 练习 07：记忆

## 你将学到

- **ChatMessageHistory** — 对话消息的内存存储
- **RunnableWithMessageHistory** — 用自动历史注入包装任何链
- **get_session_history** — 会话作用域历史的工厂函数
- **会话管理** — 多个并发的、记忆隔离的对话
- **滑动窗口** — 限制历史为最近 N 条消息以控制上下文大小

## 为什么记忆很重要

LLM 默认是**无状态**的。每次 `.invoke()` 都是独立调用——模型对之前的对话没有任何记忆。记忆使对话成为可能。没有它，每次交互都从零开始。

## 核心概念

### 会话隔离

```
session_id = "alice"    →   历史：[Alice 的消息]
session_id = "bob"      →   历史：[Bob 的消息]
```

同一条链，不同的会话 ID → 完全隔离的对话。在生产环境中，将内存中的 `store` 字典替换为数据库（Redis、Postgres）。

### 滑动窗口记忆

LLM 上下文窗口是有限的。长对话最终会超出限制。滑动窗口模式只保留最近 N 条消息——最旧的消息被丢弃，LLM"遗忘"早期对话以为最近的上下文腾出空间。

### 手动 vs 自动历史管理

| 方法 | 优点 | 缺点 |
|------|------|------|
| `RunnableWithMessageHistory` | 自动，会话感知 | 对存储内容的控制较少 |
| 手动历史列表 | 完全控制，自定义裁剪 | 代码更多，无会话隔离 |

## 常见陷阱

1. **InMemoryChatMessageHistory 不持久化**：重启进程就丢失所有对话。生产环境请使用数据库支持的存储。
2. **不裁剪的话历史无限增长**：每轮添加 2 条消息（用户 + 助手）。1000 轮后是 2000 条消息——很可能超出上下文限制。
3. **会话 ID 必须每个对话唯一**：两个用户共享 session_id 将看到对方的历史。
4. **双重嵌套的 Config**：会话 ID 在 `config["configurable"]["session_id"]` 中——双重嵌套很容易写错。
