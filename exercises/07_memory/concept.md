# Exercise 07: Memory

## What You'll Learn

- **ChatMessageHistory** — in-memory storage for conversation messages
- **RunnableWithMessageHistory** — wrap any chain with automatic history injection
- **get_session_history** — factory function for session-scoped history
- **Session management** — multiple concurrent conversations with isolated memory
- **Sliding window** — limit history to the last N messages to control context size

## Why Memory Matters

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

## How Memory Works

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

## Key Concepts

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

## Gotchas

1. **InMemoryChatMessageHistory is NOT persistent**: Restart the process, lose all conversations. For production, use a database-backed store.
2. **History grows unbounded without trimming**: Every turn adds 2 messages (user + assistant). At 1000 turns, that's 2000 messages — likely exceeding context limits.
3. **Session ID must be unique per conversation**: Two users sharing a session_id will see each other's history.
4. **MessagesPlaceholder must match the key name**: If `history_messages_key="history"` but your template uses `MessagesPlaceholder(variable_name="chat_history")`, history won't be injected.
5. **Config nesting**: Session ID goes in `config["configurable"]["session_id"]` — double nesting is easy to get wrong.
