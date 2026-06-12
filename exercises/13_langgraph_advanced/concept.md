# Exercise 13: LangGraph Advanced

## What You'll Learn

- **Subgraphs** — compose graphs as nodes in larger graphs
- **MemorySaver** — persistent checkpoints for conversation state
- **interrupt()** — pause execution for human approval
- **Command** — resume from interrupt with user input
- **Multi-agent handoff** — one agent delegates to another
- **Thread-level persistence** — multiple concurrent conversations

## Why Advanced LangGraph Matters

Basic graphs handle routing. Advanced graphs handle **orchestration**:

1. **Multi-agent systems**: Researcher → Writer → Editor pipeline
2. **Human-in-the-loop**: AI drafts, human approves, AI finalizes
3. **Persistence**: Conversations survive server restarts
4. **Handoff**: One specialized agent passes work to another

These are production patterns. They move from "a single LLM call" to "a coordinated system of LLM-powered components."

## Multi-Agent Pipeline

```
┌────────────┐     ┌────────────┐     ┌────────────┐
│ Researcher │────▶│   Writer   │────▶│   Editor   │────▶ [END]
│  (facts)   │     │  (prose)   │     │ (polish)   │
└────────────┘     └────────────┘     └────────────┘
```

Each agent is a node in the graph with a specialized system prompt:

```python
def researcher_node(state):
    # "You are a RESEARCH agent. Provide 3 key facts about {topic}."
    return {"messages": [AIMessage(content=f"[Researcher] {facts}")],
            "current_agent": "writer"}

def writer_node(state):
    # "You are a WRITER agent. Transform research into engaging prose."
    return {"messages": [AIMessage(content=f"[Writer] {paragraph}")],
            "current_agent": "editor"}

def editor_node(state):
    # "You are an EDITOR agent. Polish for clarity, grammar, and impact."
    return {"messages": [AIMessage(content=f"[Editor - Final] {polished}")],
            "task_complete": True}
```

The router inspects `current_agent` to determine the next step:

```python
def router(state):
    if state.get("task_complete"):
        return END
    return state.get("current_agent", "researcher")
```

## Human-in-the-Loop

Some actions should NOT be fully automated (sending emails, making purchases, publishing content). `interrupt()` pauses the graph and waits for human input:

```python
def human_approval(state):
    draft = state["messages"][-1].content

    # Pause! Return control to the caller with a question
    user_decision = interrupt({
        "question": "Approve this draft?",
        "draft": draft,
    })

    return {"approved": user_decision.get("approved", False)}
```

### Resuming from Interrupt

```python
# First call — runs until interrupt()
result = app.invoke(input, config=thread)

# Check if paused
snapshot = app.get_state(thread)
if snapshot.next:
    # Resume with human's decision
    result = app.invoke(
        Command(resume={"approved": True}),
        config=thread,
    )
```

The key insight: `app.invoke()` is called twice. First time runs until `interrupt()`. Second time resumes from where it paused with the human's input.

## MemorySaver — Persistence

```python
from langgraph.checkpoint.memory import MemorySaver

memory = MemorySaver()
app = graph.compile(checkpointer=memory)

thread = {"configurable": {"thread_id": "conversation-1"}}

# Each call with the same thread_id loads previous state
app.invoke(input_1, config=thread)  # State saved automatically
app.invoke(input_2, config=thread)  # Loads state from thread
```

`MemorySaver` checkpoints the graph state after each step. With the same `thread_id`, subsequent calls continue from where they left off.

## Key Concepts

### Thread-Level Isolation

```
thread_id = "alice-chat"   → Alice's conversation state
thread_id = "bob-chat"     → Bob's conversation state (completely separate)
```

Same graph, different threads — each with its own checkpoint history. In production, swap `MemorySaver` for a database-backed checkpointer (SqliteSaver, PostgresSaver).

### interrupt() vs Conditional Edges

| Mechanism | Purpose | Control |
|-----------|---------|---------|
| Conditional edges | Graph decides which path | Automatic |
| `interrupt()` | Human decides whether to proceed | Manual approval |

Use conditional edges for routing logic the AI can handle. Use `interrupt()` for decisions that need human judgment.

### Multi-Agent Handoff

The `current_agent` field in state acts as a handoff token — each node sets it to the next agent's name. The router reads it to determine the next node. This pattern scales to any number of specialized agents.

## Gotchas

1. **interrupt() only works with checkpointer**: The graph must be compiled with a `checkpointer` (e.g., `MemorySaver`) for interrupts to work. Without it, `interrupt()` raises an error.
2. **Command(resume=...) format**: The resume value must match what `interrupt()` expects. If `interrupt()` returns a dict, `Command(resume=...)` must pass the same shape.
3. **MemorySaver is NOT persistent**: It stores checkpoints in memory. Server restart = all state lost. Use `SqliteSaver` or `PostgresSaver` for production.
4. **Thread IDs must be unique per conversation**: Two conversations sharing a thread_id will mix their state.
5. **interrupt() can only be called once per node**: Multiple `interrupt()` calls in the same node are not supported.
