# Exercise 13: LangGraph Advanced / 练习 13：LangGraph 进阶

## What You'll Learn / 你将学到

- **Subgraphs** — compose graphs as nodes in larger graphs
- **MemorySaver** — persistent checkpoints for conversation state
- **interrupt()** — pause execution for human approval
- **Command** — resume from interrupt with user input
- **Multi-agent handoff** — one agent delegates to another
- **Thread-level persistence** — multiple concurrent conversations

## Why Advanced LangGraph Matters / 为什么进阶 LangGraph 很重要

Basic graphs handle routing. Advanced graphs handle **orchestration**:

1. **Multi-agent systems**: Researcher → Writer → Editor pipeline
2. **Human-in-the-loop**: AI drafts, human approves, AI finalizes
3. **Persistence**: Conversations survive server restarts
4. **Handoff**: One specialized agent passes work to another

These are production patterns. They move from "a single LLM call" to "a coordinated system of LLM-powered components."

## Multi-Agent Pipeline / 多智能体流水线

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

## Human-in-the-Loop / 人机协同

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

## MemorySaver — Persistence / 持久化

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

## Key Concepts / 核心概念

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

## Gotchas / 常见陷阱

1. **interrupt() only works with checkpointer**: The graph must be compiled with a `checkpointer` (e.g., `MemorySaver`) for interrupts to work. Without it, `interrupt()` raises an error.
2. **Command(resume=...) format**: The resume value must match what `interrupt()` expects. If `interrupt()` returns a dict, `Command(resume=...)` must pass the same shape.
3. **MemorySaver is NOT persistent**: It stores checkpoints in memory. Server restart = all state lost. Use `SqliteSaver` or `PostgresSaver` for production.
4. **Thread IDs must be unique per conversation**: Two conversations sharing a thread_id will mix their state.
5. **interrupt() can only be called once per node**: Multiple `interrupt()` calls in the same node are not supported.

---

# 练习 13：LangGraph 进阶

## 你将学到

- **子图** — 将图组合为更大图中的节点
- **MemorySaver** — 对话状态的持久化检查点
- **interrupt()** — 暂停执行等待人工审批
- **Command** — 从中断处恢复并传入用户输入
- **多智能体交接** — 一个智能体将工作委托给另一个
- **线程级持久化** — 多个并发的对话

## 多智能体流水线

```
研究员（事实）→ 写手（文章）→ 编辑（润色）→ [结束]
```

每个智能体是图中的一个节点，具有专门的系统提示词。`current_agent` 字段在状态中充当交接令牌——每个节点将其设为下一个智能体的名称。路由器读取它来决定下一步。

## 人机协同

某些操作不应完全自动化（发送邮件、购买、发布内容）。`interrupt()` 暂停图并等待人工输入。关键洞察：`app.invoke()` 被调用两次。第一次运行到 `interrupt()` 为止。第二次从暂停处恢复，带有人工输入。

## 核心概念

### interrupt() vs 条件边

| 机制 | 目的 | 控制方式 |
|------|------|----------|
| 条件边 | 图决定走哪条路径 | 自动 |
| `interrupt()` | 人决定是否继续 | 人工审批 |

## 常见陷阱

1. **interrupt() 只在有 checkpointer 时有效**：图必须用 `checkpointer` 编译（如 `MemorySaver`）才能使用中断。没有它，`interrupt()` 会报错。
2. **MemorySaver 不持久化**：它在内存中存储检查点。服务器重启 = 所有状态丢失。生产环境使用 `SqliteSaver` 或 `PostgresSaver`。
3. **线程 ID 必须每个对话唯一**：共享 thread_id 的两个对话会混合它们的状态。
