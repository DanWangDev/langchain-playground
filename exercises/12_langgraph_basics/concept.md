# Exercise 12: LangGraph Basics

## What You'll Learn

- **StateGraph** — define a graph with typed state that flows through nodes
- **TypedDict + Annotated** — graph state definition with reducer functions
- **add_node()** — add processing steps to the graph
- **add_edge()** — define fixed transitions between nodes
- **add_conditional_edges()** — dynamic routing based on state
- **.compile()** — build the executable graph
- **.invoke() / .stream()** — execute the graph
- **Command** — explicit next-node control

## Why LangGraph Matters

LCEL chains are linear: A → B → C. LangGraph handles **non-linear** flows:

```
        ┌──────────┐
        │ classify │
        └────┬─────┘
             │
        ┌────▼────┐
        │  route  │
        └───┬──┬──┘
            │  │
      ┌─────▼┐ ┌▼─────┐
      │simple│ │complex│
      └──┬───┘ └──┬───┘
         │        │
         └───┬────┘
             ▼
           [END]
```

This is the foundation of **agentic systems** — programs that can branch, loop, and make decisions based on intermediate results.

## How LangGraph Works

### 1. Define State

```python
from typing import TypedDict, Annotated
import operator

class RouterState(TypedDict):
    messages: Annotated[list, operator.add]  # Appended, not replaced!
    complexity: str
    response: str
```

`Annotated[list, operator.add]` is the key pattern: instead of replacing the list, new messages are **appended** to it. This is how state accumulates across nodes.

### 2. Define Nodes

```python
def classify_node(state: RouterState) -> dict:
    """Read state, call LLM, return partial update."""
    llm = get_llm()
    last_msg = state["messages"][-1].content
    result = chain.invoke({"query": last_msg})
    return {"complexity": result.strip()}  # Only return what changed
```

Nodes are pure functions: `(state) → partial_state_update`. They receive the full state, return only the fields they want to update.

### 3. Build the Graph

```python
graph = StateGraph(RouterState)

graph.add_node("classify", classify_node)
graph.add_node("simple_response", simple_response_node)
graph.add_node("complex_response", complex_response_node)

graph.set_entry_point("classify")

graph.add_conditional_edges(
    "classify",
    route_by_complexity,  # (state) → "simple_response" | "complex_response"
    {
        "simple_response": "simple_response",
        "complex_response": "complex_response",
    },
)

graph.add_edge("simple_response", END)
graph.add_edge("complex_response", END)

app = graph.compile()
```

### 4. Execute

```python
result = app.invoke({"messages": [HumanMessage(content="Explain REST vs GraphQL")]})

# Or stream node-by-node:
for step in app.stream(input, stream_mode="updates"):
    for node_name, update in step.items():
        print(f"Node '{node_name}' completed")
```

## Key Concepts

### Edges vs Conditional Edges

| Type | Behavior | Example |
|------|----------|---------|
| `add_edge(a, b)` | Always go from A to B | `add_edge("step2", END)` |
| `add_conditional_edges(a, router, mapping)` | Router function decides | Complex/simple routing |

### State Reducers

The `Annotated[type, reducer]` pattern controls how state updates merge:

```python
# operator.add: append to list (messages accumulate)
messages: Annotated[list, operator.add]

# No reducer: replace the value
complexity: str
response: str
```

Custom reducers can do anything: merge dicts, keep max value, concatenate strings.

### Graph Execution Modes

| Method | Behavior |
|--------|----------|
| `.invoke(input)` | Run once, return final state |
| `.stream(input, stream_mode="updates")` | Yield each node's output as it completes |
| `.stream(input, stream_mode="values")` | Yield full state after each node |

## Gotchas

1. **Forgetting `operator.add` on lists**: Without it, each node replaces the messages list instead of appending to it. Previous messages are lost.
2. **Node returns dict, not state**: Return `{"complexity": "simple"}` not `RouterState(complexity="simple")`.
3. **Must call .compile()**: The graph is a builder until `.compile()` is called. Invoking before compile raises an error.
4. **Conditional edge must route to defined nodes**: If the router returns a name not in the mapping or not a defined node, the graph crashes.
5. **Every path must reach END**: Orphaned nodes or infinite loops cause the graph to hang (until recursion limit).
6. **State keys must exist**: All nodes in the graph share the same state type. Don't access keys that another node was supposed to set but hasn't yet.
