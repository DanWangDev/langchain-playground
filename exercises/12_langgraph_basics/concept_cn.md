
# 练习 12：LangGraph 基础

## 你将学到

- **StateGraph** — 定义带有流经各节点的类型化状态的图
- **TypedDict + Annotated** — 带 reducer 函数的图状态定义
- **add_node()** — 向图添加处理步骤
- **add_edge()** — 定义节点间的固定转换
- **add_conditional_edges()** — 基于状态的动态路由
- **.compile()** — 构建可执行的图

## 为什么 LangGraph 很重要

LCEL 链是线性的：A → B → C。LangGraph 处理**非线性**流程——可以分支、循环并根据中间结果做决策。这是智能体系统的基础。

## 核心概念

### 固定边 vs 条件边

| 类型 | 行为 | 示例 |
|------|------|------|
| `add_edge(a, b)` | 总是从 A 到 B | `add_edge("step2", END)` |
| `add_conditional_edges(a, router, mapping)` | 路由函数决定 | 复杂/简单路由 |

### 状态 Reducer

`Annotated[type, reducer]` 模式控制状态更新如何合并：

```python
# operator.add：追加到列表（消息累积）
messages: Annotated[list, operator.add]

# 无 reducer：替换值
complexity: str
```

### 图执行模式

| 方法 | 行为 |
|------|------|
| `.invoke(input)` | 运行一次，返回最终状态 |
| `.stream(input, stream_mode="updates")` | 每个节点完成时产出输出 |
| `.stream(input, stream_mode="values")` | 每个节点后产出完整状态 |

## 常见陷阱

1. **忘记列表上的 `operator.add`**：没有它，每个节点替换消息列表而非追加。之前的消息会丢失。
2. **节点返回 dict，不是 state**：返回 `{"complexity": "simple"}` 而非 `RouterState(complexity="simple")`。
3. **必须调用 .compile()**：图在调用 `.compile()` 前只是构建器。编译前调用会报错。
4. **每条路径都必须到达 END**：孤立节点或无限循环会导致图挂起（直到递归限制）。

