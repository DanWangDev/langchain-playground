
# 练习 03：链与 LCEL 表达式

## 你将学到

- **LCEL 管道操作符**（`|`）— 将组件组合成流水线
- **StrOutputParser** — 从 AIMessage 提取纯文本
- **RunnableParallel** — 并发运行多条链
- **RunnableLambda** — 将任何 Python 函数包装为可链接的步骤
- **RunnablePassthrough** — 原样传递输入
- `.assign()` — 用计算字段丰富输入字典
- `.bind()` — 预设运行参数（如 temperature）

## 为什么 LCEL 很重要

LangChain 表达式语言（LCEL）是 LangChain 的**通用组合语言**。所有 "Runnable" 的东西（LLM、提示词、解析器、自定义函数）都可以用 `|` 连接起来。这是声明式的、可读的，并且自动处理流式输出、异步、并行和批处理。

## Runnable 接口

每个 LCEL 组件都实现了 `Runnable` 协议：

| 方法 | 说明 |
|--------|------|
| `.invoke(input)` | 单次同步调用 |
| `.ainvoke(input)` | 单次异步调用 |
| `.stream(input)` | 同步流式输出 |
| `.astream(input)` | 异步流式输出 |
| `.batch(inputs)` | 并行批处理 |

## 数据流

```
dict → PromptTemplate → [SystemMessage, HumanMessage] → ChatOpenAI → AIMessage → StrOutputParser → str
```

每一步都转换数据类型。理解每个阶段的数据类型有助于调试链。

## .bind() vs .assign()

| 方法 | 作用 |
|--------|------|
| `.bind(temperature=0.0)` | 为特定实例**预设 LLM 参数** |
| `.assign(field=func)` | 在传递过程中为输入字典**添加计算字段** |

`.bind()` 配置组件**如何**运行。`.assign()` 丰富**什么**数据在流动。

## 常见陷阱

1. **输入字典的键必须匹配模板变量**：如果提示词有 `{topic}` 但你传入 `{"subject": "AI"}`，将得到 `KeyError`。
2. **RunnableParallel 保留所有键**：每个分支的输出成为结果字典中的一个键。键名必须唯一。
3. **RunnableLambda 必须返回下一步的有效输入**：如果下一个组件期望一个 dict，你的 lambda 必须返回一个 dict。
4. **.assign() 中的顺序很重要**：先声明的赋值先运行。后面的赋值可以引用前面的结果。

