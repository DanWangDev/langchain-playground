
# 练习 01：初识大模型

## 你将学到

- 如何实例化指向不同服务商的 **ChatOpenAI** 客户端
- `.invoke()` 如何进行同步调用并返回 `AIMessage`
- **DeepSeek** 和 **Qwen** 作为大模型服务商的区别
- 如何读取**响应元数据**（模型名称、Token 使用量）

## 为什么模型选择很重要

不同的 LLM 服务商有不同的优势：

| 服务商 | 优势 | 最佳场景 |
|--------|------|----------|
| DeepSeek | 推理能力强，长上下文（128K），性价比高 | 复杂分析、代码生成 |
| Qwen（百炼） | 阿里云生态，中文优化，有竞争力的定价 | 中文任务、阿里云集成 |

两者都遵循 **OpenAI 兼容协议**，这意味着你可以用同一个 `langchain-openai` 的 `ChatOpenAI` 类——只需把 `base_url` 指向各自的端点即可。这是一个关键的设计优势：一个接口，多种后端。

## 工作原理

```
用户调用 .invoke("什么是 LangChain？")
  → ChatOpenAI 向服务商 API 发送 HTTP POST 请求
  → 服务商通过模型处理请求
  → 返回包含 .content 和 .response_metadata 的 AIMessage
```

### 关键方法

| 方法 | 返回值 | 适用场景 |
|--------|---------|----------|
| `.invoke(prompt)` | `AIMessage` | 单次同步调用 |
| `.ainvoke(prompt)` | `AIMessage`（可等待） | 单次异步调用 |
| `.stream(prompt)` | `AIMessage` 片段迭代器 | 实时逐字显示 |
| `.batch([p1, p2])` | `AIMessage` 列表 | 批量处理多个提示词 |

## 核心概念

### OpenAI 兼容协议

DeepSeek 和 Qwen 都实现了 OpenAI 对话补全 API 格式。这意味着：
- 相同的请求/响应结构
- 相同的认证方式（Bearer Token）
- 相同的 `langchain-openai` 集成
- 只需更改 `base_url` 和 `api_key` 即可切换服务商

## 常见陷阱

1. **API 密钥必须在实例化前设置**：`ChatOpenAI` 构造函数会读取密钥。如果为 `None`，调用将因认证错误而失败。
2. **服务商特定的模型名称**：DeepSeek 用 `deepseek-chat`，Qwen 用 `qwen-turbo`——它们在各服务商之间不可互换。
3. **Base URL 的路径后缀很重要**：DeepSeek 用 `/v1`，Qwen 百炼用 `/compatible-mode/v1`。错误的 URL 会返回 404 错误。
4. **CI 中的 FakeLLM 回退**：本游乐场自动检测缺失的 API 密钥并使用 `FakeLLM`。真正的 API 调用需要在 `.env` 中配置真实密钥。

