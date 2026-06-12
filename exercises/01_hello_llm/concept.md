# Exercise 01: Hello LLM

## What You'll Learn

- How to instantiate a **ChatOpenAI** client pointed at provider-specific base URLs
- How `.invoke()` makes a synchronous call and returns an `AIMessage`
- The difference between **DeepSeek** and **Qwen** as LLM providers
- How to read **response metadata** (model name, token usage)

## Why Provider Choice Matters

Different LLM providers have different strengths:

| Provider | Strengths | Best For |
|----------|-----------|----------|
| DeepSeek | Strong reasoning, long context (128K), cost-effective | Complex analysis, code generation |
| Qwen (DashScope) | Alibaba ecosystem, Chinese-optimized, competitive pricing | Chinese-language tasks, Alibaba Cloud integration |

Both speak the **OpenAI-compatible protocol**, which means you can use `langchain-openai`'s `ChatOpenAI` class for both — just point the `base_url` at each provider's endpoint. This is a key design advantage: one interface, many backends.

## How It Works

```
User call .invoke("What is LangChain?")
  → ChatOpenAI sends HTTP POST to provider's API
  → Provider processes request through the model
  → AIMessage returned with .content and .response_metadata
```

### The ChatOpenAI Constructor

```python
from langchain_openai import ChatOpenAI

llm = ChatOpenAI(
    model="deepseek-chat",          # Provider's model name
    api_key="sk-...",               # From environment variable
    base_url="https://api.deepseek.com/v1",  # Provider endpoint
)
```

### Key Methods

| Method | Returns | Use Case |
|--------|---------|----------|
| `.invoke(prompt)` | `AIMessage` | Single synchronous call |
| `.ainvoke(prompt)` | `AIMessage` (awaitable) | Single async call |
| `.stream(prompt)` | Iterator of `AIMessage` chunks | Real-time token display |
| `.batch([p1, p2])` | List of `AIMessage` | Multiple prompts at once |

### AIMessage Structure

```python
response = llm.invoke("Hello")
response.content         # "Hello! How can I help you?"
response.response_metadata  # {"model_name": "deepseek-chat", "finish_reason": "stop", ...}
```

## Key Concepts

### OpenAI-Compatible Protocol

Both DeepSeek and Qwen implement the OpenAI chat completions API format. This means:
- Same request/response shapes
- Same authentication pattern (Bearer token)
- Same `langchain-openai` integration
- You can swap providers by changing `base_url` and `api_key` only

### Synchronous vs Asynchronous

- **`.invoke()`**: Blocks until complete. Good for scripts and simple chains.
- **`.ainvoke()`**: Non-blocking. Essential for web servers and parallel execution.
- **`.stream()`**: Yields tokens as they arrive. Best for chat UIs.

## Gotchas

1. **API key must be set before instantiation**: The `ChatOpenAI` constructor reads the key. If it's `None`, calls will fail with an authentication error.
2. **Provider-specific model names**: `deepseek-chat` for DeepSeek, `qwen-turbo` for Qwen — they are NOT interchangeable between providers.
3. **Base URL trailing path matters**: DeepSeek uses `/v1`, Qwen DashScope uses `/compatible-mode/v1`. An incorrect URL gives 404 errors.
4. **FakeLLM fallback in CI**: This playground auto-detects missing API keys and uses `FakeLLM`. Real API calls need real keys in `.env`.
