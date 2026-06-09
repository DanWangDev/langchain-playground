# LangChain Playground / LangChain 学习游乐场

Hands-on learning environment for the [LangChain](https://www.langchain.com/) framework, using **DeepSeek**, **Qwen**, and **AWS Bedrock** models.

基于 [LangChain](https://www.langchain.com/) 框架的动手学习环境，使用 **DeepSeek**、**Qwen** 和 **AWS Bedrock** 模型。

## Setup / 环境搭建

### Prerequisites / 前置条件

- Python 3.11+
- [uv](https://docs.astral.sh/uv/) package manager
- DeepSeek API key — [platform.deepseek.com](https://platform.deepseek.com/api_keys)
- Qwen / DashScope API key — [dashscope.console.aliyun.com](https://dashscope.console.aliyun.com/apiKey)
- (Optional / 可选) LangSmith API key — [smith.langchain.com](https://smith.langchain.com) (free tier available / 免费额度可用)
- (Optional / 可选) AWS credentials — for Bedrock exercises (`aws configure`)

### Install / 安装

```bash
# Clone the repo / 克隆仓库
git clone git@github.com:DanWangDev/langchain-playground.git
cd langchain-playground

# Copy and fill in your API keys / 复制并填入 API 密钥
cp .env.example .env
# Edit .env with your actual keys / 编辑 .env 填入实际密钥

# Install dependencies / 安装依赖
uv sync
```

### Verify / 验证

```bash
uv run python -c "from shared.llm import get_llm; llm = get_llm(); print(llm.invoke('Hello!'))"
```

### No API Keys? / 没有 API 密钥？

The playground works without real API keys — it falls back to `FakeLLM` which mimics real model responses so you can still run all exercises and see the chain structure work. CI passes green with zero secrets configured.

即使没有 API 密钥也能运行 — 会自动使用 `FakeLLM` 模拟真实模型响应，所有练习都能端到端运行。CI 无需任何密钥即可绿色通过。

## Curriculum / 课程

16 progressive exercises from zero to LangGraph + Bedrock mastery.
16 个渐进式练习，从零到 LangGraph + Bedrock 精通。

| # | Topic / 主题 | Key Concepts / 核心概念 |
|---|-------------|----------------------|
| 01 | Hello LLM | `ChatOpenAI`, `.invoke()`, DeepSeek vs Qwen |
| 02 | Prompt Templates / 提示模板 | `ChatPromptTemplate`, `MessagesPlaceholder`, few-shot |
| 03 | Chains & LCEL / 链与表达式 | Pipe operator, `RunnableParallel`, `RunnableLambda` |
| 04 | Output Parsers / 输出解析器 | `with_structured_output()`, `PydanticOutputParser` |
| 05 | RAG Basics / RAG 基础 | Splitters, `Chroma`, `create_retrieval_chain` |
| 06 | RAG Advanced / RAG 进阶 | `MultiQueryRetriever`, compression, hybrid BM25+vector |
| 07 | Memory / 记忆 | `RunnableWithMessageHistory`, sliding window |
| 08 | Tools / 工具 | `@tool` decorator, `bind_tools()`, tool-calling loop |
| 09 | Agents / 智能体 | `create_agent`, multi-step, system prompts |
| 10 | Streaming / 流式输出 | `.stream()`, `astream_events()`, parallel streaming |
| 11 | Callbacks / 回调 | `BaseCallbackHandler`, timing, token tracking |
| 12 | LangGraph Basics / LangGraph 基础 | `StateGraph`, conditional edges, routing |
| 13 | LangGraph Advanced / LangGraph 进阶 | Multi-agent pipeline, human-in-the-loop |
| 14 | Production Patterns / 生产模式 | Caching, fallbacks, retry, async parallel |
| 15 | LangSmith / 可观测性 | Tracing, `@traceable`, datasets, evaluation |
| 16 | AWS Bedrock | `ChatBedrock`, `BedrockEmbeddings`, provider comparison |

## Running Exercises / 运行练习

```bash
# Run a single exercise / 运行单个练习
uv run python exercises/01_hello_llm/main.py

# Run all exercises / 运行所有练习
for dir in exercises/*/; do
    uv run python "${dir}main.py"
done
```

## Project Structure / 项目结构

```
├── .github/workflows/    # CI — runs all exercises on PR / 自动运行所有练习
├── exercises/            # 16 progressive exercises / 16 个渐进式练习
│   ├── 01_hello_llm/
│   ├── 02_prompt_templates/
│   ├── ...
│   ├── 15_langsmith/
│   └── 16_bedrock/
├── shared/               # Shared utilities / 共享工具
│   ├── llm.py            # DeepSeek + Qwen LLM factory / 模型工厂
│   ├── embeddings.py     # Qwen embeddings factory / 嵌入工厂
│   └── bedrock.py        # AWS Bedrock factory / Bedrock 工厂
├── data/sample_docs/     # Sample documents for RAG / RAG 练习用示例文档
├── pyproject.toml
└── README.md
```

## Key Design Decisions / 关键设计决策

| Decision / 决策 | Why / 原因 |
|----------------|-----------|
| **FakeLLM fallback** | All exercises run without real API keys — CI stays green / 无需真实 API 密钥即可运行 |
| **OpenAI-compatible APIs** | DeepSeek + Qwen both speak OpenAI protocol → single `langchain-openai` dep / 两个提供商都用 OpenAI 兼容协议 |
| **langchain-classic** | v1.3 moved chains/retrievers to separate package / v1.3 将链和检索器移至独立包 |
| **create_agent** | LangGraph v1.0 deprecated `create_react_agent` — use `langchain.agents.create_agent` |
| **boto3 credential chain** | Bedrock exercises auto-detect AWS access, fall back to FakeLLM if none |
| **LangSmith no-op mode** | `@traceable` works as pass-through when tracing not configured |

## License / 许可证

MIT
