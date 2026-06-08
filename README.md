# LangChain Playground

Hands-on learning environment for the [LangChain](https://www.langchain.com/) framework, using **DeepSeek** and **Qwen** models via OpenAI-compatible APIs.

## Setup

### Prerequisites

- Python 3.11+
- [uv](https://docs.astral.sh/uv/) package manager
- DeepSeek API key — [platform.deepseek.com](https://platform.deepseek.com/api_keys)
- Qwen/DashScope API key — [dashscope.console.aliyun.com](https://dashscope.console.aliyun.com/apiKey)

### Install

```bash
# Clone the repo
git clone git@github.com:DanWangDev/langchain-playground.git
cd langchain-playground

# Copy and fill in your API keys
cp .env.example .env
# Edit .env with your actual keys

# Install dependencies
uv sync
```

### Verify

```bash
uv run python -c "from shared.llm import get_llm; llm = get_llm(); print(llm.invoke('Hello!'))"
```

## Curriculum

14 progressive exercises from zero to LangGraph mastery.

| # | Topic | Key Concepts |
|---|-------|-------------|
| 01 | Hello LLM | `ChatOpenAI`, `.invoke()`, comparing providers |
| 02 | Prompt Templates | `ChatPromptTemplate`, `MessagesPlaceholder`, few-shot |
| 03 | Chains & LCEL | Pipe operator, `RunnableParallel`, `RunnableLambda` |
| 04 | Output Parsers | `with_structured_output()`, `PydanticOutputParser` |
| 05 | RAG Basics | Splitters, `Chroma`, `create_retrieval_chain` |
| 06 | RAG Advanced | `MultiQueryRetriever`, compression, hybrid search |
| 07 | Memory | `RunnableWithMessageHistory`, session management |
| 08 | Tools | `@tool` decorator, `bind_tools()` |
| 09 | Agents | `create_react_agent`, tool-calling agents |
| 10 | Streaming | `astream_events()`, token streaming |
| 11 | Callbacks | `BaseCallbackHandler`, token counting |
| 12 | LangGraph Basics | `StateGraph`, conditional edges, `Command` |
| 13 | LangGraph Advanced | Subgraphs, human-in-the-loop, persistence |
| 14 | Production Patterns | Fallbacks, caching, retry, FastAPI integration |

## Project Structure

```
├── .github/workflows/    # CI — runs all exercises on PR
├── exercises/            # 14 progressive exercises
│   ├── 01_hello_llm/
│   ├── 02_prompt_templates/
│   └── ...
├── shared/               # Shared utilities
│   └── llm.py            # DeepSeek + Qwen LLM factory
├── data/sample_docs/     # Sample documents for RAG exercises
├── pyproject.toml
└── README.md
```

## Workflow

- **Feature branches** — one branch per exercise group
- **PR to master** — code review + CI passes before merge
- **CI** — GitHub Actions runs every exercise on PR

## License

MIT
