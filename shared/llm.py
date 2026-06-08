"""
Shared LLM factory for DeepSeek and Qwen models.

Both providers offer OpenAI-compatible APIs, so we use langchain_openai's
ChatOpenAI pointed at each provider's base URL.

Usage:
    from shared.llm import get_llm, get_deepseek, get_qwen

    llm = get_llm("deepseek")          # default
    llm = get_llm("qwen", temperature=0.7)
    deepseek = get_deepseek(model="deepseek-chat")
    qwen = get_qwen(model="qwen-max")
"""

import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI

load_dotenv()

DEEPSEEK_BASE_URL = "https://api.deepseek.com/v1"
DASHSCOPE_BASE_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"


def get_deepseek(model: str = "deepseek-chat", temperature: float = 0) -> ChatOpenAI:
    """Create a ChatOpenAI instance pointed at DeepSeek's API."""
    api_key = os.environ.get("DEEPSEEK_API_KEY")
    if not api_key:
        raise ValueError(
            "DEEPSEEK_API_KEY not found. Set it in .env or your environment."
        )
    return ChatOpenAI(
        model=model,
        temperature=temperature,
        base_url=DEEPSEEK_BASE_URL,
        api_key=api_key,
    )


def get_qwen(model: str = "qwen-plus", temperature: float = 0) -> ChatOpenAI:
    """Create a ChatOpenAI instance pointed at Qwen's DashScope API."""
    api_key = os.environ.get("DASHSCOPE_API_KEY")
    if not api_key:
        raise ValueError(
            "DASHSCOPE_API_KEY not found. Set it in .env or your environment."
        )
    return ChatOpenAI(
        model=model,
        temperature=temperature,
        base_url=DASHSCOPE_BASE_URL,
        api_key=api_key,
    )


def get_llm(provider: str = "deepseek", temperature: float = 0) -> ChatOpenAI:
    """Get an LLM by provider name.

    Args:
        provider: "deepseek" (default) or "qwen"
        temperature: 0-2, lower = more deterministic

    Returns:
        A configured ChatOpenAI instance
    """
    if provider == "qwen":
        return get_qwen(temperature=temperature)
    return get_deepseek(temperature=temperature)
