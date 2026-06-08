"""
Shared LLM factory for DeepSeek and Qwen models.

Both providers offer OpenAI-compatible APIs, so we use langchain_openai's
ChatOpenAI pointed at each provider's base URL.

When API keys are not set (e.g. in CI), a FakeLLM is returned that
mimics the ChatOpenAI interface so exercises can run end-to-end without
real API calls.

Usage:
    from shared.llm import get_llm, get_deepseek, get_qwen

    llm = get_llm("deepseek")          # default
    llm = get_llm("qwen", temperature=0.7)
    deepseek = get_deepseek(model="deepseek-chat")
    qwen = get_qwen(model="qwen-max")
"""

import os
import re
from typing import Any, Iterator, AsyncIterator
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.messages import AIMessage, BaseMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from langchain_core.runnables import Runnable

load_dotenv()

DEEPSEEK_BASE_URL = "https://api.deepseek.com/v1"
DASHSCOPE_BASE_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"

_FAKE_LLM_WARNING = (
    "[FakeLLM] API key not set — using fake responses. "
    "Set DEEPSEEK_API_KEY / DASHSCOPE_API_KEY in .env for real API calls."
)


class FakeLLM(Runnable):
    """A drop-in replacement for ChatOpenAI when no API keys are available.

    Returns predictable canned responses so that chain structure,
    streaming, tool binding, and structured output all work without
    making real API calls. Prints a warning on first use so the
    developer knows they're in fake mode.
    """

    def __init__(self, model: str = "fake", temperature: float = 0):
        self.model = model
        self.temperature = temperature
        self._tools: list = []
        self._structured_output_model: type | None = None
        self._warned = False

    def _warn_once(self):
        if not self._warned:
            print(_FAKE_LLM_WARNING)
            self._warned = True

    # --- Core Runnable interface ---

    def invoke(self, input, config=None, **kwargs) -> Any:
        self._warn_once()

        # If .with_structured_output() was called, return a populated instance
        if self._structured_output_model:
            return self._fake_structured_output(input)

        # Extract text from input
        text = self._extract_text(input)

        # If we already have a tool result in the conversation, produce final answer
        if self._tools and ("ToolMessage" in text or "tool_call_id" in text.lower() or "Result:" in text):
            return AIMessage(
                content=f"[FakeLLM/{self.model}] Based on the tool results, here is the answer: {text[:50]}...",
                response_metadata={"model_name": f"fake-{self.model}", "token_usage": {"prompt_tokens": 10, "completion_tokens": 5}},
            )

        # If tools are bound and the input asks for tool-like things, simulate ONE tool call
        if self._tools and any(
            kw in text.lower() for kw in ("calculate", "time", "weather", "count", "reverse", "length", "word")
        ):
            return self._fake_tool_call(text)

        return AIMessage(
            content=f"[FakeLLM/{self.model}] Received: {text[:100]}...",
            response_metadata={"model_name": f"fake-{self.model}", "token_usage": {"prompt_tokens": 10, "completion_tokens": 5}},
        )

    def stream(self, input, config=None, **kwargs) -> Iterator[Any]:
        self._warn_once()
        msg = self.invoke(input, config, **kwargs)
        content = msg.content if hasattr(msg, "content") else str(msg)
        # Yield word by word to simulate streaming
        for word in content.split():
            yield AIMessage(content=word + " ")

    async def ainvoke(self, input, config=None, **kwargs) -> Any:
        return self.invoke(input, config, **kwargs)

    async def astream(self, input, config=None, **kwargs) -> AsyncIterator[Any]:
        for chunk in self.stream(input, config, **kwargs):
            yield chunk

    # --- Tool binding ---

    def bind_tools(self, tools: list, **kwargs) -> "FakeLLM":
        """Store tools and return self — chain structure preserved."""
        clone = FakeLLM(model=self.model, temperature=self.temperature)
        clone._tools = tools
        clone._structured_output_model = self._structured_output_model
        return clone

    # --- Structured output ---

    def with_structured_output(self, schema: type, **kwargs) -> "FakeLLM":
        """Store the schema and return self — structured output preserved."""
        clone = FakeLLM(model=self.model, temperature=self.temperature)
        clone._tools = self._tools
        clone._structured_output_model = schema
        return clone

    # --- Internal helpers ---

    def _extract_text(self, input) -> str:
        """Pull text out of various input formats."""
        if isinstance(input, str):
            return input
        if isinstance(input, list):
            texts = []
            for msg in input:
                if hasattr(msg, "content"):
                    texts.append(str(msg.content))
                elif isinstance(msg, dict):
                    texts.append(str(msg.get("content", "")))
                elif isinstance(msg, str):
                    texts.append(msg)
            return " ".join(texts)
        if isinstance(input, dict):
            return str(input.get("input", input.get("text", str(input))))
        return str(input)

    def _fake_tool_call(self, text: str) -> AIMessage:
        """Simulate a tool call response."""
        import json
        tool_name = self._tools[0].name if self._tools else "unknown"
        return AIMessage(
            content="",
            tool_calls=[{
                "id": "fake-call-001",
                "name": tool_name,
                "args": {"expression": "2+2"} if "calc" in text.lower() else {},
            }],
            response_metadata={"model_name": f"fake-{self.model}"},
        )

    def _fake_structured_output(self, input) -> Any:
        """Return a default instance of the structured output model."""
        self._warn_once()
        model = self._structured_output_model
        try:
            # Build a default instance from the model's fields
            fields = model.model_fields
            kwargs = {}
            for name, field in fields.items():
                annotation = field.annotation
                if annotation is str:
                    kwargs[name] = f"fake-{name}"
                elif annotation is int:
                    kwargs[name] = 0
                elif annotation is float:
                    kwargs[name] = 0.0
                elif annotation is bool:
                    kwargs[name] = False
                elif hasattr(annotation, "__origin__") and annotation.__origin__ is list:
                    kwargs[name] = ["fake-item"]
                else:
                    kwargs[name] = None
            return model(**kwargs)
        except Exception:
            return model()


# --- Public API ---

def _get_api_key(var_name: str) -> str | None:
    """Get an API key, returning None if not set (instead of raising)."""
    return os.environ.get(var_name)


def _has_api_keys() -> bool:
    """Check if at least one provider has an API key set."""
    return bool(os.environ.get("DEEPSEEK_API_KEY") or os.environ.get("DASHSCOPE_API_KEY"))


def get_deepseek(model: str = "deepseek-chat", temperature: float = 0) -> ChatOpenAI | FakeLLM:
    """Create a ChatOpenAI pointed at DeepSeek's API, or FakeLLM if no key."""
    api_key = os.environ.get("DEEPSEEK_API_KEY")
    if not api_key:
        return FakeLLM(model=model, temperature=temperature)
    return ChatOpenAI(
        model=model,
        temperature=temperature,
        base_url=DEEPSEEK_BASE_URL,
        api_key=api_key,
    )


def get_qwen(model: str = "qwen-plus", temperature: float = 0) -> ChatOpenAI | FakeLLM:
    """Create a ChatOpenAI pointed at Qwen's DashScope API, or FakeLLM if no key."""
    api_key = os.environ.get("DASHSCOPE_API_KEY")
    if not api_key:
        return FakeLLM(model=model, temperature=temperature)
    return ChatOpenAI(
        model=model,
        temperature=temperature,
        base_url=DASHSCOPE_BASE_URL,
        api_key=api_key,
    )


def get_llm(provider: str = "deepseek", temperature: float = 0) -> ChatOpenAI | FakeLLM:
    """Get an LLM by provider name. Falls back to FakeLLM if no API key.

    Args:
        provider: "deepseek" (default) or "qwen"
        temperature: 0-2, lower = more deterministic

    Returns:
        A ChatOpenAI instance (real) or FakeLLM (CI / no-key mode)
    """
    if provider == "qwen":
        return get_qwen(temperature=temperature)
    return get_deepseek(temperature=temperature)
