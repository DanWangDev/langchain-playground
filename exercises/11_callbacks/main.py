"""
Exercise 11 — Callbacks
=======================
Hook into the lifecycle of every LangChain operation.

Concepts introduced:
- BaseCallbackHandler — sync callback interface
- AsyncCallbackHandler — async variant
- Lifecycle hooks: on_llm_start, on_llm_end, on_chain_start, on_chain_end
- on_tool_start, on_tool_end — monitor tool usage
- Token counting and cost tracking
- Custom logging via callbacks
- .with_config() — attach callbacks to specific runs
"""

import time
from langchain_core.callbacks import BaseCallbackHandler, AsyncCallbackHandler
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableLambda
from shared.llm import get_llm


# --- Custom Callback Handlers ---


class TimingHandler(BaseCallbackHandler):
    """Track how long each step takes."""

    def __init__(self):
        self.timings: dict[str, float] = {}
        self._start_times: dict[str, float] = {}

    def on_llm_start(self, serialized, prompts, **kwargs):
        run_id = str(kwargs.get("run_id", "unknown"))[:8]
        self._start_times[run_id] = time.time()
        print(f"  ⏱️  LLM call started [{run_id}]")

    def on_llm_end(self, response, **kwargs):
        run_id = str(kwargs.get("run_id", "unknown"))[:8]
        elapsed = time.time() - self._start_times.pop(run_id, 0)
        self.timings[run_id] = elapsed
        print(f"  ⏱️  LLM call ended [{run_id}] — {elapsed:.2f}s")


class TokenTracker(BaseCallbackHandler):
    """Track prompt and completion tokens."""

    def __init__(self):
        self.total_prompt_tokens = 0
        self.total_completion_tokens = 0

    def on_llm_end(self, response, **kwargs):
        usage = response.llm_output.get("token_usage", {})
        prompt_tokens = usage.get("prompt_tokens", 0)
        completion_tokens = usage.get("completion_tokens", 0)
        self.total_prompt_tokens += prompt_tokens
        self.total_completion_tokens += completion_tokens
        print(f"  📊 Tokens: {prompt_tokens} prompt + {completion_tokens} completion")


class ChainLogger(BaseCallbackHandler):
    """Log every chain and tool step."""

    def on_chain_start(self, serialized, inputs, **kwargs):
        name = serialized.get("name", "unnamed")
        print(f"  🔗 Chain start: {name}")

    def on_chain_end(self, outputs, **kwargs):
        print(f"  🔗 Chain end")

    def on_tool_start(self, serialized, input_str, **kwargs):
        print(f"  🔧 Tool start: {serialized.get('name', 'unknown')}")

    def on_tool_end(self, output, **kwargs):
        print(f"  🔧 Tool end: {str(output)[:100]}")


def demo_timing_handler():
    """Use the timing handler to measure LLM call duration."""
    print("\n" + "=" * 60)
    print("1. Timing Callbacks")
    print("=" * 60)

    llm = get_llm()
    timer = TimingHandler()

    prompt = ChatPromptTemplate.from_template(
        "Explain {topic} in 2-3 sentences."
    )
    chain = prompt | llm | StrOutputParser()

    result = chain.invoke(
        {"topic": "quantum computing"},
        config={"callbacks": [timer]},
    )
    print(f"\nResult: {result[:200]}...")
    print(f"Timings: {timer.timings}")


def demo_token_tracker():
    """Track token usage across multiple calls."""
    print("\n" + "=" * 60)
    print("2. Token Tracking Across Calls")
    print("=" * 60)

    llm = get_llm()
    tracker = TokenTracker()

    prompt = ChatPromptTemplate.from_template("Say: {phrase}")

    chain = prompt | llm | StrOutputParser()

    phrases = ["Hello world", "The quick brown fox"]

    for phrase in phrases:
        print(f"\nPhrase: '{phrase}'")
        chain.invoke({"phrase": phrase}, config={"callbacks": [tracker]})

    print(f"\nTotal prompt tokens: {tracker.total_prompt_tokens}")
    print(f"Total completion tokens: {tracker.total_completion_tokens}")


def demo_chain_logger():
    """Log every step in a complex chain."""
    print("\n" + "=" * 60)
    print("3. Chain Step Logging")
    print("=" * 60)

    llm = get_llm()
    logger = ChainLogger()

    # A multi-step chain
    prompt = ChatPromptTemplate.from_template(
        "Summarize: {text}"
    )

    chain = (
        RunnableLambda(lambda x: {"text": x["text"].upper()})
        | prompt
        | llm
        | StrOutputParser()
    )

    result = chain.invoke(
        {"text": "LangChain makes it easy to build LLM applications."},
        config={"callbacks": [logger]},
    )
    print(f"\nResult: {result}")


def demo_combined_callbacks():
    """Multiple callbacks on the same run."""
    print("\n" + "=" * 60)
    print("4. Combined Callbacks — Timer + Tracker + Logger")
    print("=" * 60)

    llm = get_llm()
    timer = TimingHandler()
    tracker = TokenTracker()
    logger = ChainLogger()

    prompt = ChatPromptTemplate.from_template(
        "In one sentence, what is {thing}?"
    )

    chain = prompt | llm | StrOutputParser()

    result = chain.invoke(
        {"thing": "machine learning"},
        config={"callbacks": [timer, tracker, logger]},
    )
    print(f"\nResult: {result}")
    print(f"Total time: {sum(timer.timings.values()):.2f}s")
    print(f"Tokens: {tracker.total_prompt_tokens}p + {tracker.total_completion_tokens}c")


def main():
    print("=" * 60)
    print("Exercise 11: Callbacks")
    print("=" * 60)

    demo_timing_handler()
    demo_token_tracker()
    demo_chain_logger()
    demo_combined_callbacks()


if __name__ == "__main__":
    main()
