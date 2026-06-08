"""
Exercise 10 — Streaming
=======================
Stream tokens and events for real-time UIs and responsive applications.

Concepts introduced:
- .stream() — synchronous token-by-token streaming
- .astream() — async streaming
- .astream_events() — detailed event-level streaming (v2 API)
- Event types: on_chat_model_stream, on_chain_start, on_chain_end
- Token-level vs chunk-level streaming
- Async generator patterns
"""

import asyncio
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableParallel
from shared.llm import get_llm


def demo_sync_stream():
    """Synchronous token streaming — simplest approach."""
    print("\n" + "=" * 60)
    print("1. Synchronous Token Streaming (.stream)")
    print("=" * 60)

    llm = get_llm()
    prompt = ChatPromptTemplate.from_template(
        "Write a haiku about {topic}."
    )

    chain = prompt | llm | StrOutputParser()

    print("\nStreaming tokens:")
    for chunk in chain.stream({"topic": "programming"}):
        print(chunk, end="", flush=True)
    print()


def demo_parallel_stream():
    """Stream from multiple parallel chains simultaneously."""
    print("\n" + "=" * 60)
    print("2. Parallel Streaming (RunnableParallel)")
    print("=" * 60)

    llm = get_llm()

    haiku_chain = (
        ChatPromptTemplate.from_template("Write a haiku about {topic}.")
        | llm
        | StrOutputParser()
    )

    limerick_chain = (
        ChatPromptTemplate.from_template("Write a limerick about {topic}.")
        | llm
        | StrOutputParser()
    )

    parallel = RunnableParallel(haiku=haiku_chain, limerick=limerick_chain)

    print("\nStreaming both simultaneously:")
    for chunk in parallel.stream({"topic": "coffee"}):
        # Each chunk has partial results from whichever chain produced output
        for key, value in chunk.items():
            if value:
                print(f"[{key}] {value}", end="", flush=True)
    print()


async def demo_astream_events():
    """astream_events() — the most detailed streaming API.

    Emits events for every lifecycle step: chain start, LLM start,
    token stream, LLM end, chain end. Perfect for building UIs that
    show what the system is doing at each step.
    """
    print("\n" + "=" * 60)
    print("3. Async Event Streaming (astream_events v2)")
    print("=" * 60)

    llm = get_llm()

    prompt = ChatPromptTemplate.from_template(
        "In one sentence, explain what {concept} is."
    )

    chain = prompt | llm | StrOutputParser()

    print("\nStreaming with full event visibility:")
    async for event in chain.astream_events(
        {"concept": "recursion"},
        version="v2",
    ):
        kind = event["event"]

        if kind == "on_chat_model_start":
            print("\n[LLM START] Model is thinking...")

        elif kind == "on_chat_model_stream":
            content = event["data"]["chunk"].content
            if content:
                print(content, end="", flush=True)

        elif kind == "on_chat_model_end":
            print("\n[LLM END] Response complete")

    print()


async def demo_async_stream():
    """Async streaming with .astream() — for async applications."""
    print("\n" + "=" * 60)
    print("4. Async Streaming (.astream)")
    print("=" * 60)

    llm = get_llm()

    prompt = ChatPromptTemplate.from_template(
        "List 3 benefits of {technology}."
    )

    chain = prompt | llm | StrOutputParser()

    print("\nAsync streaming:")
    async for chunk in chain.astream({"technology": "async programming"}):
        print(chunk, end="", flush=True)
    print()


def main():
    print("=" * 60)
    print("Exercise 10: Streaming")
    print("=" * 60)

    demo_sync_stream()
    demo_parallel_stream()

    # Run async demos
    asyncio.run(demo_astream_events())
    asyncio.run(demo_async_stream())


if __name__ == "__main__":
    main()
