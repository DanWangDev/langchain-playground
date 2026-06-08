"""
Exercise 03 — Chains & LCEL (LangChain Expression Language)
===========================================================
Compose components using the pipe operator (|).

Concepts introduced:
- LCEL pipe operator: component_a | component_b | component_c
- StrOutputParser — extract string content from AIMessage
- RunnableParallel — run multiple chains concurrently
- RunnableLambda — wrap a plain function as a runnable
- RunnablePassthrough — pass input through unchanged
- .assign() — add computed fields to the input dict
- .bind() — pre-set arguments on a runnable
"""

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableParallel, RunnableLambda, RunnablePassthrough
from shared.llm import get_llm


def demo_simple_chain():
    """The simplest LCEL chain: prompt | llm | parser."""
    print("\n--- 1. Simple Chain (prompt | llm | parser) ---")

    prompt = ChatPromptTemplate.from_template(
        "Explain {topic} in exactly one sentence."
    )
    llm = get_llm()

    chain = prompt | llm | StrOutputParser()

    result = chain.invoke({"topic": "vector databases"})
    print(f"Result: {result}")


def demo_parallel():
    """RunnableParallel — run multiple chains at once."""
    print("\n--- 2. RunnableParallel — Multiple Perspectives ---")

    llm = get_llm()

    summary_chain = (
        ChatPromptTemplate.from_template(
            "Summarize {topic} in one sentence."
        )
        | llm
        | StrOutputParser()
    )

    pros_chain = (
        ChatPromptTemplate.from_template(
            "List 2 advantages of {topic}."
        )
        | llm
        | StrOutputParser()
    )

    cons_chain = (
        ChatPromptTemplate.from_template(
            "List 2 disadvantages of {topic}."
        )
        | llm
        | StrOutputParser()
    )

    multi_perspective = RunnableParallel(
        summary=summary_chain,
        pros=pros_chain,
        cons=cons_chain,
    )

    result = multi_perspective.invoke({"topic": "microservices architecture"})
    for key, value in result.items():
        print(f"\n[{key}]")
        print(f"  {value}")


def demo_lambda_and_passthrough():
    """RunnableLambda + RunnablePassthrough for data transformation."""
    print("\n--- 3. RunnableLambda + RunnablePassthrough ---")

    llm = get_llm()

    # A simple chain that counts words in the response
    prompt = ChatPromptTemplate.from_template(
        "Describe {thing} in 2-3 sentences."
    )

    chain = (
        prompt
        | llm
        | StrOutputParser()
        | RunnableLambda(lambda text: f"Response ({len(text.split())} words): {text}")
    )

    result = chain.invoke({"thing": "the Python GIL"})
    print(f"Result: {result}")


def demo_assign():
    """.assign() — add computed fields while passing through the original input."""
    print("\n--- 4. .assign() — Enrich Input with Computed Fields ---")

    llm = get_llm()

    chain = (
        RunnablePassthrough.assign(
            word_count=lambda x: len(x["text"].split()),
            is_long=lambda x: len(x["text"]) > 100,
        )
    )

    result = chain.invoke({"text": "The quick brown fox jumps over the lazy dog."})
    print(f"Input text: {result['text']}")
    print(f"Word count: {result['word_count']}")
    print(f"Is long? {result['is_long']}")


def demo_bind():
    """.bind() — pre-set arguments on a runnable."""
    print("\n--- 5. .bind() — Pre-set Model Parameters ---")

    llm = get_llm()

    # Create two variants of the same LLM with different temperatures
    creative_llm = llm.bind(temperature=1.5)
    precise_llm = llm.bind(temperature=0.0)

    prompt = ChatPromptTemplate.from_template(
        "Give me a creative name for a {thing} startup."
    )

    creative_chain = prompt | creative_llm | StrOutputParser()
    precise_chain = prompt | precise_llm | StrOutputParser()

    topic = "pet-sitting"
    print(f"Creative (temp=1.5): {creative_chain.invoke({'thing': topic})}")
    print(f"Precise  (temp=0.0): {precise_chain.invoke({'thing': topic})}")


def main():
    print("=" * 60)
    print("Exercise 03: Chains & LCEL")
    print("=" * 60)

    demo_simple_chain()
    demo_parallel()
    demo_lambda_and_passthrough()
    demo_assign()
    demo_bind()


if __name__ == "__main__":
    main()
