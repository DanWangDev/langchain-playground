"""
Exercise 02 — Prompt Templates
==============================
Learn structured prompting with templates.

Concepts introduced:
- ChatPromptTemplate — define a message structure with placeholders
- SystemMessage / HumanMessage — role-based messages
- MessagesPlaceholder — slot for dynamic message history
- Few-shot prompting — provide examples to guide the model
- .format_messages() — render templates into concrete messages
"""

from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import SystemMessage, HumanMessage
from shared.llm import get_llm


def demo_basic_template():
    """A simple system + human prompt template."""
    print("\n--- 1. Basic System + Human Template ---")

    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", "You are a {role}. Answer concisely in {language}."),
            ("human", "{question}"),
        ]
    )

    messages = prompt.format_messages(
        role="Python expert",
        language="English",
        question="What is the difference between a list and a tuple?",
    )

    print("Rendered messages:")
    for msg in messages:
        print(f"  [{msg.type}] {msg.content}")

    llm = get_llm()
    response = llm.invoke(messages)
    print(f"\nResponse: {response.content}")


def demo_history_placeholder():
    """Template with a placeholder for conversation history."""
    print("\n--- 2. MessagesPlaceholder (Conversation History) ---")

    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", "You are a helpful assistant."),
            MessagesPlaceholder(variable_name="history"),
            ("human", "{input}"),
        ]
    )

    # Simulated history
    messages = prompt.format_messages(
        history=[
            HumanMessage(content="My name is Alice."),
            SystemMessage(content="Assistant: Nice to meet you, Alice!"),
        ],
        input="What's my name?",
    )

    print("Rendered messages (with history):")
    for msg in messages:
        print(f"  [{msg.type}] {msg.content}")

    llm = get_llm()
    response = llm.invoke(messages)
    print(f"\nResponse: {response.content}")


def demo_few_shot():
    """Few-shot prompting — provide examples to guide output format."""
    print("\n--- 3. Few-Shot Prompting ---")

    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "Classify the sentiment as 'positive', 'negative', or 'neutral'. "
                "Respond with ONLY the label.\n\n"
                "Examples:\n"
                "Input: I love this product! → positive\n"
                "Input: This is terrible. → negative\n"
                "Input: It arrived on Tuesday. → neutral",
            ),
            ("human", "Input: {text} →"),
        ]
    )

    test_inputs = [
        "The food was absolutely delicious!",
        "I waited 2 hours and it was cold.",
        "The package is blue.",
    ]

    llm = get_llm()
    for text in test_inputs:
        messages = prompt.format_messages(text=text)
        response = llm.invoke(messages)
        print(f"  '{text}' → {response.content.strip()}")


def main():
    print("=" * 60)
    print("Exercise 02: Prompt Templates")
    print("=" * 60)

    demo_basic_template()
    demo_history_placeholder()
    demo_few_shot()


if __name__ == "__main__":
    main()
