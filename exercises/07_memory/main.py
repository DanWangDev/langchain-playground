"""
Exercise 07 — Memory
====================
Manage conversation history across turns so the LLM remembers previous exchanges.

Concepts introduced:
- ChatMessageHistory — simple in-memory message list
- RunnableWithMessageHistory — wrap any chain with automatic history injection
- get_session_history — factory function for session-scoped history
- ConversationBufferMemory — the classic (legacy) approach
- Session management — multiple concurrent conversations

Pattern:
    store = {}  # session_id → ChatMessageHistory
    chain_with_history = RunnableWithMessageHistory(chain, get_session_history, ...)
"""

from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.chat_history import (
    BaseChatMessageHistory,
    InMemoryChatMessageHistory,
)
from langchain_core.messages import HumanMessage, AIMessage
from shared.llm import get_llm


# Session store — maps session_id to chat history
# In production, replace with a database-backed store
store: dict[str, InMemoryChatMessageHistory] = {}


def get_session_history(session_id: str) -> BaseChatMessageHistory:
    """Factory: return (or create) chat history for a session."""
    if session_id not in store:
        store[session_id] = InMemoryChatMessageHistory()
    return store[session_id]


def demo_basic_memory():
    """Wrap a prompt chain with conversation history."""
    print("\n" + "=" * 60)
    print("1. RunnableWithMessageHistory — Basic Conversation")
    print("=" * 60)

    llm = get_llm()

    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", "You are a helpful assistant. Answer concisely."),
            MessagesPlaceholder(variable_name="history"),
            ("human", "{input}"),
        ]
    )

    chain = prompt | llm | StrOutputParser()

    chain_with_history = RunnableWithMessageHistory(
        chain,
        get_session_history,
        input_messages_key="input",
        history_messages_key="history",
    )

    # Simulate a multi-turn conversation
    session_id = "user-123"
    turns = [
        "My name is Alice and I'm learning Python.",
        "What was my name again?",
        "Can you suggest a good first Python project for me?",
    ]

    for turn in turns:
        print(f"\n👤 User: {turn}")
        response = chain_with_history.invoke(
            {"input": turn},
            config={"configurable": {"session_id": session_id}},
        )
        print(f"🤖 Assistant: {response}")


def demo_multiple_sessions():
    """Multiple concurrent sessions with isolated history."""
    print("\n" + "=" * 60)
    print("2. Multiple Sessions — Isolated Conversations")
    print("=" * 60)

    llm = get_llm()

    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", "You are a helpful assistant."),
            MessagesPlaceholder(variable_name="history"),
            ("human", "{input}"),
        ]
    )

    chain = prompt | llm | StrOutputParser()
    chain_with_history = RunnableWithMessageHistory(
        chain,
        get_session_history,
        input_messages_key="input",
        history_messages_key="history",
    )

    # Session A: talks about cats
    chain_with_history.invoke(
        {"input": "I have a cat named Whiskers."},
        config={"configurable": {"session_id": "session-a"}},
    )

    # Session B: talks about dogs
    chain_with_history.invoke(
        {"input": "I have a dog named Rover."},
        config={"configurable": {"session_id": "session-b"}},
    )

    # Now ask both sessions the same question — different answers expected
    for sid in ["session-a", "session-b"]:
        response = chain_with_history.invoke(
            {"input": "What pet do I have and what's its name?"},
            config={"configurable": {"session_id": sid}},
        )
        print(f"\nSession '{sid}': {response}")


def demo_manual_history():
    """Manually manage history — useful for custom history logic."""
    print("\n" + "=" * 60)
    print("3. Manual History Management")
    print("=" * 60)

    llm = get_llm()

    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", "Answer the user's question. Be brief."),
            MessagesPlaceholder(variable_name="history"),
            ("human", "{input}"),
        ]
    )

    chain = prompt | llm | StrOutputParser()

    # Manual history list
    history = []

    interactions = [
        "What is 2 + 2?",
        "Multiply that by 3.",
    ]

    for user_input in interactions:
        result = chain.invoke({"input": user_input, "history": history})
        print(f"\n👤: {user_input}")
        print(f"🤖: {result}")
        # Update history manually
        history.append(HumanMessage(content=user_input))
        history.append(AIMessage(content=result))


def demo_window_memory():
    """Limit history to last N messages — prevent context overflow."""
    print("\n" + "=" * 60)
    print("4. Sliding Window — Keep Only Last N Messages")
    print("=" * 60)

    llm = get_llm()

    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", "You are a helpful assistant."),
            MessagesPlaceholder(variable_name="history"),
            ("human", "{input}"),
        ]
    )

    chain = prompt | llm | StrOutputParser()

    # Fixed-size window: keep only last 4 messages (2 turns)
    MAX_HISTORY = 4
    history = []

    facts = [
        "My favorite color is blue.",
        "I live in London.",
        "I'm a software engineer.",
        "I play the guitar.",
        "What's my favorite color?",  # Should remember (was 4 turns ago = borderline)
        "What city do I live in?",  # Blue might be dropped by window
    ]

    for user_input in facts:
        result = chain.invoke({"input": user_input, "history": history})
        print(f"\n👤: {user_input}")
        print(f"🤖: {result}")
        # Update history and trim
        history.append(HumanMessage(content=user_input))
        history.append(AIMessage(content=result))
        if len(history) > MAX_HISTORY:
            history = history[-MAX_HISTORY:]
            print(f"   [history trimmed to last {MAX_HISTORY} messages]")


def main():
    print("=" * 60)
    print("Exercise 07: Memory")
    print("=" * 60)

    demo_basic_memory()
    demo_multiple_sessions()
    demo_manual_history()
    demo_window_memory()


if __name__ == "__main__":
    main()
