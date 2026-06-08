"""
Exercise 12 — LangGraph Basics
==============================
Build stateful graphs with branching logic — the foundation of agentic systems.

Concepts introduced:
- StateGraph — define a graph with typed state
- TypedDict + Annotated — graph state definition with reducers
- add_node() — add processing steps
- add_edge() — fixed transitions
- add_conditional_edges() — dynamic routing based on state
- .compile() — build the runnable graph
- .invoke() / .stream() — execute the graph
- Command — explicit next-node control

Graph:
    [START] → classify → [route] → respond → [END]
                             ↘ escalate → [END]
"""

from typing import TypedDict, Annotated, Literal
import operator
from langgraph.graph import StateGraph, END
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from shared.llm import get_llm


# --- State Definition ---


class RouterState(TypedDict):
    """State for the routing graph.

    Annotated[list, operator.add] means messages are appended, not replaced.
    """
    messages: Annotated[list, operator.add]
    complexity: str
    response: str


# --- Node Functions ---


def classify_node(state: RouterState) -> dict:
    """Classify the query as 'simple' or 'complex'."""
    llm = get_llm()

    prompt = ChatPromptTemplate.from_template(
        "Classify this query as 'simple' or 'complex'. "
        "A simple query is a greeting or basic fact. "
        "A complex query requires analysis or explanation. "
        "Respond with ONLY the word 'simple' or 'complex'.\n\n"
        "Query: {query}"
    )

    chain = prompt | llm | StrOutputParser()
    last_msg = state["messages"][-1].content if state["messages"] else ""
    result = chain.invoke({"query": last_msg})

    complexity = result.strip().lower()
    print(f"  [Classify] Query complexity: {complexity}")

    return {"complexity": complexity}


def simple_response_node(state: RouterState) -> dict:
    """Handle simple queries with a brief response."""
    llm = get_llm()

    prompt = ChatPromptTemplate.from_template(
        "Answer briefly in one sentence: {query}"
    )

    chain = prompt | llm | StrOutputParser()
    last_msg = state["messages"][-1].content if state["messages"] else ""
    response = chain.invoke({"query": last_msg})

    print(f"  [SimpleRespond] {response[:100]}...")

    return {"response": response}


def complex_response_node(state: RouterState) -> dict:
    """Handle complex queries with a detailed response."""
    llm = get_llm()

    prompt = ChatPromptTemplate.from_template(
        "Provide a detailed, structured answer with 2-3 key points: {query}"
    )

    chain = prompt | llm | StrOutputParser()
    last_msg = state["messages"][-1].content if state["messages"] else ""
    response = chain.invoke({"query": last_msg})

    print(f"  [ComplexRespond] {response[:100]}...")

    return {"response": response}


# --- Router ---


def route_by_complexity(state: RouterState) -> Literal["simple_response", "complex_response"]:
    """Route to the appropriate handler based on complexity."""
    if state["complexity"] == "complex":
        return "complex_response"
    return "simple_response"


def demo_basic_graph():
    """Build and run a simple routing graph."""
    print("\n" + "=" * 60)
    print("1. Basic Routing Graph")
    print("=" * 60)

    # Build the graph
    graph = StateGraph(RouterState)

    graph.add_node("classify", classify_node)
    graph.add_node("simple_response", simple_response_node)
    graph.add_node("complex_response", complex_response_node)

    graph.set_entry_point("classify")
    graph.add_conditional_edges(
        "classify",
        route_by_complexity,
        {
            "simple_response": "simple_response",
            "complex_response": "complex_response",
        },
    )
    graph.add_edge("simple_response", END)
    graph.add_edge("complex_response", END)

    app = graph.compile()

    # Test with different queries
    from langchain_core.messages import HumanMessage

    queries = [
        "Hello! How are you?",
        "Explain the difference between REST and GraphQL APIs.",
    ]

    for q in queries:
        print(f"\n👤 Query: {q}")
        result = app.invoke({"messages": [HumanMessage(content=q)]})
        print(f"Path: classify → {result['complexity']}_response")
        print(f"Response: {result['response'][:200]}...")


def demo_streaming_graph():
    """Stream graph execution — see each step as it happens."""
    print("\n" + "=" * 60)
    print("2. Streaming Graph Execution")
    print("=" * 60)

    graph = StateGraph(RouterState)
    graph.add_node("classify", classify_node)
    graph.add_node("simple_response", simple_response_node)
    graph.add_node("complex_response", complex_response_node)
    graph.set_entry_point("classify")
    graph.add_conditional_edges(
        "classify", route_by_complexity,
        {"simple_response": "simple_response", "complex_response": "complex_response"},
    )
    graph.add_edge("simple_response", END)
    graph.add_edge("complex_response", END)
    app = graph.compile()

    from langchain_core.messages import HumanMessage

    print("\nStreaming graph steps:")
    for step in app.stream(
        {"messages": [HumanMessage(content="What is the capital of France?")]},
        stream_mode="updates",
    ):
        for node_name, update in step.items():
            print(f"  Node '{node_name}' completed: {list(update.keys())}")


def main():
    print("=" * 60)
    print("Exercise 12: LangGraph Basics")
    print("=" * 60)

    demo_basic_graph()
    demo_streaming_graph()


if __name__ == "__main__":
    main()
