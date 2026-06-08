"""
Exercise 13 — LangGraph Advanced
================================
Multi-agent collaboration, subgraphs, and human-in-the-loop patterns.

Concepts introduced:
- Subgraphs — compose graphs as nodes in larger graphs
- MemorySaver — persistent checkpoints for conversation state
- interrupt() — pause execution for human approval
- Command — resume from interrupt with user input
- Multi-agent handoff — one agent delegates to another
- Thread-level persistence — multiple concurrent conversations
"""

from typing import TypedDict, Annotated, Literal
import operator
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver
from langgraph.types import interrupt, Command
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from shared.llm import get_llm


# --- Multi-Agent State ---


class MultiAgentState(TypedDict):
    messages: Annotated[list, operator.add]
    current_agent: str
    task_complete: bool


# --- Agent Nodes ---


def researcher_node(state: MultiAgentState) -> dict:
    """Research agent — gathers information about a topic."""
    llm = get_llm()

    prompt = ChatPromptTemplate.from_template(
        "You are a RESEARCH agent. Provide 3 key facts about: {topic}. "
        "Be concise and factual."
    )

    chain = prompt | llm | StrOutputParser()
    last_msg = state["messages"][-1].content if state["messages"] else ""

    response = chain.invoke({"topic": last_msg})
    print(f"  [Researcher] Found facts")

    return {
        "messages": [AIMessage(content=f"[Researcher] {response}")],
        "current_agent": "writer",
    }


def writer_node(state: MultiAgentState) -> dict:
    """Writer agent — transforms research into prose."""
    llm = get_llm()

    facts = state["messages"][-1].content if state["messages"] else ""

    prompt = ChatPromptTemplate.from_template(
        "You are a WRITER agent. Based on these research facts, write "
        "a short engaging paragraph:\n\n{facts}"
    )

    chain = prompt | llm | StrOutputParser()
    response = chain.invoke({"facts": facts})
    print(f"  [Writer] Composed paragraph")

    return {
        "messages": [AIMessage(content=f"[Writer] {response}")],
        "current_agent": "editor",
    }


def editor_node(state: MultiAgentState) -> dict:
    """Editor agent — reviews and polishes the output."""
    llm = get_llm()

    text = state["messages"][-1].content if state["messages"] else ""

    prompt = ChatPromptTemplate.from_template(
        "You are an EDITOR agent. Review this text and improve it for "
        "clarity, grammar, and impact. Return the improved version:\n\n{text}"
    )

    chain = prompt | llm | StrOutputParser()
    response = chain.invoke({"text": text})
    print(f"  [Editor] Final polish applied")

    return {
        "messages": [AIMessage(content=f"[Editor - Final] {response}")],
        "task_complete": True,
    }


def router(state: MultiAgentState) -> Literal["researcher", "writer", "editor", END]:
    """Route based on current agent in the pipeline."""
    if state.get("task_complete"):
        return END

    agent = state.get("current_agent", "researcher")
    if agent == "writer":
        return "writer"
    elif agent == "editor":
        return "editor"
    return "researcher"


def demo_multi_agent_pipeline():
    """Researcher → Writer → Editor pipeline."""
    print("\n" + "=" * 60)
    print("1. Multi-Agent Pipeline (Researcher → Writer → Editor)")
    print("=" * 60)

    graph = StateGraph(MultiAgentState)

    graph.add_node("researcher", researcher_node)
    graph.add_node("writer", writer_node)
    graph.add_node("editor", editor_node)

    graph.set_entry_point("researcher")
    graph.add_conditional_edges("researcher", router, {
        "writer": "writer", "editor": "editor", END: END,
    })
    graph.add_conditional_edges("writer", router, {
        "researcher": "researcher", "editor": "editor", END: END,
    })
    graph.add_conditional_edges("editor", router, {
        "researcher": "researcher", "writer": "writer", END: END,
    })

    memory = MemorySaver()
    app = graph.compile(checkpointer=memory)

    thread = {"configurable": {"thread_id": "demo-1"}}

    print("\nTopic: 'How do black holes form?'")
    result = app.invoke(
        {"messages": [HumanMessage(content="How do black holes form?")]},
        config=thread,
    )

    # Print the pipeline output
    for msg in result["messages"]:
        role = "👤" if isinstance(msg, HumanMessage) else "🤖"
        print(f"\n{role} {msg.content[:300]}...")


def demo_human_in_the_loop():
    """Pause execution for human approval before proceeding."""
    print("\n" + "=" * 60)
    print("2. Human-in-the-Loop (Approval Gate)")
    print("=" * 60)

    # Simplified state
    class ApprovalState(TypedDict):
        messages: Annotated[list, operator.add]
        approved: bool

    def generate_draft(state: ApprovalState) -> dict:
        llm = get_llm()
        prompt = ChatPromptTemplate.from_template(
            "Write a one-sentence tagline for a product: {product}"
        )
        chain = prompt | llm | StrOutputParser()
        last = state["messages"][-1].content if state["messages"] else ""
        draft = chain.invoke({"product": last})
        return {"messages": [AIMessage(content=f"DRAFT: {draft}")]}

    def human_approval(state: ApprovalState) -> dict:
        """This node pauses and asks for human input."""
        last_msg = state["messages"][-1].content if state["messages"] else ""

        # interrupt() pauses the graph and returns the value to the caller
        user_decision = interrupt({
            "question": "Approve this draft?",
            "draft": last_msg,
        })

        return {"approved": user_decision.get("approved", False)}

    def finalize(state: ApprovalState) -> dict:
        last = state["messages"][-1].content.replace("DRAFT: ", "")
        return {"messages": [AIMessage(content=f"FINAL APPROVED: {last}")]}

    def approval_router(state: ApprovalState) -> Literal["finalize", END]:
        if state.get("approved"):
            return "finalize"
        return END

    # Build graph
    graph = StateGraph(ApprovalState)
    graph.add_node("generate", generate_draft)
    graph.add_node("approval", human_approval)
    graph.add_node("finalize", finalize)
    graph.set_entry_point("generate")
    graph.add_edge("generate", "approval")
    graph.add_conditional_edges("approval", approval_router, {"finalize": "finalize", END: END})
    graph.add_edge("finalize", END)

    app = graph.compile(checkpointer=MemorySaver())

    thread = {"configurable": {"thread_id": "approval-demo"}}

    # First invocation — will interrupt at approval node
    print("\nRunning graph (will pause for approval)...")
    result = app.invoke(
        {"messages": [HumanMessage(content="AI-powered coffee mug")]},
        config=thread,
    )

    # Check if graph was interrupted
    snapshot = app.get_state(thread)
    if snapshot.next:
        print(f"\nGraph paused at: {snapshot.next}")
        print(f"Draft to approve: {snapshot.values.get('messages', [])[-1].content if snapshot.values.get('messages') else 'N/A'}")

        # Resume with approval
        print("\nResuming with APPROVAL...")
        result = app.invoke(
            Command(resume={"approved": True}),
            config=thread,
        )
        for msg in result.get("messages", []):
            if hasattr(msg, "content"):
                print(f"  {msg.content[:200]}...")


def main():
    print("=" * 60)
    print("Exercise 13: LangGraph Advanced")
    print("=" * 60)

    demo_multi_agent_pipeline()
    demo_human_in_the_loop()


if __name__ == "__main__":
    main()
