"""
Exercise 15 — LangSmith Tracing & Evaluation
============================================
Observability for LangChain — see every chain step, tool call, and token in a web UI.
Also covers evaluation: create test datasets, score outputs, compare models.

Concepts introduced:
- Automatic tracing — zero code changes, just set env vars
- @traceable decorator — granular span-level tracing
- Custom run names, tags, and metadata via RunnableConfig
- Dataset creation — build eval datasets from examples
- Evaluation — score chain outputs for correctness
- Feedback — programmatic scoring and run annotation

Prerequisite: free account at https://smith.langchain.com → API key
Set in .env: LANGCHAIN_TRACING_V2=true, LANGCHAIN_API_KEY=ls_...
"""

import os
from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableConfig
from shared.llm import get_llm

load_dotenv()

# Check if LangSmith is configured
LANGSMITH_READY = bool(
    os.environ.get("LANGCHAIN_API_KEY")
    and os.environ.get("LANGCHAIN_TRACING_V2") == "true"
)


def demo_tracing_check():
    """Verify LangSmith setup without making API calls."""
    print("\n" + "=" * 60)
    print("1. Tracing Setup Check")
    print("=" * 60)

    if LANGSMITH_READY:
        print("LangSmith tracing is ENABLED")
        print(f"  Project: {os.environ.get('LANGCHAIN_PROJECT', 'default')}")
        print(f"  API key: {os.environ['LANGCHAIN_API_KEY'][:10]}...")
        print("  All chain/agent runs will appear in smith.langchain.com")
    else:
        print("LangSmith tracing is DISABLED")
        print("  To enable: set LANGCHAIN_TRACING_V2=true in .env")
        print("  and LANGCHAIN_API_KEY=ls_... from smith.langchain.com")
        print("  (Free tier works — 3,000 traces/month)")
        print("\n  Running in demo mode — tracing code is present but inert.")


def demo_traceable_decorator():
    """@traceable — add custom spans to any function.

    Even without LangSmith set up, @traceable is a no-op and the code runs fine.
    """
    print("\n" + "=" * 60)
    print("2. @traceable Decorator — Custom Spans")
    print("=" * 60)

    # Import traceable — it works as a no-op when LangSmith is not configured
    from langsmith import traceable

    @traceable(run_type="chain", name="preprocess-input")
    def preprocess(text: str) -> str:
        """Clean and normalize input text."""
        return text.strip().lower()

    @traceable(run_type="chain", name="postprocess-output")
    def postprocess(text: str) -> str:
        """Format the output for display."""
        return text.capitalize().rstrip(".") + "."

    # Use traceable functions in a pipeline
    llm = get_llm()
    prompt = ChatPromptTemplate.from_template(
        "In one word, describe the opposite of: {word}"
    )
    chain = prompt | llm | StrOutputParser()

    words = ["hot", "fast", "light"]

    for word in words:
        cleaned = preprocess(word)
        result = chain.invoke({"word": cleaned})
        final = postprocess(result)
        print(f"  '{word}' → cleaned: '{cleaned}' → LLM: '{result}' → final: '{final}'")

    print("\n  If LangSmith is enabled, each preprocess/LLM/postprocess call")
    print("  appears as a separate span in the trace.")

    if not LANGSMITH_READY:
        print("  (Currently in no-op mode — set LANGCHAIN_TRACING_V2=true to see)")



def demo_run_config():
    """Customize runs with RunnableConfig — names, tags, metadata."""
    print("\n" + "=" * 60)
    print("3. Run Names, Tags & Metadata (RunnableConfig)")
    print("=" * 60)

    llm = get_llm()
    prompt = ChatPromptTemplate.from_template(
        "Give me one fun fact about {topic}."
    )
    chain = prompt | llm | StrOutputParser()

    topics = [
        ("dolphins", "production", "science"),
        ("black holes", "experiment", "astronomy"),
        ("quantum computing", "production", "computing"),
    ]

    for topic, env, tag in topics:
        config = RunnableConfig(
            run_name=f"fact-{topic.replace(' ', '-')}",
            tags=[env, tag, "exercise-15"],
            metadata={
                "topic": topic,
                "experiment": "langsmith-demo",
            },
        )
        result = chain.invoke({"topic": topic}, config=config)
        print(f"  [{tag}] {topic}: {result}")

    print("\n  In LangSmith: filter by tag='experiment' or metadata.topic='dolphins'")


def demo_dataset_creation():
    """Create an evaluation dataset programmatically.

    In production, you'd load these from a CSV or curated examples.
    """
    print("\n" + "=" * 60)
    print("4. Dataset Creation & Management")
    print("=" * 60)

    # Define eval examples
    examples = [
        {
            "input": "What is LangChain?",
            "expected": "A framework for building LLM-powered applications.",
        },
        {
            "input": "What is RAG?",
            "expected": "Retrieval-Augmented Generation — grounding LLM responses in documents.",
        },
        {
            "input": "What is an agent?",
            "expected": "An LLM-powered system that uses tools to accomplish tasks.",
        },
    ]

    print(f"Defined {len(examples)} evaluation examples")

    if LANGSMITH_READY:
        from langsmith import Client

        client = Client()
        dataset_name = "langchain-basics-qa"

        # Check if dataset exists, create if not
        existing = list(client.list_datasets(dataset_name=dataset_name))
        if existing:
            dataset = existing[0]
            print(f"Using existing dataset: {dataset_name} (id={dataset.id})")
        else:
            dataset = client.create_dataset(
                dataset_name=dataset_name,
                description="Basic LangChain Q&A evaluation dataset",
            )
            print(f"Created dataset: {dataset_name} (id={dataset.id})")

            # Add examples
            client.create_examples(
                inputs=[{"question": ex["input"]} for ex in examples],
                outputs=[{"answer": ex["expected"]} for ex in examples],
                dataset_id=dataset.id,
            )
            print(f"Added {len(examples)} examples to dataset")
    else:
        print("\n  Skipping actual dataset creation (LangSmith not configured)")
        print("  With LangSmith enabled, this would:")
        print("  1. Create dataset 'langchain-basics-qa'")
        print("  2. Add 3 (question, expected_answer) examples")
        print("  3. Ready for evaluation with evaluate()")


def demo_evaluation():
    """Run evaluation: score a chain against a dataset.

    Demonstrates the evaluation pattern without requiring LangSmith.
    """
    print("\n" + "=" * 60)
    print("5. Evaluation Pattern")
    print("=" * 60)

    # Define a simple evaluation function
    def simple_evaluator(output: str, expected: str) -> dict:
        """Score output against expected answer."""
        output_lower = output.lower()
        expected_lower = expected.lower()

        # Check for key terms from expected answer
        key_terms = expected_lower.split()
        matches = sum(1 for term in key_terms if term in output_lower)
        score = min(1.0, matches / max(len(key_terms), 1))

        return {
            "score": round(score, 2),
            "key_terms_matched": matches,
            "total_key_terms": len(key_terms),
        }

    # Simulate evaluating a chain against examples
    llm = get_llm()
    prompt = ChatPromptTemplate.from_template(
        "Answer in one sentence: {question}"
    )
    chain = prompt | llm | StrOutputParser()

    test_cases = [
        ("What is LangChain?", "framework for building LLM applications"),
        ("What is RAG?", "grounding responses in documents"),
    ]

    total_score = 0
    for question, expected in test_cases:
        output = chain.invoke({"question": question})
        eval_result = simple_evaluator(output, expected)
        total_score += eval_result["score"]
        print(f"\n  Q: {question}")
        print(f"  Expected: {expected}")
        print(f"  Got: {output[:80]}...")
        print(f"  Score: {eval_result['score']} ({eval_result['key_terms_matched']}/{eval_result['total_key_terms']} terms)")

    avg = total_score / len(test_cases)
    print(f"\n  Average score: {avg:.2f}")

    if LANGSMITH_READY:
        print("\n  With LangSmith: use evaluate() to run this at scale, with")
        print("  side-by-side comparisons and automatic score tracking.")
    else:
        print("\n  Done in local mode. Enable LangSmith for dataset-level eval.")


def demo_comparative_eval():
    """Compare DeepSeek vs Qwen on the same questions.

    This is the foundation of model selection — which provider
    performs better for YOUR specific use case?
    """
    print("\n" + "=" * 60)
    print("6. Comparative Evaluation — DeepSeek vs Qwen")
    print("=" * 60)

    from shared.llm import get_deepseek, get_qwen

    deepseek = get_deepseek()
    qwen = get_qwen()

    prompt = ChatPromptTemplate.from_template(
        "Answer concisely in one sentence: {question}"
    )

    questions = [
        "What is the difference between a list and a tuple in Python?",
        "Explain what a vector database does.",
        "What is the GIL and why does it matter?",
    ]

    for question in questions:
        print(f"\n  Q: {question}")
        for name, llm in [("DeepSeek", deepseek), ("Qwen", qwen)]:
            chain = prompt | llm | StrOutputParser()
            result = chain.invoke({"question": question})
            print(f"    {name}: {result[:120]}...")


def main():
    print("=" * 60)
    print("Exercise 15: LangSmith Tracing & Evaluation")
    print("=" * 60)

    if LANGSMITH_READY:
        print("LangSmith: ENABLED — traces will appear in smith.langchain.com")
    else:
        print("LangSmith: DISABLED — running in demo/no-op mode")

    demo_tracing_check()
    demo_traceable_decorator()
    demo_run_config()
    demo_dataset_creation()
    demo_evaluation()
    demo_comparative_eval()


if __name__ == "__main__":
    main()
