"""
Exercise 16 — AWS Bedrock Integration
=====================================
Use managed foundation models via AWS Bedrock — Claude, Titan embeddings,
and full RAG/agent pipelines on AWS infrastructure.

Concepts introduced:
- ChatBedrock — invoke Claude models through AWS Bedrock
- BedrockEmbeddings — Titan embeddings for RAG
- boto3 credential chain — env vars → ~/.aws/credentials → instance profile
- STS identity check — validate AWS access before calling Bedrock
- Provider comparison — Bedrock Claude vs DeepSeek vs Qwen
- Bedrock RAG — Titan embeddings + ChromaDB
- Bedrock Agent — tool-calling agent with Claude on Bedrock

Prerequisite: AWS credentials configured, Bedrock model access requested.
"""

import os
from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_chroma import Chroma
from langchain_community.document_loaders import TextLoader, DirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.agents import create_agent
from langchain_core.tools import tool
from shared.llm import get_deepseek, get_qwen, get_llm
from shared.bedrock import (
    get_bedrock_llm,
    get_bedrock_embeddings,
    get_caller_identity,
)

load_dotenv()


def demo_aws_check():
    """Verify AWS access before attempting Bedrock calls."""
    print("\n" + "=" * 60)
    print("1. AWS Access Check")
    print("=" * 60)

    identity = get_caller_identity()
    if identity:
        print(f"  AWS access: OK")
        print(f"  Account: {identity['Account']}")
        print(f"  ARN: {identity['Arn']}")
        print(f"  Region: {os.environ.get('AWS_REGION', 'us-east-1')}")
    else:
        print("  AWS access: NOT CONFIGURED")
        print("  Configure via one of:")
        print("    - aws configure (sets ~/.aws/credentials)")
        print("    - AWS_ACCESS_KEY_ID + AWS_SECRET_ACCESS_KEY env vars")
        print("    - IAM instance role (EC2/Lambda)")
        print("  Bedrock calls will use FakeLLM fallback.")


def demo_bedrock_chat():
    """Basic Bedrock chat — like exercise 01 but via AWS."""
    print("\n" + "=" * 60)
    print("2. Bedrock Chat — Claude via AWS")
    print("=" * 60)

    llm = get_bedrock_llm()
    prompt = ChatPromptTemplate.from_template(
        "In one sentence, what is AWS Bedrock?"
    )

    chain = prompt | llm | StrOutputParser()

    result = chain.invoke({})
    print(f"  Bedrock Claude: {result}")


def demo_provider_comparison():
    """Same prompt → 3 providers. Compare latency, quality, cost profile."""
    print("\n" + "=" * 60)
    print("3. Provider Comparison — Bedrock vs DeepSeek vs Qwen")
    print("=" * 60)

    import time

    prompt = ChatPromptTemplate.from_template(
        "List 3 key features of {technology} in bullet points."
    )

    question = {"technology": "container orchestration"}

    providers = {
        "Bedrock Claude (Haiku)": get_bedrock_llm(),
        "DeepSeek": get_deepseek(),
        "Qwen": get_qwen(),
    }

    results = {}
    for name, llm in providers.items():
        chain = prompt | llm | StrOutputParser()
        start = time.time()
        result = chain.invoke(question)
        elapsed = time.time() - start
        results[name] = {"output": result, "latency": elapsed}
        print(f"\n  [{name}] — {elapsed:.2f}s")
        print(f"    {result[:150]}...")

    fastest = min(results, key=lambda k: results[k]["latency"])
    print(f"\n  Fastest: {fastest} ({results[fastest]['latency']:.2f}s)")


def demo_bedrock_rag():
    """Full RAG pipeline with Bedrock Titan embeddings — mirror of exercise 05."""
    print("\n" + "=" * 60)
    print("4. Bedrock RAG — Titan Embeddings + ChromaDB")
    print("=" * 60)

    # Load documents
    loader = DirectoryLoader(
        "data/sample_docs",
        glob="*.txt",
        loader_cls=TextLoader,
        loader_kwargs={"encoding": "utf-8"},
    )
    documents = loader.load()
    print(f"  Loaded {len(documents)} document(s)")

    # Split
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500, chunk_overlap=100
    )
    chunks = splitter.split_documents(documents)
    print(f"  Split into {len(chunks)} chunks")

    # Embed + store with Bedrock Titan
    embeddings = get_bedrock_embeddings()
    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        collection_name="bedrock_rag",
    )
    retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

    # QA chain
    llm = get_bedrock_llm()

    qa_prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "Answer the question based on the provided context.\n\n"
                "Context:\n{context}",
            ),
            ("human", "{input}"),
        ]
    )

    from langchain_classic.chains import create_retrieval_chain
    from langchain_classic.chains.combine_documents import create_stuff_documents_chain

    combine_docs_chain = create_stuff_documents_chain(llm=llm, prompt=qa_prompt)
    qa_chain = create_retrieval_chain(retriever, combine_docs_chain)

    questions = [
        "What is LangChain Expression Language?",
        "What are the key components of LangChain?",
    ]

    for q in questions:
        result = qa_chain.invoke({"input": q})
        print(f"\n  Q: {q}")
        print(f"  A: {result['answer'][:200]}...")


def demo_bedrock_agent():
    """Tool-calling agent with Bedrock Claude — mirror of exercise 09."""
    print("\n" + "=" * 60)
    print("5. Bedrock Agent — Tool-Calling with Claude")
    print("=" * 60)

    import math
    from datetime import datetime

    @tool
    def calculator(expression: str) -> str:
        """Evaluate a mathematical expression."""
        try:
            allowed = {k: v for k, v in math.__dict__.items() if not k.startswith("_")}
            allowed["__builtins__"] = {}
            return str(eval(expression, allowed, {}))
        except Exception as e:
            return f"Error: {e}"

    @tool
    def get_current_time() -> str:
        """Get the current date and time."""
        return datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    llm = get_bedrock_llm()
    tools = [calculator, get_current_time]

    agent = create_agent(llm, tools)

    result = agent.invoke(
        {"messages": [("user", "What time is it, and what is 25 * 4?")]}
    )

    for msg in result["messages"]:
        role = msg.__class__.__name__.replace("Message", "")
        content = getattr(msg, "content", "")
        tool_calls = getattr(msg, "tool_calls", [])
        if content:
            print(f"  [{role}] {str(content)[:200]}")
        for tc in tool_calls:
            print(f"  [{role} → 🔧] {tc['name']}({tc['args']})")


def demo_boto3_session():
    """boto3 session management — profiles, cross-region, credential chain."""
    print("\n" + "=" * 60)
    print("6. boto3 Session Management")
    print("=" * 60)

    import boto3

    # Default session — uses env vars → credentials file → instance profile
    default_session = boto3.Session()
    print(f"  Default region: {default_session.region_name or 'unset'}")

    # Explicit region
    us_east = boto3.Session(region_name="us-east-1")
    print(f"  us-east-1 session region: {us_east.region_name}")

    # Named profile (if configured in ~/.aws/credentials)
    profile = os.environ.get("AWS_PROFILE", "default")
    print(f"  Current profile: {profile}")

    # List available profiles (from ~/.aws/credentials and ~/.aws/config)
    try:
        available = boto3.Session().available_profiles
        print(f"  Available profiles: {', '.join(available)}")
    except Exception:
        print("  Available profiles: (unable to list)")

    # Credential chain resolution order:
    print("\n  boto3 credential resolution order:")
    print("    1. Environment variables (AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY)")
    print("    2. Shared credentials file (~/.aws/credentials)")
    print("    3. AWS config file (~/.aws/config)")
    print("    4. IAM instance profile (EC2, Lambda, ECS)")
    print("    5. Container credential provider (ECS task role)")
    print("    6. SSO credential provider (~/.aws/sso/cache)")

    # Verify which method is active
    identity = get_caller_identity()
    if identity:
        print(f"\n  Currently authenticated as: {identity['Arn']}")
    else:
        print("\n  Not authenticated — Bedrock will use FakeLLM fallback")


def main():
    print("=" * 60)
    print("Exercise 16: AWS Bedrock Integration")
    print("=" * 60)

    demo_aws_check()
    demo_bedrock_chat()
    demo_provider_comparison()
    demo_bedrock_rag()
    demo_bedrock_agent()
    demo_boto3_session()


if __name__ == "__main__":
    main()
