"""
Exercise 06 — RAG Advanced
===========================
Go beyond naive retrieval with advanced techniques.

Concepts introduced:
- MultiQueryRetriever — generate multiple search queries for better recall
- ContextualCompressionRetriever — compress/rerank retrieved documents
- LLMChainExtractor — use an LLM to extract only relevant parts of each doc
- EnsembleRetriever — combine multiple retrievers (semantic + keyword)
- BM25Retriever — keyword-based retrieval for hybrid search
"""

from langchain_chroma import Chroma
from langchain_community.document_loaders import TextLoader, DirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.retrievers import (
    MultiQueryRetriever,
    ContextualCompressionRetriever,
    EnsembleRetriever,
)
from langchain.retrievers.document_compressors import LLMChainExtractor
from langchain_community.retrievers import BM25Retriever
from shared.llm import get_llm
from shared.embeddings import get_embeddings


def load_chunks():
    """Shared: load and split documents."""
    loader = DirectoryLoader(
        "data/sample_docs",
        glob="*.txt",
        loader_cls=TextLoader,
        loader_kwargs={"encoding": "utf-8"},
    )
    documents = loader.load()
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500, chunk_overlap=100
    )
    return splitter.split_documents(documents)


def create_vectorstore(chunks):
    """Shared: create Chroma vector store."""
    return Chroma.from_documents(
        documents=chunks,
        embedding=get_embeddings(),
        collection_name="rag_advanced",
    )


def demo_multi_query():
    """MultiQueryRetriever: generate multiple versions of the question.

    The LLM rephrases the query multiple ways, retrieves docs for each,
    and deduplicates the results. This improves recall for ambiguous queries.
    """
    print("\n" + "=" * 60)
    print("1. MultiQueryRetriever")
    print("=" * 60)

    chunks = load_chunks()
    vectorstore = create_vectorstore(chunks)
    base_retriever = vectorstore.as_retriever(search_kwargs={"k": 2})

    llm = get_llm(temperature=0.3)
    mq_retriever = MultiQueryRetriever.from_llm(
        retriever=base_retriever,
        llm=llm,
    )

    # Show what queries it generates (logging)
    import logging
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger("langchain.retrievers.multi_query")
    logger.setLevel(logging.INFO)

    question = "How do I write better Python code?"
    print(f"\nOriginal question: {question}")

    docs = mq_retriever.invoke(question)
    print(f"Retrieved {len(docs)} unique documents:")
    for i, doc in enumerate(docs):
        print(f"  [{i}] {doc.page_content[:120]}...")


def demo_contextual_compression():
    """ContextualCompressionRetriever: compress docs to only relevant parts.

    Uses LLMChainExtractor to extract only the contextually relevant
    portions of each retrieved document, reducing noise.
    """
    print("\n" + "=" * 60)
    print("2. ContextualCompressionRetriever (LLMChainExtractor)")
    print("=" * 60)

    chunks = load_chunks()
    vectorstore = create_vectorstore(chunks)
    base_retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

    llm = get_llm()
    compressor = LLMChainExtractor.from_llm(llm)
    compression_retriever = ContextualCompressionRetriever(
        base_compressor=compressor,
        base_retriever=base_retriever,
    )

    question = "What tips does the document give about error handling and streaming?"
    print(f"\nQuestion: {question}")

    # Compare: without compression vs with compression
    base_docs = base_retriever.invoke(question)
    print(f"\nWithout compression: {len(base_docs)} docs")
    for i, doc in enumerate(base_docs):
        print(f"  [{i}] {len(doc.page_content)} chars: {doc.page_content[:100]}...")

    compressed_docs = compression_retriever.invoke(question)
    print(f"\nWith compression: {len(compressed_docs)} docs")
    for i, doc in enumerate(compressed_docs):
        print(f"  [{i}] {len(doc.page_content)} chars: {doc.page_content[:100]}...")


def demo_hybrid_search():
    """EnsembleRetriever: combine semantic + keyword search.

    BM25 (keyword) + Vector (semantic) = Hybrid Search.
    BM25 catches exact keyword matches; vector search catches meaning.
    """
    print("\n" + "=" * 60)
    print("3. Hybrid Search (BM25 + Vector)")
    print("=" * 60)

    chunks = load_chunks()
    vectorstore = create_vectorstore(chunks)

    # Semantic retriever
    semantic_retriever = vectorstore.as_retriever(search_kwargs={"k": 2})

    # Keyword retriever (BM25)
    bm25_retriever = BM25Retriever.from_documents(chunks)
    bm25_retriever.k = 2

    # Ensemble combines both, weighted equally
    ensemble = EnsembleRetriever(
        retrievers=[bm25_retriever, semantic_retriever],
        weights=[0.4, 0.6],  # 40% keyword, 60% semantic
    )

    question = "LangChain components and models"
    print(f"\nQuestion: {question}")

    docs = ensemble.invoke(question)
    print(f"Retrieved {len(docs)} documents (hybrid):")
    for i, doc in enumerate(docs):
        print(f"  [{i}] {doc.page_content[:120]}...")


def demo_compare_strategies():
    """Side-by-side comparison: naive vs multi-query vs compression."""
    print("\n" + "=" * 60)
    print("4. Strategy Comparison — Same Query, Different Retrievers")
    print("=" * 60)

    chunks = load_chunks()
    vectorstore = create_vectorstore(chunks)
    base_retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
    llm = get_llm()

    # Build retrievers
    mq_retriever = MultiQueryRetriever.from_llm(retriever=base_retriever, llm=llm)
    compressor = LLMChainExtractor.from_llm(llm)
    compression_retriever = ContextualCompressionRetriever(
        base_compressor=compressor, base_retriever=base_retriever
    )

    question = "How do AI agents work?"

    strategies = {
        "Naive (base)": base_retriever,
        "Multi-query": mq_retriever,
        "Compression": compression_retriever,
    }

    for name, retriever in strategies.items():
        docs = retriever.invoke(question)
        total_chars = sum(len(d.page_content) for d in docs)
        print(f"\n{name}: {len(docs)} docs, {total_chars} total chars")
        print(f"  First chunk: {docs[0].page_content[:100]}..." if docs else "  No results")


def main():
    print("=" * 60)
    print("Exercise 06: RAG Advanced")
    print("=" * 60)

    demo_multi_query()
    demo_contextual_compression()
    demo_hybrid_search()
    demo_compare_strategies()


if __name__ == "__main__":
    main()
