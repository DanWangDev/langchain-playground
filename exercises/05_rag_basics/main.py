"""
Exercise 05 — RAG Basics
=========================
Build a complete Retrieval-Augmented Generation pipeline from scratch.

Concepts introduced:
- TextLoader — load documents from files
- RecursiveCharacterTextSplitter — chunk documents for embedding
- OpenAIEmbeddings (pointed at DashScope) — generate embeddings via Qwen
- Chroma — local vector store for semantic search
- create_stuff_documents_chain — combine retrieved docs into a prompt
- create_retrieval_chain — wire retriever + QA chain together
- create_history_aware_retriever — rephrase questions using chat history

RAG Pipeline:
  Documents → Split → Embed → Store → Retrieve → Augment → Generate
"""

from langchain_chroma import Chroma
from langchain_community.document_loaders import TextLoader, DirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser
from langchain_classic.chains import create_retrieval_chain
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_classic.chains.history_aware_retriever import create_history_aware_retriever
from shared.llm import get_llm
from shared.embeddings import get_embeddings


def load_and_split_documents(docs_dir: str = "data/sample_docs"):
    """Load .txt files from the data directory and split into chunks."""
    print(f"\nLoading documents from: {docs_dir}")

    loader = DirectoryLoader(
        docs_dir,
        glob="*.txt",
        loader_cls=TextLoader,
        loader_kwargs={"encoding": "utf-8"},
    )
    documents = loader.load()
    print(f"Loaded {len(documents)} document(s)")

    for doc in documents:
        print(f"  - {doc.metadata['source']} ({len(doc.page_content)} chars)")

    # Split into overlapping chunks
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=100,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    chunks = splitter.split_documents(documents)
    print(f"Split into {len(chunks)} chunks")

    return chunks


def demo_basic_rag():
    """Step-by-step: build a simple RAG pipeline manually."""
    print("\n" + "=" * 60)
    print("1. Basic RAG — Step by Step")
    print("=" * 60)

    chunks = load_and_split_documents()

    # Create vector store
    print("\nCreating vector store with Qwen embeddings...")
    embeddings = get_embeddings()
    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        collection_name="langchain_playground",
    )

    # Retrieve relevant chunks
    retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
    question = "What is LangChain and what are its key components?"
    retrieved = retriever.invoke(question)

    print(f"\nQuestion: {question}")
    print(f"Retrieved {len(retrieved)} chunks:")
    for i, doc in enumerate(retrieved):
        print(f"  [{i}] {doc.page_content[:150]}...")

    # Generate answer from retrieved context
    llm = get_llm()
    context = "\n\n".join(doc.page_content for doc in retrieved)

    prompt = ChatPromptTemplate.from_template(
        "Answer the question based on the following context.\n\n"
        "Context:\n{context}\n\n"
        "Question: {question}\n\n"
        "Answer:"
    )

    chain = prompt | llm | StrOutputParser()
    answer = chain.invoke({"context": context, "question": question})
    print(f"\nAnswer: {answer}")


def demo_retrieval_chain():
    """Use LangChain's built-in create_retrieval_chain for cleaner code."""
    print("\n" + "=" * 60)
    print("2. Using create_retrieval_chain (Cleaner)")
    print("=" * 60)

    chunks = load_and_split_documents()
    embeddings = get_embeddings()
    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        collection_name="lc_retrieval_chain",
    )
    retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

    llm = get_llm()

    # The QA prompt — receives context and question
    qa_prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "Answer the question based on the provided context. "
                "If you cannot answer from the context, say so.\n\n"
                "Context:\n{context}",
            ),
            ("human", "{input}"),
        ]
    )

    # combine_docs_chain formats retrieved docs into the prompt and calls the LLM
    combine_docs_chain = create_stuff_documents_chain(
        llm=llm,
        prompt=qa_prompt,
    )

    # retrieval_chain wires retriever → combine_docs
    qa_chain = create_retrieval_chain(retriever, combine_docs_chain)

    questions = [
        "What are the key components of LangChain?",
        "What Python tips are recommended for LangChain development?",
    ]

    for q in questions:
        result = qa_chain.invoke({"input": q})
        print(f"\nQ: {q}")
        print(f"A: {result['answer'][:300]}...")


def demo_history_aware_rag():
    """Add conversation history — the retriever rephrases follow-up questions."""
    print("\n" + "=" * 60)
    print("3. History-Aware RAG (Conversational)")
    print("=" * 60)

    chunks = load_and_split_documents()
    embeddings = get_embeddings()
    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        collection_name="lc_history_rag",
    )
    retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

    llm = get_llm()

    # Prompt to rephrase a follow-up question into a standalone question
    rephrase_prompt = ChatPromptTemplate.from_messages(
        [
            MessagesPlaceholder("chat_history"),
            (
                "human",
                "Given the chat history, rephrase this follow-up into a "
                "standalone question:\n{input}",
            ),
        ]
    )

    # History-aware retriever rephrases the question before retrieving
    history_aware_retriever = create_history_aware_retriever(
        llm=llm,
        retriever=retriever,
        prompt=rephrase_prompt,
    )

    qa_prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "Answer based on context:\n{context}",
            ),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}"),
        ]
    )

    combine_docs_chain = create_stuff_documents_chain(llm, qa_prompt)
    qa_chain = create_retrieval_chain(history_aware_retriever, combine_docs_chain)

    # Simulate a conversation
    chat_history = []
    turns = [
        "What is RAG?",
        "How does it relate to vector stores?",  # follow-up
    ]

    for turn in turns:
        result = qa_chain.invoke(
            {"input": turn, "chat_history": chat_history}
        )
        print(f"\nQ: {turn}")
        print(f"A: {result['answer'][:250]}...")
        # Update history
        chat_history.extend(
            [{"role": "user", "content": turn}, {"role": "assistant", "content": result["answer"]}]
        )


def main():
    print("=" * 60)
    print("Exercise 05: RAG Basics")
    print("=" * 60)

    demo_basic_rag()
    demo_retrieval_chain()
    demo_history_aware_rag()


if __name__ == "__main__":
    main()
