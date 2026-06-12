# Exercise 05: RAG Basics

## What You'll Learn

- **Document Loading** — load `.txt` files from a directory with `DirectoryLoader`
- **Text Splitting** — chunk documents into overlapping pieces with `RecursiveCharacterTextSplitter`
- **Embeddings** — convert text to vectors using Qwen embeddings via DashScope
- **Vector Store** — store and search embeddings with `Chroma`
- **Retrieval Chain** — `create_retrieval_chain` wires retriever + QA chain together
- **History-Aware Retrieval** — rephrase follow-up questions using chat history

## Why RAG Matters

Retrieval-Augmented Generation (RAG) is the most important LLM application pattern. It solves two fundamental problems:

1. **Knowledge cutoff**: LLMs only know what was in their training data. RAG gives them access to your documents.
2. **Hallucination**: LLMs make things up when they don't know. RAG grounds responses in real documents.

```
Without RAG:  User: "What's our return policy?"
              LLM: "I don't have access to your company's policies." ❌

With RAG:     User: "What's our return policy?"
              System: [retrieves policy doc] → "Returns accepted within 30 days with receipt." ✓
```

## The RAG Pipeline

```
┌──────────┐    ┌──────────┐    ┌──────────┐    ┌──────────┐    ┌──────────┐
│  Load    │───▶│  Split   │───▶│  Embed   │───▶│  Store   │───▶│ Retrieve │
│Documents │    │Documents │    │ Chunks   │    │ Vectors  │    │ Top-K    │
└──────────┘    └──────────┘    └──────────┘    └──────────┘    └──────────┘
                                                                     │
                                                                     ▼
                          ┌──────────┐    ┌──────────┐    ┌──────────────┐
                          │  Answer  │◀───│ Generate │◀───│  Augment     │
                          │          │    │  (LLM)   │    │  Prompt+ Docs│
                          └──────────┘    └──────────┘    └──────────────┘
```

### Step 1: Load Documents

```python
loader = DirectoryLoader("data/sample_docs", glob="*.txt", loader_cls=TextLoader)
documents = loader.load()
```

### Step 2: Split into Chunks

```python
splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,       # Max characters per chunk
    chunk_overlap=100,    # Overlap between chunks (prevents split sentences)
    separators=["\n\n", "\n", ". ", " ", ""],  # Split on natural boundaries
)
chunks = splitter.split_documents(documents)
```

**Why overlap?** If a sentence is split across chunks, the overlap ensures it appears completely in at least one chunk.

**Why recursive?** The splitter tries each separator in order — it splits on paragraphs first, then lines, then sentences, then words. This keeps chunks semantically coherent.

### Step 3: Embed & Store

```python
embeddings = get_embeddings()  # Qwen embeddings
vectorstore = Chroma.from_documents(documents=chunks, embedding=embeddings)
```

Embeddings convert text → vectors (lists of numbers). Similar texts have similar vectors. Chroma stores these vectors and can find the most similar ones to any query.

### Step 4: Retrieve

```python
retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
docs = retriever.invoke("What is LangChain?")
```

This finds the 3 chunks most semantically similar to the question.

### Step 5: Augment & Generate

```python
# The QA prompt receives both context and question
qa_prompt = ChatPromptTemplate.from_messages([
    ("system", "Answer based on context:\n{context}"),
    ("human", "{input}"),
])

combine_docs_chain = create_stuff_documents_chain(llm, qa_prompt)
qa_chain = create_retrieval_chain(retriever, combine_docs_chain)
```

`create_stuff_documents_chain` formats retrieved documents into the prompt. `create_retrieval_chain` wires the retriever and QA chain together.

## History-Aware Retrieval

Follow-up questions are often incomplete without context:

```
User: "What is RAG?"              → Full question
User: "How does it work?"         → "It" refers to RAG — needs rephrasing!
```

`create_history_aware_retriever` uses the LLM to rephrase follow-ups before retrieval:

```
"It" → "How does RAG work?"
```

## Key Concepts

### Chunk Size Trade-offs

| Chunk Size | Pros | Cons |
|------------|------|------|
| Small (100-300) | Precise retrieval, less noise | May miss context, split concepts |
| Medium (500-1000) | Good balance | Standard choice |
| Large (2000+) | Full context in one chunk | Noisy retrieval, more tokens |

### Embedding Models

- **Qwen embeddings** (via DashScope): 1024-dimensional vectors, Chinese + English support
- Same embedding model must be used for indexing AND querying

### Vector Similarity

Chroma uses cosine similarity by default: the angle between vectors determines relevance. Smaller angle = more similar.

## Gotchas

1. **Chunk overlap must be < chunk_size**: If overlap ≥ chunk_size, you get duplicate (or near-duplicate) chunks.
2. **Embedding dimension mismatch**: Cannot search a store created with one embedding model using a different model.
3. **Collection names matter**: Using the same collection name reuses existing data. Use unique names or clear between runs.
4. **Metadata is lost on chunking**: `RecursiveCharacterTextSplitter` preserves metadata from the source document on each chunk.
5. **Cost**: Embedding 1000 pages costs money. For learning, use small document sets.
