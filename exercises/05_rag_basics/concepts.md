# Exercise 05: RAG Basics / 练习 05：RAG 基础

## What You'll Learn / 你将学到

- **Document Loading** — load `.txt` files from a directory with `DirectoryLoader`
- **Text Splitting** — chunk documents into overlapping pieces with `RecursiveCharacterTextSplitter`
- **Embeddings** — convert text to vectors using Qwen embeddings via DashScope
- **Vector Store** — store and search embeddings with `Chroma`
- **Retrieval Chain** — `create_retrieval_chain` wires retriever + QA chain together
- **History-Aware Retrieval** — rephrase follow-up questions using chat history

## Why RAG Matters / 为什么 RAG 很重要

Retrieval-Augmented Generation (RAG) is the most important LLM application pattern. It solves two fundamental problems:

1. **Knowledge cutoff**: LLMs only know what was in their training data. RAG gives them access to your documents.
2. **Hallucination**: LLMs make things up when they don't know. RAG grounds responses in real documents.

```
Without RAG:  User: "What's our return policy?"
              LLM: "I don't have access to your company's policies." ❌

With RAG:     User: "What's our return policy?"
              System: [retrieves policy doc] → "Returns accepted within 30 days with receipt." ✓
```

## The RAG Pipeline / RAG 流水线

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

## History-Aware Retrieval / 历史感知检索

Follow-up questions are often incomplete without context:

```
User: "What is RAG?"              → Full question
User: "How does it work?"         → "It" refers to RAG — needs rephrasing!
```

`create_history_aware_retriever` uses the LLM to rephrase follow-ups before retrieval:

```
"It" → "How does RAG work?"
```

## Key Concepts / 核心概念

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

## Gotchas / 常见陷阱

1. **Chunk overlap must be < chunk_size**: If overlap ≥ chunk_size, you get duplicate (or near-duplicate) chunks.
2. **Embedding dimension mismatch**: Cannot search a store created with one embedding model using a different model.
3. **Collection names matter**: Using the same collection name reuses existing data. Use unique names or clear between runs.
4. **Metadata is lost on chunking**: `RecursiveCharacterTextSplitter` preserves metadata from the source document on each chunk.
5. **Cost**: Embedding 1000 pages costs money. For learning, use small document sets.

---

# 练习 05：RAG 基础

## 你将学到

- **文档加载** — 用 `DirectoryLoader` 从目录加载 `.txt` 文件
- **文本切分** — 用 `RecursiveCharacterTextSplitter` 将文档切成带重叠的片段
- **嵌入** — 通过 DashScope 用 Qwen 嵌入模型将文本转为向量
- **向量存储** — 用 `Chroma` 存储和搜索嵌入向量
- **检索链** — `create_retrieval_chain` 将检索器与问答链连接
- **历史感知检索** — 利用对话历史改写后续问题

## 为什么 RAG 很重要

RAG（检索增强生成）解决了 LLM 的两个根本问题：
1. **知识截止日期** — LLM 只知道训练数据中的内容，RAG 让它们能访问你的文档
2. **幻觉** — LLM 不知道时会编造内容，RAG 将回答建立在真实文档之上

## RAG 流水线

```
加载文档 → 切分文档 → 嵌入片段 → 存储向量 → 检索 Top-K → 增强提示词+文档 → LLM 生成 → 答案
```

## 核心概念

### 片段大小权衡

| 片段大小 | 优点 | 缺点 |
|----------|------|------|
| 小（100-300） | 精确检索，噪音少 | 可能丢失上下文 |
| 中（500-1000） | 良好平衡 | 标准选择 |
| 大（2000+） | 单片段内有完整上下文 | 检索噪音大，Token 消耗多 |

## 常见陷阱

1. **片段重叠必须 < 片段大小**：如果重叠 ≥ 片段大小，会产生重复片段。
2. **嵌入维度必须匹配**：不能用不同的嵌入模型搜索用另一个模型创建的存储。
3. **集合名称很重要**：使用相同的集合名称会复用已有数据。使用唯一名称或在运行之间清理。
4. **成本**：嵌入大量页面需要费用。学习时使用小型文档集。
