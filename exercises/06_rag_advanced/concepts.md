# Exercise 06: RAG Advanced / 练习 06：RAG 进阶

## What You'll Learn / 你将学到

- **MultiQueryRetriever** — generate multiple search queries for better recall
- **ContextualCompressionRetriever** — compress/rerank retrieved documents to remove noise
- **LLMChainExtractor** — use an LLM to extract only relevant portions of each document
- **EnsembleRetriever** — combine multiple retrievers (semantic + keyword)
- **BM25Retriever** — keyword-based retrieval for hybrid search

## Why Advanced Retrieval Matters / 为什么高级检索很重要

Naive retrieval (one query → top-K chunks) works for simple questions but fails when:

1. **The query is ambiguous**: "How do I write better code?" — better Python code? Better architecture? Better testing?
2. **The query uses different vocabulary**: User says "AI memory", document says "context persistence"
3. **Long documents are noisy**: A 2000-word document might have one relevant sentence buried in it
4. **Keyword vs meaning mismatch**: "LangChain components" — semantic search might miss exact keyword matches

Each advanced technique addresses one of these failure modes.

## Techniques / 技术详解

### 1. MultiQueryRetriever — Better Recall

```
Original Question: "How do I write better code?"
        │
        ▼
    LLM rephrases into multiple queries:
    ┌─────────────────────────────────────────┐
    │ "Best practices for writing clean code" │
    │ "How to improve code quality"           │
    │ "Techniques for better programming"     │
    └─────────────────────────────────────────┘
        │
        ▼
    Retrieve for each query → Deduplicate → Return unique results
```

**Why it works**: Different phrasings match different chunks. By searching with multiple formulations, you catch more relevant documents that might use different vocabulary.

### 2. ContextualCompressionRetriever — Less Noise

```
Retrieve 3 documents (500 chars each = 1500 total)
        │
        ▼
    LLMChainExtractor: "Extract only parts relevant to the question"
        │
        ▼
    Compressed: 3 documents (80, 45, 120 chars = 245 total)
    Only the relevant sentences remain
```

**Why it works**: LLMs have limited context windows and get confused by irrelevant text. Compression removes noise so the LLM focuses on what matters.

### 3. Hybrid Search (BM25 + Vector) — Best of Both Worlds

```
┌─────────────────┐     ┌─────────────────┐
│  BM25 (Keyword) │     │  Vector (Semantic)│
│                  │     │                   │
│ "LangChain"      │     │ "LLM framework"   │
│ matches exactly  │     │ matches meaning   │
│                  │     │                   │
│ Weight: 40%      │     │ Weight: 60%       │
└────────┬─────────┘     └────────┬──────────┘
         │                        │
         └────────┬───────────────┘
                  ▼
         EnsembleRetriever
         (weighted combination)
```

**Why it works**:
- **BM25** catches exact terms: "RunnableParallel", "Chroma", "DeepSeek"
- **Vector** catches semantic meaning: "running things at once" → RunnableParallel
- Combined, they cover both cases

### 4. Strategy Comparison

| Technique | Problem Solved | Cost |
|-----------|---------------|------|
| Naive | — | 1 LLM call |
| MultiQuery | Ambiguous queries, different vocab | N queries × 1 LLM call |
| Compression | Noisy long documents | 1 LLM call per retrieved doc |
| Hybrid (BM25+Vector) | Exact keyword matching | No extra LLM calls |

## Key Concepts / 核心概念

### When to Use Each Technique

```
Is the query domain-specific with jargon?
  → YES: Add BM25 (hybrid search)
  → NO: Vector-only is fine

Are retrieved documents long and noisy?
  → YES: Add compression
  → NO: Skip compression

Are queries often ambiguous or brief?
  → YES: Add MultiQuery
  → NO: Single query is fine
```

### Ensemble Weights

```python
EnsembleRetriever(
    retrievers=[bm25, semantic],
    weights=[0.4, 0.6],  # 40% keyword, 60% semantic
)
```

Weights control the importance of each retriever. Tune based on your domain:
- **Technical docs**: Higher BM25 weight (keywords are precise)
- **Narrative content**: Higher vector weight (meaning matters more than words)

## Gotchas / 常见陷阱

1. **MultiQuery costs multiply**: 3 query variants = 3× retrieval cost + 1 extra LLM call. Use only when needed.
2. **Compression can over-trim**: If the LLMChainExtractor is too aggressive, it might remove context the QA LLM needs.
3. **BM25 requires the original documents**: Unlike vector search, BM25 works on raw text (no embeddings). Pass the same chunks to `BM25Retriever.from_documents()`.
4. **Ensemble deduplication**: The ensemble combines results from both retrievers. Near-duplicate chunks may appear — the algorithm handles deduplication.
5. **BM25 is language-specific**: It tokenizes based on whitespace by default. Non-English text may need custom tokenization.

---

# 练习 06：RAG 进阶

## 你将学到

- **MultiQueryRetriever** — 生成多个搜索查询以提高召回率
- **ContextualCompressionRetriever** — 压缩/重排检索到的文档以减少噪音
- **LLMChainExtractor** — 使用 LLM 提取文档中相关部分
- **EnsembleRetriever** — 组合多个检索器（语义 + 关键词）
- **BM25Retriever** — 基于关键词的检索，实现混合搜索

## 为什么高级检索很重要

简单检索（一个问题 → Top-K 片段）对简单问题有效，但在以下情况下会失败：
1. 查询模糊不清
2. 查询用的词汇与文档不同
3. 长文档包含大量噪音
4. 关键词匹配与语义匹配之间的差距

## 技术对比

| 技术 | 解决的问题 | 成本 |
|------|-----------|------|
| 基础检索 | — | 1 次 LLM 调用 |
| MultiQuery | 模糊查询、词汇不匹配 | N 个查询 × 1 次 LLM 调用 |
| Compression | 长文档噪音过多 | 每个文档 1 次 LLM 调用 |
| 混合搜索（BM25+向量） | 精确关键词匹配 | 无额外 LLM 调用 |

## 常见陷阱

1. **MultiQuery 成本成倍增加**：3 个查询变体 = 3 倍检索成本 + 1 次额外 LLM 调用。只在必要时使用。
2. **压缩可能过度裁剪**：如果 LLMChainExtractor 过于激进，可能删除 QA LLM 需要的上下文。
3. **BM25 需要原始文档**：与向量搜索不同，BM25 直接处理原始文本（无需嵌入）。将相同的片段传给 `BM25Retriever.from_documents()`。
4. **BM25 与语言相关**：默认基于空格分词。非英文文本可能需要自定义分词。
