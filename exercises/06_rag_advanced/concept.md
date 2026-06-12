# Exercise 06: RAG Advanced

## What You'll Learn

- **MultiQueryRetriever** — generate multiple search queries for better recall
- **ContextualCompressionRetriever** — compress/rerank retrieved documents to remove noise
- **LLMChainExtractor** — use an LLM to extract only relevant portions of each document
- **EnsembleRetriever** — combine multiple retrievers (semantic + keyword)
- **BM25Retriever** — keyword-based retrieval for hybrid search

## Why Advanced Retrieval Matters

Naive retrieval (one query → top-K chunks) works for simple questions but fails when:

1. **The query is ambiguous**: "How do I write better code?" — better Python code? Better architecture? Better testing?
2. **The query uses different vocabulary**: User says "AI memory", document says "context persistence"
3. **Long documents are noisy**: A 2000-word document might have one relevant sentence buried in it
4. **Keyword vs meaning mismatch**: "LangChain components" — semantic search might miss exact keyword matches

Each advanced technique addresses one of these failure modes.

## Techniques

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

## Key Concepts

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

## Gotchas

1. **MultiQuery costs multiply**: 3 query variants = 3× retrieval cost + 1 extra LLM call. Use only when needed.
2. **Compression can over-trim**: If the LLMChainExtractor is too aggressive, it might remove context the QA LLM needs.
3. **BM25 requires the original documents**: Unlike vector search, BM25 works on raw text (no embeddings). Pass the same chunks to `BM25Retriever.from_documents()`.
4. **Ensemble deduplication**: The ensemble combines results from both retrievers. Near-duplicate chunks may appear — the algorithm handles deduplication.
5. **BM25 is language-specific**: It tokenizes based on whitespace by default. Non-English text may need custom tokenization.
