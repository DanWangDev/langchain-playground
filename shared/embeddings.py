"""
Shared embeddings factory with FakeEmbeddings fallback for CI.

Usage:
    from shared.embeddings import get_embeddings
    embeddings = get_embeddings()
"""

import os
from typing import List
from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings
from langchain_core.embeddings import Embeddings

load_dotenv()

DASHSCOPE_BASE_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"


class FakeEmbeddings(Embeddings):
    """Returns zero-vector embeddings — enough for chain structure to work in CI."""

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        return [[0.0] * 1024 for _ in texts]

    def embed_query(self, text: str) -> List[float]:
        return [0.0] * 1024


def get_embeddings(model: str = "text-embedding-v3") -> OpenAIEmbeddings | FakeEmbeddings:
    """Get Qwen embeddings via DashScope, or FakeEmbeddings if no API key."""
    api_key = os.environ.get("DASHSCOPE_API_KEY")
    if not api_key:
        print("[FakeEmbeddings] DASHSCOPE_API_KEY not set — using fake embeddings.")
        return FakeEmbeddings()
    return OpenAIEmbeddings(
        model=model,
        base_url=DASHSCOPE_BASE_URL,
        api_key=api_key,
    )
