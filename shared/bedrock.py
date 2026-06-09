"""
Shared AWS Bedrock factory with FakeLLM fallback.

Usage:
    from shared.bedrock import get_bedrock_llm, get_bedrock_embeddings

    llm = get_bedrock_llm()
    embeddings = get_bedrock_embeddings()
"""

import os
import boto3
from botocore.exceptions import ClientError, NoCredentialsError
from shared.llm import FakeLLM
from shared.embeddings import FakeEmbeddings

# Bedrock model IDs (us-east-1)
CLAUDE_HAIKU = "us.anthropic.claude-3-5-haiku-20241022-v1:0"
CLAUDE_SONNET = "us.anthropic.claude-3-5-sonnet-20241022-v2:0"
TITAN_EMBED = "amazon.titan-embed-text-v2:0"


def _has_aws_creds(region: str = "us-east-1") -> bool:
    """Check if AWS credentials are available and Bedrock is accessible."""
    try:
        session = boto3.Session(region_name=region)
        sts = session.client("sts")
        sts.get_caller_identity()
        return True
    except (ClientError, NoCredentialsError, Exception):
        return False


def _get_region() -> str:
    return os.environ.get("AWS_REGION", "us-east-1")


def get_bedrock_llm(
    model: str = CLAUDE_HAIKU,
    temperature: float = 0,
    region: str | None = None,
):
    """ChatBedrock with Claude — or FakeLLM fallback if no AWS access.

    Args:
        model: Bedrock model ID (default: Claude 3.5 Haiku)
        temperature: 0-1
        region: AWS region (default: AWS_REGION env var or us-east-1)
    """
    region = region or _get_region()

    if not _has_aws_creds(region):
        print("[Bedrock] AWS credentials not found — using FakeLLM fallback.")
        return FakeLLM(model=f"bedrock-{model}", temperature=temperature)

    from langchain_aws import ChatBedrock

    return ChatBedrock(
        model_id=model,
        temperature=temperature,
        region_name=region,
    )


def get_bedrock_embeddings(
    model: str = TITAN_EMBED,
    region: str | None = None,
):
    """Bedrock Titan embeddings — or FakeEmbeddings fallback.

    Args:
        model: Embedding model ID (default: Titan Embed v2)
        region: AWS region
    """
    region = region or _get_region()

    if not _has_aws_creds(region):
        print("[Bedrock] AWS credentials not found — using FakeEmbeddings fallback.")
        return FakeEmbeddings()

    from langchain_aws import BedrockEmbeddings

    return BedrockEmbeddings(model_id=model, region_name=region)


def get_caller_identity():
    """Return AWS caller identity info, or None if no access."""
    try:
        session = boto3.Session(region_name=_get_region())
        sts = session.client("sts")
        return sts.get_caller_identity()
    except Exception:
        return None
