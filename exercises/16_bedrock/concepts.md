# Exercise 16: AWS Bedrock / 练习 16：AWS Bedrock 集成

## What You'll Learn / 你将学到

- **ChatBedrock** — invoke Claude models through AWS Bedrock
- **BedrockEmbeddings** — Titan embeddings for RAG
- **boto3 credential chain** — env vars → ~/.aws/credentials → instance profile
- **STS identity check** — validate AWS access before calling Bedrock
- **Provider comparison** — Bedrock Claude vs DeepSeek vs Qwen
- **Bedrock RAG** — Titan embeddings + ChromaDB
- **Bedrock Agent** — tool-calling agent with Claude on Bedrock

## Why Bedrock Matters / 为什么 Bedrock 很重要

AWS Bedrock is the **enterprise gateway** to foundation models. Unlike direct API providers (DeepSeek, Qwen), Bedrock provides:

1. **Single API, many models** — Claude, Llama, Titan, Mistral — all through one interface
2. **AWS security** — IAM policies, VPC endpoints, CloudTrail auditing
3. **Data residency** — your data never leaves your AWS account
4. **Compliance** — SOC, HIPAA, GDPR — already certified
5. **No separate billing** — charges go to your AWS bill

For enterprises already on AWS, Bedrock is the natural choice — no new vendor relationships, no new security reviews.

## Bedrock Architecture / 架构

```
┌─────────────────────────────────────────────────┐
│                  Your Application                │
│  ┌──────────────────────────────────────────┐   │
│  │           langchain-aws                   │   │
│  │  ChatBedrock  BedrockEmbeddings           │   │
│  └──────────────────┬───────────────────────┘   │
└─────────────────────┼───────────────────────────┘
                      │
              ┌───────▼────────┐
              │    boto3        │
              │  (AWS SDK)      │
              └───────┬────────┘
                      │
              ┌───────▼────────┐
              │  AWS Bedrock    │
              │  ┌────────────┐ │
              │  │   Claude   │ │
              │  │   Titan    │ │
              │  │   Llama    │ │
              │  └────────────┘ │
              └────────────────┘
```

### Credential Resolution

boto3 tries credentials in this order:

```
1. Environment variables (AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY)
2. Shared credentials file (~/.aws/credentials)
3. AWS config file (~/.aws/config)
4. IAM instance profile (EC2, Lambda, ECS)
5. Container credential provider (ECS task role)
6. SSO credential provider (~/.aws/sso/cache)
```

The first valid credential wins. You don't manage this manually — `boto3.Session()` handles it.

### Verifying Access

```python
from shared.bedrock import get_caller_identity

identity = get_caller_identity()
if identity:
    print(f"Authenticated as: {identity['Arn']}")
else:
    print("Not authenticated — using FakeLLM fallback")
```

Always verify access before making Bedrock calls. The STS `get_caller_identity()` call is free and confirms your credentials work.

## Bedrock Model Access / 模型访问

Models must be **explicitly requested** in the AWS Console:

1. Go to AWS Bedrock → Model access
2. Request access to: Claude (Anthropic), Titan (Amazon)
3. Access is granted instantly for most models

Without access: Bedrock returns `AccessDeniedException`. The playground falls back to `FakeLLM`.

## Provider Comparison / 服务商对比

| Feature | Bedrock (Claude) | DeepSeek | Qwen |
|---------|-----------------|----------|------|
| Provider | AWS | DeepSeek | Alibaba |
| Best model | Claude 3.5 Sonnet | DeepSeek-V3 | Qwen-Max |
| Security | IAM, VPC, CloudTrail | API key | API key |
| Data residency | In your AWS account | DeepSeek servers | Alibaba servers |
| Compliance | SOC, HIPAA, GDPR | Limited | Limited |
| Latency | ~1-3s | ~1-2s | ~1-3s |
| Pricing | Per-token (AWS bill) | Per-token (RMB) | Per-token (RMB) |

## Key Concepts / 核心概念

### ChatBedrock vs ChatOpenAI

```python
# DeepSeek/Qwen (OpenAI-compatible)
from langchain_openai import ChatOpenAI
llm = ChatOpenAI(model="deepseek-chat", base_url="https://api.deepseek.com/v1")

# Bedrock Claude
from langchain_aws import ChatBedrock
llm = ChatBedrock(model_id="anthropic.claude-3-haiku-20240307-v1:0")
```

Different classes, same `Runnable` interface. Both work with `.invoke()`, `.stream()`, `|` chains, and `create_agent()`.

### BedrockEmbeddings

```python
from langchain_aws import BedrockEmbeddings

embeddings = BedrockEmbeddings(
    model_id="amazon.titan-embed-text-v2:0",
    region_name="us-east-1",
)
```

Titan embeddings are AWS's native embedding model. Like Qwen embeddings, they produce vector representations of text for semantic search. Titan v2 produces 1024-dimensional vectors.

### boto3 Session Management

```python
import boto3

# Default: uses credential chain
session = boto3.Session()

# Explicit region
session = boto3.Session(region_name="us-east-1")

# Named profile
session = boto3.Session(profile_name="production")
```

## Gotchas / 常见陷阱

1. **Model access is NOT automatic**: You must request access to each model in the AWS Bedrock console. Without it, calls fail with `AccessDeniedException`.
2. **Model IDs are region-specific**: Not all models are available in all regions. Check the AWS Bedrock console for regional availability.
3. **Bedrock does NOT use OPENAI_API_KEY**: It uses AWS credentials (boto3). Setting `OPENAI_API_KEY` has no effect on Bedrock calls.
4. **Titan embeddings need model access too**: Same access request flow as Claude — separate model, separate approval.
5. **boto3 credential chain is automatic but opaque**: If you have multiple AWS profiles, use `AWS_PROFILE` env var to select. Otherwise the default profile is used.
6. **Claude message format differs from OpenAI**: `ChatBedrock` handles the conversion automatically, but raw API calls need Claude-specific message formatting.
7. **Cross-region latency**: Bedrock calls go to the configured region. Choose a region close to your application servers.

---

# 练习 16：AWS Bedrock 集成

## 你将学到

- **ChatBedrock** — 通过 AWS Bedrock 调用 Claude 模型
- **BedrockEmbeddings** — RAG 的 Titan 嵌入模型
- **boto3 凭证链** — 环境变量 → ~/.aws/credentials → 实例配置文件
- **STS 身份检查** — 调用 Bedrock 之前验证 AWS 访问
- **服务商对比** — Bedrock Claude vs DeepSeek vs Qwen
- **Bedrock RAG** — Titan 嵌入 + ChromaDB
- **Bedrock 智能体** — Claude on Bedrock 的工具调用智能体

## 为什么 Bedrock 很重要

AWS Bedrock 是基础模型的**企业网关**。与直接 API 提供商不同，Bedrock 提供：单一 API 访问多个模型、IAM 策略/VPC 端点/CloudTrail 审计的 AWS 安全、数据不出 AWS 账户的数据驻留、SOC/HIPAA/GDPR 等合规认证、统一 AWS 账单。

## 凭证解析

boto3 按以下顺序尝试凭证：环境变量 → ~/.aws/credentials → ~/.aws/config → IAM 实例配置文件 → 容器凭证 → SSO 凭证。第一个有效凭证生效。

## 服务商对比

| 特性 | Bedrock（Claude） | DeepSeek | Qwen |
|------|------------------|----------|------|
| 提供商 | AWS | DeepSeek | 阿里巴巴 |
| 安全 | IAM, VPC, CloudTrail | API 密钥 | API 密钥 |
| 数据驻留 | 在你的 AWS 账户内 | DeepSeek 服务器 | 阿里巴巴服务器 |
| 合规 | SOC, HIPAA, GDPR | 有限 | 有限 |

## 常见陷阱

1. **模型访问不是自动的**：你必须在 AWS Bedrock 控制台中为每个模型申请访问。否则调用会因 `AccessDeniedException` 失败。
2. **模型 ID 是区域特定的**：并非所有模型在所有区域都可用。在 AWS Bedrock 控制台中检查区域可用性。
3. **Bedrock 不使用 OPENAI_API_KEY**：它使用 AWS 凭证（boto3）。设置 `OPENAI_API_KEY` 对 Bedrock 调用无效。
4. **Titan 嵌入也需要模型访问**：与 Claude 相同的访问申请流程——独立的模型，独立的审批。
5. **Claude 消息格式与 OpenAI 不同**：`ChatBedrock` 自动处理转换，但原始 API 调用需要 Claude 特定的消息格式。
