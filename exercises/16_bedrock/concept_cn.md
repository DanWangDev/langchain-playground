
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

