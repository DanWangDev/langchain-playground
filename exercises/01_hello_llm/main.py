"""
Exercise 01 — Hello LLM
=======================
Verify the setup works. Call both DeepSeek and Qwen models and compare responses.

Concepts introduced:
- ChatOpenAI (pointed at provider-specific base URLs)
- .invoke() — synchronous call
- AIMessage — the response object
- Provider comparison — different models give different answers
"""

from shared.llm import get_deepseek, get_qwen


def main():
    print("=" * 60)
    print("Exercise 01: Hello LLM — DeepSeek vs Qwen")
    print("=" * 60)

    deepseek_llm = get_deepseek()
    qwen_llm = get_qwen()

    prompt = "In one sentence, what is LangChain?"

    for name, llm in [("DeepSeek", deepseek_llm), ("Qwen", qwen_llm)]:
        print(f"\n--- {name} ---")
        response = llm.invoke(prompt)
        print(f"Response: {response.content}")
        print(f"Model used: {response.response_metadata.get('model_name', 'N/A')}")


if __name__ == "__main__":
    main()
