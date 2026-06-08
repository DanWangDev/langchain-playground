"""
Exercise 04 — Output Parsers
============================
Get structured, typed data from LLM outputs.

Concepts introduced:
- StrOutputParser — extract plain string from ChatMessage
- PydanticOutputParser — parse into a Pydantic model via format instructions
- with_structured_output() — modern approach using tool-calling (preferred)
- CommaSeparatedListOutputParser — parse comma-separated values
- Error handling — what happens when parsing fails
"""

from pydantic import BaseModel, Field
from langchain_core.output_parsers import (
    StrOutputParser,
    PydanticOutputParser,
    CommaSeparatedListOutputParser,
)
from langchain_core.prompts import ChatPromptTemplate
from shared.llm import get_llm


# --- Pydantic models for structured output ---

class MovieReview(BaseModel):
    """Structured movie review."""
    title: str = Field(description="The movie title")
    rating: float = Field(description="Rating out of 10")
    summary: str = Field(description="One-sentence summary")
    pros: list[str] = Field(description="List of 2-3 positive points")
    cons: list[str] = Field(description="List of 2-3 negative points")


class Person(BaseModel):
    """Basic person info."""
    name: str = Field(description="Person's full name")
    age: int = Field(description="Person's age in years")


def demo_str_parser():
    """StrOutputParser — simplest: extract string from AIMessage."""
    print("\n--- 1. StrOutputParser ---")

    prompt = ChatPromptTemplate.from_template("Say hello in {count} different languages.")
    llm = get_llm()

    chain = prompt | llm | StrOutputParser()
    result = chain.invoke({"count": "three"})
    print(f"Type: {type(result).__name__}")
    print(f"Result: {result}")


def demo_pydantic_parser():
    """PydanticOutputParser — classic approach using format instructions."""
    print("\n--- 2. PydanticOutputParser (format instructions) ---")

    parser = PydanticOutputParser(pydantic_object=MovieReview)
    llm = get_llm(temperature=0.3)

    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "You are a film critic. Respond with a JSON object that matches "
                "the following schema. Do NOT include any text outside the JSON.\n\n"
                "{format_instructions}",
            ),
            ("human", "Review the fictional movie '{title}'."),
        ]
    )

    chain = prompt | llm | parser

    try:
        result = chain.invoke(
            {
                "title": "The Last Programmer",
                "format_instructions": parser.get_format_instructions(),
            }
        )
        print(f"Type: {type(result).__name__}")
        print(f"Title: {result.title}")
        print(f"Rating: {result.rating}/10")
        print(f"Summary: {result.summary}")
        print(f"Pros: {result.pros}")
        print(f"Cons: {result.cons}")
    except Exception as e:
        print(f"Parse error: {e}")


def demo_structured_output():
    """with_structured_output() — the modern, recommended approach.

    Uses the model's native tool-calling / JSON mode capability.
    Much more reliable than format-instruction-based parsing.
    """
    print("\n--- 3. with_structured_output() (modern, recommended) ---")

    llm = get_llm(temperature=0.3)
    structured_llm = llm.with_structured_output(Person)

    # This can be called directly — no prompt template needed
    result = structured_llm.invoke(
        "My name is Sarah Chen and I am 34 years old."
    )
    print(f"Type: {type(result).__name__}")
    print(f"Name: {result.name}")
    print(f"Age: {result.age}")


def demo_list_parser():
    """CommaSeparatedListOutputParser — get a list from the LLM."""
    print("\n--- 4. CommaSeparatedListOutputParser ---")

    parser = CommaSeparatedListOutputParser()
    llm = get_llm()

    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "List the items as a comma-separated list.\n{format_instructions}",
            ),
            ("human", "Name 5 popular programming languages."),
        ]
    )

    chain = prompt | llm | parser

    result = chain.invoke({"format_instructions": parser.get_format_instructions()})
    print(f"Type: {type(result).__name__}")
    print(f"Result: {result}")


def demo_structured_output_in_chain():
    """Using with_structured_output as part of a larger chain."""
    print("\n--- 5. Structured Output in a Chain ---")

    class SentimentResult(BaseModel):
        sentiment: str = Field(description="'positive', 'negative', or 'neutral'")
        confidence: float = Field(description="Confidence score 0.0 to 1.0")
        key_words: list[str] = Field(description="Key sentiment-bearing words")

    llm = get_llm().with_structured_output(SentimentResult)

    prompt = ChatPromptTemplate.from_template(
        "Analyze the sentiment of this text: {text}"
    )

    chain = prompt | llm

    test_texts = [
        "I absolutely love this new feature, it's amazing!",
        "The service was slow and the staff was rude.",
    ]

    for text in test_texts:
        result = chain.invoke({"text": text})
        print(f"\nText: '{text}'")
        print(f"  Sentiment: {result.sentiment} (confidence: {result.confidence})")
        print(f"  Key words: {result.key_words}")


def main():
    print("=" * 60)
    print("Exercise 04: Output Parsers")
    print("=" * 60)

    demo_str_parser()
    demo_pydantic_parser()
    demo_structured_output()
    demo_list_parser()
    demo_structured_output_in_chain()


if __name__ == "__main__":
    main()
