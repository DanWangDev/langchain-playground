# Exercise 04: Output Parsers

## What You'll Learn

- **StrOutputParser** — simplest parser: extract string from AIMessage
- **PydanticOutputParser** — parse LLM output into Pydantic models via format instructions
- **with_structured_output()** — modern approach using native tool-calling (preferred)
- **CommaSeparatedListOutputParser** — parse comma-separated values into a list
- **Error handling** — what happens when parsing fails and how to recover

## Why Output Parsers Matter

LLMs return **text**. Your application needs **structured data**. Output parsers bridge this gap:

```
LLM Output: "{\"name\": \"Alice\", \"age\": 30}"
                │
                ▼
         Output Parser
                │
                ▼
Structured Data: Person(name="Alice", age=30)
```

Without parsers, you'd write fragile regex or `json.loads()` in every application. Parsers provide:
1. **Type safety** — Pydantic models validate types at runtime
2. **Error messages** — clear feedback when the LLM produces malformed output
3. **Retry logic** — some parsers can ask the LLM to fix its output
4. **Schema communication** — `get_format_instructions()` tells the LLM exactly what format to use

## Two Approaches

### 1. PydanticOutputParser (Classic — Format Instructions)

```
Prompt (with format_instructions) → LLM → JSON string → PydanticOutputParser → Pydantic model
```

The parser injects format instructions into the prompt telling the LLM to output JSON matching a schema. The LLM must follow these instructions precisely — if it adds extra text or malforms the JSON, parsing fails.

### 2. with_structured_output() (Modern — Tool-Calling)

```
Prompt → LLM (with structured output mode) → Pydantic model (directly)
```

Uses the model's native **tool-calling** or **JSON mode** capability. The LLM is constrained to produce valid JSON matching the schema. This is much more reliable because:
- The model's token sampling is constrained to valid outputs
- No format instructions needed in the prompt
- No fragile JSON parsing step

**Rule of thumb**: Always prefer `with_structured_output()`. Use `PydanticOutputParser` only for models that don't support tool-calling.

## How with_structured_output() Works

```python
from pydantic import BaseModel, Field

class Person(BaseModel):
    name: str = Field(description="Person's full name")
    age: int = Field(description="Person's age in years")

# The LLM now returns Person objects directly
structured_llm = llm.with_structured_output(Person)
person = structured_llm.invoke("My name is Sarah and I am 34.")
# person = Person(name="Sarah", age=34)
```

### In a Chain

```python
class SentimentResult(BaseModel):
    sentiment: str = Field(description="positive, negative, or neutral")
    confidence: float = Field(description="0.0 to 1.0")

chain = prompt | llm.with_structured_output(SentimentResult)
result = chain.invoke({"text": "I love this!"})
# result.sentiment == "positive", result.confidence == 0.95
```

## Key Concepts

### Pydantic Field Descriptions

```python
class MovieReview(BaseModel):
    rating: float = Field(description="Rating out of 10")  # ← This IS the prompt for the LLM
```

Field descriptions serve double duty: they document your code AND tell the LLM what to output. Write them clearly — the LLM reads them as instructions.

### Parser Chain Pattern

```
Input → PromptTemplate → LLM → OutputParser → Typed Data
```

The parser is always the LAST step in the chain. It transforms the raw LLM output into application-ready data.

### Error Handling

```python
try:
    result = chain.invoke(input)
except Exception as e:
    # Parser failed — log and handle gracefully
    print(f"Parse error: {e}")
```

Common failures:
- LLM wraps JSON in markdown code blocks (```json...```)
- LLM adds explanatory text before/after JSON
- LLM uses single quotes instead of double quotes
- Missing required fields
- Wrong types (string instead of number)

## Gotchas

1. **with_structured_output() requires tool-calling support**: Not all models support it. FakeLLM and older models fall back to JSON mode or fail.
2. **Field descriptions ARE your prompt**: The LLM sees `Field(description="...")` as instructions. Vague descriptions → vague outputs.
3. **Temperature matters**: Use `temperature=0.0` or low values for structured output — you want determinism, not creativity.
4. **Nested models work but add complexity**: Deeply nested Pydantic models can confuse the LLM. Keep schemas flat when possible.
5. **Streaming and structured output don't mix**: `with_structured_output()` returns the complete object, not tokens. You can't stream partial objects.
