# Exercise 02: Prompt Templates

## What You'll Learn

- **ChatPromptTemplate** — define a reusable message structure with placeholders
- **SystemMessage** vs **HumanMessage** — role-based message composition
- **MessagesPlaceholder** — a dynamic slot for conversation history
- **Few-shot prompting** — provide examples to guide the model's output format
- `.format_messages()` — render templates into concrete messages ready for the LLM

## Why Prompt Templates Matter

Writing prompts as raw strings doesn't scale. When you have variables (user input, conversation history, context documents), you need a structured way to compose messages. Prompt templates provide:

1. **Reusability** — define once, use with many different inputs
2. **Type safety** — templates validate that required variables are provided
3. **Role awareness** — system, human, and AI messages are handled differently by the model
4. **History integration** — `MessagesPlaceholder` makes conversation memory seamless

## How Prompt Templates Work

```
Template Definition
┌──────────────────────────────────────┐
│ ChatPromptTemplate.from_messages([   │
│   ("system", "You are a {role}."),   │  ← Placeholders: {role}, {question}
│   ("human", "{question}"),           │
│ ])                                   │
└──────────────────────────────────────┘
         │
         │ .format_messages(role="Python expert", question="What is...")
         ▼
┌──────────────────────────────────────┐
│ [SystemMessage("You are a Python     │
│   expert."),                         │  ← Concrete messages
│  HumanMessage("What is...")]         │
└──────────────────────────────────────┘
         │
         │ llm.invoke(messages)
         ▼
     AIMessage (response)
```

### Message Roles

| Role | Class | Purpose |
|------|-------|---------|
| System | `SystemMessage` | Sets the AI's behavior, tone, and constraints |
| Human | `HumanMessage` | The user's input or question |
| AI | `AIMessage` | The model's previous responses (used in history) |
| Tool | `ToolMessage` | Results from tool calls (see Exercise 08) |

### MessagesPlaceholder

```python
MessagesPlaceholder(variable_name="history")
```

This is a slot that expands into zero or more messages at render time. It's essential for:
- **Conversation history**: The full chat log is inserted here
- **Few-shot examples**: Dynamically insert example exchanges
- **Multi-turn context**: Accumulate messages across turns without changing the template

### Few-Shot Prompting

Instead of describing what you want, **show examples**:

```
System: "Classify sentiment as positive, negative, or neutral.

Examples:
Input: I love this! → positive
Input: Terrible. → negative
Input: It's blue. → neutral"

Human: "Input: This is amazing! →"
```

The model learns the pattern from examples and follows it for new inputs. This is more reliable than describing the format in words alone.

## Key Concepts

### Template Syntax

- `{variable_name}` — simple string substitution
- `MessagesPlaceholder(variable_name="...")` — expands to a list of messages
- Tuples `("role", "content")` — shorthand for creating messages

### Composition Patterns

```python
# Basic: system + human
prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a {role}."),
    ("human", "{question}"),
])

# With history slot
prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful assistant."),
    MessagesPlaceholder(variable_name="history"),
    ("human", "{input}"),
])

# Few-shot: examples inline in system prompt
prompt = ChatPromptTemplate.from_messages([
    ("system", "Classify sentiment.\n\nExamples:\n{each_example}\n"),
    ("human", "{text}"),
])
```

## Gotchas

1. **Missing variables**: If a template expects `{name}` but you don't pass it to `.format_messages()`, the call raises a `KeyError`.
2. **MessagesPlaceholder must be a list**: The variable passed for a placeholder must be a list of `BaseMessage` objects, not a string.
3. **Few-shot examples count against context**: Every example consumes tokens. For long prompts, limit to 2-4 high-quality examples.
4. **Template is NOT the prompt**: `ChatPromptTemplate` is a factory — it produces messages only after `.format_messages()`. You can't invoke it directly.
