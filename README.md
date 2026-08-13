# llmadapt

A small, zero-dependency, provider-agnostic agent for chatting with LLMs and
giving them tools. Works with Anthropic, OpenAI, Gemini, and Ollama using only
the Python standard library for HTTP — no `requests`, no provider SDKs.
v0.1.0

## Install

From source (until published to PyPI):

```bash
git clone https://github.com/Shyam-Parikh-2025/llmadapt.git
cd llmadapt
pip install -e .
```

Once published:

```bash
pip install llmadapt
```

## Quick start

```python
from llmadapt import Agent

agent = Agent(
    provider="anthropic",
    model="claude-sonnet-4-6",
    api_key="sk-ant-...",  # or set ANTHROPIC_API_KEY in your environment
    system_instruction="You are a helpful assistant.",
)

print(agent.chat("What's 12 * 7?"))
```

## Giving the agent tools

```python
def get_weather(city: str) -> dict:
    """Look up the current weather for a city."""
    return {"city": city, "tempF": 72, "condition": "sunny"}

agent.add_tool(get_weather)
print(agent.chat("What's the weather in NYC?"))
```

A JSON schema is auto-generated from the function's type hints, default
values, and docstring. Tool outputs are JSON-serialized automatically
(plain strings pass through untouched).

## Switching providers

```python
agent.switch_api(provider="openai", model="gpt-4o", api_key="sk-...")
```

## Limiting tool-call loops

```python
agent.set_max_tool_iterations(20)  # default is 10
```

If the model gets stuck calling tools repeatedly without producing a final
answer, `chat()` raises a `RuntimeError` instead of looping forever.

## Supported providers

| provider    | env var for API key   | notes                          |
|-------------|------------------------|--------------------------------|
| `anthropic` | `ANTHROPIC_API_KEY`    | Messages API                   |
| `openai`    | `OPENAI_API_KEY`       | Chat Completions API           |
| `gemini`    | `GEMINI_API_KEY`       | generateContent endpoint       |
| `ollama`    | *(none — local)*       | defaults to `localhost:11434`  |
| `custom`    | *(none — pass directly)* | requires `base_url` + a `custom_format_func` you supply to `chat()` |

## Running tests

```bash
python tests/test_agent.py
```

Tests run entirely offline against scripted fake HTTP responses — no API key
or network access required.

## License

MIT
