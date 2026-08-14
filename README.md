# llmadapt

A small, provider-agnostic agent for chatting with LLMs and giving them
tools. Works with Anthropic, OpenAI, Gemini, and Ollama using only the
Python standard library for HTTP — no `requests`, no provider SDKs.

v0.2.0

## Install

```bash
pip install llmadapt
```

From source:

```bash
git clone https://github.com/Shyam-Parikh-2025/llmadapt.git
cd llmadapt
pip install -e .
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

## Compressing tool output

Long tool results (file dumps, logs, search results) eat into the context
window fast, especially for smaller/local models. `CompressionPolicy` lets
an agent opt in to automatic truncation and dedup of oversized tool output
before it's added to conversation history:

```python
from llmadapt import Agent, CompressionPolicy

agent = Agent(
    provider="ollama",
    model="llama3.1:8b",
    compression_policy=CompressionPolicy(enabled=True, max_chars=1500, min_chars_to_bother=300),
)

# or change it later on an existing agent
agent.set_compression_policy(CompressionPolicy(enabled=True, max_tokens=800))
```

Compression is **off by default** — a plain `Agent()` with no policy behaves
exactly as before. When enabled, it:

- collapses repeated consecutive lines (logs/stack traces love to repeat one
  line dozens of times),
- truncates from the middle, snapped to line boundaries, keeping head and
  tail context,
- optionally accepts a `summarizer` callable (`(text, budget) -> str`, e.g. a
  closure around another agent) instead of plain truncation.

`CompressionPolicy` is a thin, stateless config wrapper around
`ContextCompressor`, which is also usable directly for other jobs:

```python
from llmadapt.compressor import ContextCompressor

# strip function/class bodies down to signatures + docstrings, for feeding
# code into a model without spending tokens on implementation details
stub = ContextCompressor.code_to_stub(source_code)

# water-fill one shared character budget across several tool outputs
trimmed = ContextCompressor.compress_batch(outputs, total_budget=4000)
```

## Tracking token usage

`Agent` can estimate how many tokens the current conversation would cost on
the next request, and how many are left before a budget you set is hit. This
uses the same no-external-tokenizer heuristic as `CompressionPolicy`
(`ContextCompressor.token_estimate`), so treat it as an estimate rather than
an exact provider count:

```python
agent = Agent(provider="anthropic", model="claude-3-5-sonnet-20241022",
               max_context_tokens=100_000)

agent.chat("Summarize this document...")

agent.tokens_used()   # -> estimated tokens system instruction + history + tools would cost
agent.tokens_left()   # -> max_context_tokens - tokens_used(), floored at 0
```

`max_context_tokens` is a separate concept from `max_tokens`: `max_tokens` is
the per-request *output* cap sent to the provider (Anthropic's `max_tokens`,
Gemini's `maxOutputTokens`, ...), while `max_context_tokens` is a budget you
track your whole conversation against. If you don't set `max_context_tokens`
(default `None`), `tokens_left()` returns `None` since there's no budget to
count down from. Change it later with `agent.set_max_context_tokens(200_000)`.

## Rank-based routing

`ModelRouter` maps an organizational "rank" (intern up through C-suite) to a
model choice and a matching compression policy, so a multi-agent setup can
pick both in one place instead of hardcoding them per agent:

```python
from llmadapt import Agent, ModelRouter, RoleRank

model = ModelRouter.allocate_model(
    rank=RoleRank.JUNIOR,
    policy={"cost_priority": "high"},
    model_map={"local_default": "ollama/llama3.1:8b", "frontier_default": "claude-sonnet-4-6"},
)
compression = ModelRouter.allocate_compression_policy(RoleRank.JUNIOR)

agent = Agent(provider="ollama", model="llama3.1:8b", compression_policy=compression)
```

`ModelRouter.check_hardware_safety(...)` also guards against routing a rank
to a local model binary too large for the current machine's RAM, falling
back to a configured alternative when it would exceed the allowed threshold.

## Hardware awareness (for local models)

`hardware.py` provides standard-library-only utilities for reasoning about
the machine an agent is running on, mainly aimed at running local model
binaries (e.g. via Ollama) safely alongside everything else on the box:

```python
from llmadapt import HardwareProfiler, ResourceQuota, MoELayerOffloader, LocalModelSingleton

specs = HardwareProfiler.inspect()
# {"os": "Windows", "cpu_cores": 16, "system_ram_gb": 32.0, "gpu_vram_gb": 12.0}

quota = ResourceQuota(mode="auto", ttl_seconds=120)
plan = MoELayerOffloader.calculate_offload(
    total_layers=32, num_experts=8, model_size_gb=14.0, quota=quota,
)
# how many layers fit on GPU vs. spill to system RAM

# Ensures only one local model binary is loaded at a time, and unloads it
# automatically after `quota.ttl_seconds` of inactivity to free VRAM/RAM.
process = LocalModelSingleton.acquire_model("llama3.1:8b", launch_func=my_launcher, quota=quota)
```

## Local models

Pass `is_local=True` when constructing an `Agent` pointed at a local
provider (e.g. Ollama) to flag it as a local-model agent for the rest of
your routing/hardware logic:

```python
agent = Agent(provider="ollama", model="llama3.1:8b", is_local=True)
```

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
python tests/test_full.py
```

Tests run entirely offline against scripted fake HTTP responses — no API key
or network access required.

## License

MIT
