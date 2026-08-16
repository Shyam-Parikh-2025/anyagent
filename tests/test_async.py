# Test made with AI to speed up validation, same style as test_agent.py: plain
# asserts + prints, no pytest, fully offline. Async cases are driven with
# asyncio.run() from this file - the library takes no third-party test helper
# for async any more than it does for anything else.
#
# Covers Phase B: generate_async is the one implementation and the synchronous
# API is a wrapper over it, async-native tools are awaited directly, and a sync
# call made from inside somebody else's event loop is recovered loudly instead
# of raising Python's native "cannot be called from a running event loop".
import asyncio
import json
import warnings

from llmadapt import Agent
from llmadapt.core import run_coroutine_blocking


class FakeResponder:
    def __init__(self, responses):
        self.responses = list(responses)
        self.calls = []

    def __call__(self, payload, headers):
        self.calls.append(json.loads(json.dumps(payload)))
        return self.responses.pop(0)


def text(t):
    return {"content": [{"type": "text", "text": t}]}


def tool_use(name, args, call_id="c1"):
    return {"content": [{"type": "tool_use", "id": call_id, "name": name, "input": args}]}


def make_agent():
    return Agent(provider="anthropic", model="claude-x", api_key="test-key")


# --- 1. the sync API still behaves exactly like a sync API ------------------
agent = make_agent()
agent._send_request = FakeResponder([text("plain answer")])
assert agent.chat("hi", spinner=False) == "plain answer"
print("PASS: Agent.chat() still returns a plain string with no event loop in sight")


# --- 2. ...and chat_async() gets the same result on the same inputs ---------
agent = make_agent()
agent._send_request = FakeResponder([text("plain answer")])
assert asyncio.run(agent.chat_async("hi", spinner=False)) == "plain answer"
print("PASS: Agent.chat_async() returns the same answer as the sync twin")


# --- 3. the old hand-written sync transport is gone, not left lying around --
assert not hasattr(Agent, "_execute_with_thread"), "the sync request thread helper should be gone"
assert not hasattr(Agent, "spinner"), "the blocking spinner should be gone"
assert asyncio.iscoroutinefunction(Agent.generate_async)
assert not asyncio.iscoroutinefunction(Agent.generate), "generate() must stay callable from sync code"
print("PASS: the duplicated sync transport is deleted, leaving one request loop")


# --- 4. an async-native tool is awaited directly ----------------------------
ORDER = []


async def slow_async_tool(x: str) -> str:
    """An async-native tool."""
    ORDER.append(("start", x))
    await asyncio.sleep(0.01)
    ORDER.append(("end", x))
    return f"async:{x}"


agent = make_agent()
agent.add_tool(slow_async_tool)
agent._send_request = FakeResponder([
    tool_use("slow_async_tool", {"x": "1"}),
    text("done"),
])
assert asyncio.run(agent.chat_async("go", spinner=False)) == "done"
assert ORDER == [("start", "1"), ("end", "1")], ORDER
tool_msg = [m for m in agent.conversation.history if m.get("role") == "tool"][0]
assert tool_msg["content"] == "async:1", tool_msg
print("PASS: an `async def` tool is awaited directly and its result reaches the model")


# --- 5. an async tool that raises is still a message for the model ----------
async def async_boom() -> str:
    """Always raises."""
    raise ValueError("kaboom")


agent = make_agent()
agent.add_tool(async_boom)
agent._send_request = FakeResponder([tool_use("async_boom", {}), text("recovered")])
assert asyncio.run(agent.chat_async("go", spinner=False)) == "recovered"
tool_msg = [m for m in agent.conversation.history if m.get("role") == "tool"][0]
assert "Tool Execution Failure (ValueError)" in tool_msg["content"], tool_msg
print("PASS: an async tool that raises is converted to a model-visible string, same as a sync one")


# --- 6. a sync tool still runs, off the event loop --------------------------
def sync_tool(x: str) -> str:
    """An ordinary blocking tool."""
    return f"sync:{x}"


agent = make_agent()
agent.add_tool(sync_tool)
agent._send_request = FakeResponder([tool_use("sync_tool", {"x": "7"}), text("ok")])
assert asyncio.run(agent.chat_async("go", spinner=False)) == "ok"
assert [m for m in agent.conversation.history if m.get("role") == "tool"][0]["content"] == "sync:7"
print("PASS: an ordinary sync tool still runs correctly on the async path")


# --- 7. a sync tool that calls back into an Agent does NOT trip the fallback -
# This is the shape of delegate_to_<name>: a plain function that runs a whole
# nested Agent turn. Sync tools are dispatched on a worker thread, which has no
# event loop of its own, so the nested asyncio.run() is legal and quiet. If this
# ever regresses, every single delegation starts emitting a warning.
inner = make_agent()
inner._send_request = FakeResponder([text("inner answer")])


def calls_an_agent(task: str) -> str:
    """Runs a nested agent, the way a delegation tool does."""
    return inner.chat(task, spinner=False)


outer = make_agent()
outer.add_tool(calls_an_agent)
outer._send_request = FakeResponder([
    tool_use("calls_an_agent", {"task": "sub-task"}),
    text("outer answer"),
])
with warnings.catch_warnings(record=True) as caught:
    warnings.simplefilter("always")
    assert asyncio.run(outer.chat_async("go", spinner=False)) == "outer answer"
nested = [w for w in caught if issubclass(w.category, RuntimeWarning)]
assert not nested, f"a nested sync agent call should not warn: {[str(w.message) for w in nested]}"
assert [m for m in outer.conversation.history if m.get("role") == "tool"][0]["content"] == "inner answer"
print("PASS: a sync tool that runs a nested Agent turn works without tripping the loop fallback")


# --- 8. calling the sync API from inside a running loop warns, and works ----
agent = make_agent()
agent._send_request = FakeResponder([text("recovered result")])


async def a_handler_that_forgot_to_await():
    # The realistic mistake: a bare sync call inside an `async def`.
    return agent.chat("hi", spinner=False)


with warnings.catch_warnings(record=True) as caught:
    warnings.simplefilter("always")
    result = asyncio.run(a_handler_that_forgot_to_await())
assert result == "recovered result", result
messages = [str(w.message) for w in caught if issubclass(w.category, RuntimeWarning)]
assert len(messages) == 1, messages
# The warning has to say the thing that is actually true, not just "recovered":
# the caller's loop was blocked anyway, and only their call site can fix that.
assert "running event loop" in messages[0]
assert "blocked" in messages[0]
assert "await" in messages[0]
assert "chat_async" in messages[0] or "run_async" in messages[0]
print("PASS: a sync call from inside a running loop returns the right answer and warns about the block")


# --- 9. the fallback re-raises rather than swallowing -----------------------
class Boom(Exception):
    pass


async def raises():
    raise Boom("from inside the coroutine")


async def outer_ctx():
    return run_coroutine_blocking(raises, what="test call")


with warnings.catch_warnings(record=True):
    warnings.simplefilter("always")
    try:
        asyncio.run(outer_ctx())
        assert False, "the fallback must re-raise the coroutine's exception"
    except Boom as e:
        assert str(e) == "from inside the coroutine"
print("PASS: the nested-loop fallback re-raises the coroutine's exception instead of swallowing it")

print("\nAll checks passed.")
