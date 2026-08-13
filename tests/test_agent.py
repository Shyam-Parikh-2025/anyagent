# Test made with AI to speed up validation
import json
from llmadapt import Agent

class FakeResponder:
    def __init__(self, responses):
        self.responses = list(responses)
        self.calls = []
    def __call__(self, payload, headers):
        self.calls.append(json.loads(json.dumps(payload)))
        return self.responses.pop(0)

def get_weather(city: str) -> dict:
    """Look up the weather for a city."""
    return {"city": city, "tempF": 72}

def book_flight(dest: str) -> dict:
    """Book a flight."""
    return {"status": "booked", "dest": dest}

def make_agent():
    return Agent(provider="anthropic", model="claude-x", api_key="test-key")

# Test 1: parallel tool calls -> ONE user message
agent = make_agent()
agent.add_tool(get_weather)
agent.add_tool(book_flight)
responder = FakeResponder([
    {"content": [
        {"type": "tool_use", "id": "c1", "name": "get_weather", "input": {"city": "NYC"}},
        {"type": "tool_use", "id": "c2", "name": "book_flight", "input": {"dest": "LAX"}},
    ]},
    {"content": [{"type": "text", "text": "All done!"}]},
])
agent._send_request = responder
result = agent.chat("weather and flight please")
assert result == "All done!"
second_payload = responder.calls[1]
user_tool_msgs = [m for m in second_payload["messages"]
                  if m["role"] == "user" and isinstance(m["content"], list)
                  and any(b.get("type") == "tool_result" for b in m["content"])]
assert len(user_tool_msgs) == 1, f"FAIL: expected 1 user msg, got {len(user_tool_msgs)}"
assert len(user_tool_msgs[0]["content"]) == 2
print("PASS: parallel tool calls bundle into one user message")

# Test 2: two sequential round trips stay valid (no leftover generic tool_calls, no same-role-twice)
agent = make_agent()
agent.add_tool(get_weather)
agent.add_tool(book_flight)
responder = FakeResponder([
    {"content": [{"type": "tool_use", "id": "c1", "name": "get_weather", "input": {"city": "NYC"}}]},
    {"content": [{"type": "text", "text": "Sunny."}]},
    {"content": [{"type": "tool_use", "id": "c2", "name": "book_flight", "input": {"dest": "LAX"}}]},
    {"content": [{"type": "text", "text": "Booked!"}]},
])
agent._send_request = responder
r1 = agent.chat("weather in NYC?")
r2 = agent.chat("now book LAX")
assert r1 == "Sunny." and r2 == "Booked!"
final_payload = responder.calls[-1]
roles = [m["role"] for m in final_payload["messages"]]
for i in range(len(roles)-1):
    assert roles[i] != roles[i+1], f"FAIL: consecutive same-role messages: {roles}"
for m in final_payload["messages"]:
    assert "tool_calls" not in m, f"FAIL: leaked generic tool_calls: {m}"
print("PASS: two sequential round trips stay valid for Anthropic")

# Test 3: tool output is now valid JSON, not python repr
agent = make_agent()
agent.add_tool(get_weather)
responder = FakeResponder([
    {"content": [{"type": "tool_use", "id": "c1", "name": "get_weather", "input": {"city": "NYC"}}]},
    {"content": [{"type": "text", "text": "ok"}]},
])
agent._send_request = responder
agent.chat("weather?")
tool_msg = next(m for m in responder.calls[1]["messages"]
                 if m["role"] == "user" and isinstance(m["content"], list)
                 and any(b.get("type") == "tool_result" for b in m["content"]))
parsed = json.loads(tool_msg["content"][0]["content"])  # must not raise
assert parsed == {"city": "NYC", "tempF": 72}
print("PASS: tool output is valid JSON (dicts no longer come back as Python repr)")

# Test 3b: plain string tool outputs still pass through untouched
def echo(msg: str) -> str:
    """Echo back the message."""
    return msg
agent = make_agent()
agent.add_tool(echo)
responder = FakeResponder([
    {"content": [{"type": "tool_use", "id": "c1", "name": "echo", "input": {"msg": "hello world"}}]},
    {"content": [{"type": "text", "text": "ok"}]},
])
agent._send_request = responder
agent.chat("echo hello world")
tool_msg = next(m for m in responder.calls[1]["messages"]
                 if m["role"] == "user" and isinstance(m["content"], list)
                 and any(b.get("type") == "tool_result" for b in m["content"]))
assert tool_msg["content"][0]["content"] == "hello world"
print("PASS: plain string tool outputs pass through untouched")

# Test 4: max_tool_iterations default is 10 and raises when exceeded
agent = make_agent()
agent.add_tool(get_weather)
assert agent.max_tool_iterations == 6
infinite = {"content": [{"type": "tool_use", "id": "x", "name": "get_weather", "input": {"city": "NYC"}}]}
responder = FakeResponder([infinite] * 15)
agent._send_request = responder
try:
    agent.chat("loop")
    assert False, "should have raised"
except RuntimeError as e:
    assert "max_tool_iterations" in str(e)
print("PASS: max_tool_iterations defaults to 6 and raises when exceeded")

# Test 5: set_max_tool_iterations lets the user change it easily
agent = make_agent()
agent.add_tool(get_weather)
agent.set_max_tool_iterations(2)
assert agent.max_tool_iterations == 2
responder = FakeResponder([infinite] * 5)
agent._send_request = responder
try:
    agent.chat("loop")
    assert False, "should have raised"
except RuntimeError as e:
    assert "2" in str(e)
print("PASS: set_max_tool_iterations() works")

try:
    agent.set_max_tool_iterations(0)
    assert False
except ValueError:
    print("PASS: set_max_tool_iterations rejects invalid values")

# Test 6: schemas is a dict, re-registering overwrites instead of duplicating
agent = make_agent()
agent.add_tool(get_weather)
agent.add_tool(get_weather)  # register twice
assert isinstance(agent.tool_registry.schemas, dict)
assert len(agent.tool_registry.schemas) == 1, "re-registering should overwrite, not duplicate"
exported = agent.tool_registry.export_for("anthropic")
assert len(exported) == 1
print("PASS: schemas is a dict; duplicate registration overwrites instead of duplicating")

print("\nAll checks passed.")
