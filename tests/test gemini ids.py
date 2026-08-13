import json
from llmadapt import Agent, Conversation

class FakeResponder:
    def __init__(self, responses):
        self.responses = list(responses)
        self.calls = []
    def __call__(self, payload, headers):
        self.calls.append(json.loads(json.dumps(payload)))
        return self.responses.pop(0)

def get_weather(city: str) -> dict:
    """Look up the weather."""
    return {"city": city, "tempF": 72}

def book_flight(dest: str) -> dict:
    """Book a flight."""
    return {"status": "booked", "dest": dest}


# ---- 14. Gemini WITH ids in functionCall -> matching ids echoed in functionResponse ----
agent = Agent(provider="gemini", api_key="k")
agent.add_tool(get_weather)
agent.add_tool(book_flight)
responder = FakeResponder([
    {"candidates": [{"content": {"parts": [
        {"functionCall": {"name": "get_weather", "args": {"city": "NYC"}, "id": "fc_1"}},
        {"functionCall": {"name": "book_flight", "args": {"dest": "LAX"}, "id": "fc_2"}},
    ]}}]},
    {"candidates": [{"content": {"parts": [{"text": "All done"}]}}]},
])
agent._send_request = responder
result = agent.chat("weather and flight")
assert result == "All done"

second_payload = responder.calls[1]
# Find the functionResponse parts sent back
function_response_ids = []
for msg in second_payload["contents"]:
    for part in msg.get("parts", []):
        if "functionResponse" in part:
            function_response_ids.append(part["functionResponse"].get("id"))
assert function_response_ids == ["fc_1", "fc_2"], function_response_ids
print("PASS 14: Gemini functionCall ids are echoed back correctly in functionResponse")


# ---- 15. Gemini WITHOUT ids (older model behavior) -> no id sent, no crash ----
agent = Agent(provider="gemini", api_key="k")
agent.add_tool(get_weather)
responder = FakeResponder([
    {"candidates": [{"content": {"parts": [
        {"functionCall": {"name": "get_weather", "args": {"city": "NYC"}}}  # no id key at all
    ]}}]},
    {"candidates": [{"content": {"parts": [{"text": "sunny"}]}}]},
])
agent._send_request = responder
result = agent.chat("weather?")
assert result == "sunny"

second_payload = responder.calls[1]
fr_part = None
for msg in second_payload["contents"]:
    for part in msg.get("parts", []):
        if "functionResponse" in part:
            fr_part = part["functionResponse"]
assert fr_part is not None
assert "id" not in fr_part, f"should not fabricate an id when Gemini didn't send one: {fr_part}"
print("PASS 15: Gemini without ids (older models) doesn't fabricate one or crash")


# ---- 16. Single tool call WITH id, two round trips: id still correctly tracked per-turn ----
agent = Agent(provider="gemini", api_key="k")
agent.add_tool(get_weather)
agent.add_tool(book_flight)
responder = FakeResponder([
    {"candidates": [{"content": {"parts": [{"functionCall": {"name": "get_weather", "args": {"city": "NYC"}, "id": "call_A"}}]}}]},
    {"candidates": [{"content": {"parts": [{"text": "Sunny."}]}}]},
    {"candidates": [{"content": {"parts": [{"functionCall": {"name": "book_flight", "args": {"dest": "LAX"}, "id": "call_B"}}]}}]},
    {"candidates": [{"content": {"parts": [{"text": "Booked."}]}}]},
])
agent._send_request = responder
assert agent.chat("weather?") == "Sunny."
assert agent.chat("book it") == "Booked."

final_payload = responder.calls[-1]
ids_seen = []
for msg in final_payload["contents"]:
    for part in msg.get("parts", []):
        if "functionCall" in part:
            ids_seen.append(("call", part["functionCall"].get("id")))
        if "functionResponse" in part:
            ids_seen.append(("response", part["functionResponse"].get("id")))
assert ("call", "call_A") in ids_seen
assert ("response", "call_A") in ids_seen
assert ("call", "call_B") in ids_seen
assert ("response", "call_B") in ids_seen
print("PASS 16: ids stay correctly matched across two sequential Gemini round trips")


# ---- 17. Conversation._export_gemini fallback path also includes ids when present ----
conv = Conversation()
conv.add_user_msg("weather?")
conv.add_model_msg(text=None, tool_calls=[{"id": "xyz", "function": {"name": "get_weather", "arguments": {"city": "NYC"}}}])
conv.add_tool_response("get_weather", '{"city": "NYC", "tempF": 72}', tool_call_id="xyz")
exported = conv.export_for("gemini")
model_msg = next(m for m in exported if m["role"] == "model")
assert model_msg["parts"][0]["functionCall"]["id"] == "xyz"
response_msg = next(m for m in exported if m["role"] == "user" and "functionResponse" in m["parts"][0])
assert response_msg["parts"][0]["functionResponse"]["id"] == "xyz"
print("PASS 17: fallback Conversation export also carries ids when present")

print("\nAll Gemini-id checks passed.")