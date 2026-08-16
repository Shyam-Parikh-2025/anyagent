# An interactive REPL scratchpad, not a test - run_all.py skips it by name.
#
# It used to `import dotenv` to pick up GEMINI_API_KEY from the project's .env,
# which was the last third-party import anywhere in this repo. It no longer
# needs one: Agent resolves the key itself, reading .env through llmadapt's own
# loader (see env.py). Nothing here has to be called for that to happen - the
# explicit load_env() below only makes it visible.
import llmadapt

llmadapt.load_env()
agent = llmadapt.Agent('gemini', 'gemini-3.5-flash')
print(f"key: {agent.api_key_source}")

while True:
    user_input = input("You: ")
    if user_input.lower() == "quit":
        break
    response = agent.chat(user_input)
    print(f"Agent: {response}")
