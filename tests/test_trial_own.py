import llmadapt
import dotenv

dotenv.load_dotenv()
agent = llmadapt.Agent('gemini', 'gemini-3.5-flash')

while True:
    user_input = input("You: ")
    if user_input.lower() == "quit":
        break
    response = agent.chat(user_input)
    print(f"Agent: {response}")

