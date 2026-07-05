from func_agent import FunctionCallingAgent
from llm_sdk import Small_LLM_Model
import json
import model
import functions



def main():
    agent = FunctionCallingAgent()
    agent.register_tools([functions.fn_add_numbers, functions.fn_greet, functions.get_current_weather, functions.fn_divide])
    results = []
    for prompt_item in agent.prompts:
        result = agent.run(prompt_item['prompt'])
        print(f"prompt: {prompt_item}\nanswer: {result}")

    # then you join them with commas yourself
    output = "[" + ",".join(results) + "]"
    print(output)
if __name__ == "__main__":
    main()

