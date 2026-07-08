from func_agent import FunctionCallingAgent
from llm_sdk import Small_LLM_Model
import json
import model
import functions



def main():
    agent = FunctionCallingAgent()
    agent.register_tools([
        functions.fn_add_numbers,
        functions.fn_subtract_numbers,
        functions.fn_multiply_numbers,
        functions.fn_greet,
        functions.fn_reverse_string,
        functions.get_current_weather,
        functions.fn_divide,
    ])
    results = []
    for prompt_item in agent.prompts:
        results += "".join(agent.run(prompt_item['prompt']))
        print(results)

    output = "[" + "".join(results) + "]"
    print(output)
if __name__ == "__main__":
    main()

