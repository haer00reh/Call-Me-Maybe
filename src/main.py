from func_agent import FunctionCallingAgent
from llm_sdk import Small_LLM_Model

def main():
    a = FunctionCallingAgent(Small_LLM_Model)
    print(a._format_prompt())

    pass

if __name__ == "__main__":
    main()

