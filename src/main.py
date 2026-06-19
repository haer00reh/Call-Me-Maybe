from func_agent import FunctionCallingAgent
from llm_sdk import Small_LLM_Model
import json
import model




def main():
    a = FunctionCallingAgent()
    a.allowed_tokens()

if __name__ == "__main__":
    main()

