import inspect
import json
from typing import Any, Callable, Dict, List
from utils import parse_json
from model import LLMClient


def tool(func: Callable) -> Callable:
    """
    A decorator to register a function as a tool that the FunctionCallingAgent can use.

    It stores the function's signature and docstring as metadata.
    """
    func._is_tool = True
    func._tool_name = func.__name__

    return func


class FunctionCallingAgent:
    """
    the agent that can use an LLM to reason about and call external functions (tools).
    """

    def __init__(self):
        self.llm_client = LLMClient()
        self._tools: Dict[str, Callable] = {}
        self.defines = parse_json("io/input/functions_definition.json")
        self.prompts = parse_json("io/input/function_calling_tests.json")


    def register_tool(self, func: Callable):
        """
        Manually registers a function that has been decorated with @tool.
        """
        if not getattr(func, "_is_tool", False):
            raise TypeError("Function must be decorated with @tool to be registered.")
        self._tools[func._tool_name] = func

    def register_tools(self, tools: List[Callable]):
        """
        Registers a list of tool-decorated functions.
        """
        for t in tools:
            self.register_tool(t)

    def _format_prompt(self) -> str:
        """
        Formats the prompt with a system message, tool definitions, and the user query.
        """
        console_prompt = (
            "You are a helpful assistant with access to the following tools. "
            "To answer the user's question, you can choose to use a tool. "
            "If you decide to use a tool, you must respond with a JSON object containing "
            'the key "tool_call" with an object with "name" and "arguments" keys. '
            "The `arguments` must be a dictionary of parameter names to values.\n\n"
            "If you have the answer, respond with the answer directly."
        )
        prompt = (
            f"{console_prompt}\n\n"
            f"available tools defines:\n{self.defines}\n\n"
            f"User queries:\n" + "\n".join(item['prompt'] for item in self.prompts)
        )
        return prompt
    
    def allowed_tokens(self) -> list:
        f_names = [f["name"] for f in self.defines]
        param_names = set()
        for f in self.defines:
            param_names.update(f.get("parameters", {}).keys())
        char_batch = set("".join(f_names) + "".join(param_names))
        char_batch |= set('{[}]":,. -') | set("0123456789")
        char_batch |= set('function') | set("parameter")

        allowed = self.llm_client.encode_ids(list(char_batch))
        return allowed
    
    def run(self, user_query: str, max_new_tokens=250) -> str:
        pass


