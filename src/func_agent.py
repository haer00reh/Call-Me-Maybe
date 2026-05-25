import inspect
import json
from typing import Any, Callable, Dict, List

from .model import LLMClient


def tool(func: Callable) -> Callable:
    """
    A decorator to register a function as a tool that the FunctionCallingAgent can use.

    It stores the function's signature and docstring as metadata.
    """
    func._is_tool = True
    func._tool_name = func.__name__
    sig = inspect.signature(func)
    docstring = inspect.getdoc(func) or "No description available."

    func._tool_metadata = {
        "name": func._tool_name,
        "description": docstring.strip(),
        "parameters": {
            "type": "object",
            "properties": {
                name: {"type": "string"}
                for name, param in sig.parameters.items()
            },
            "required": [
                name for name, param in sig.parameters.items()
                if param.default == inspect.Parameter.empty
            ],
        },
    }
    return func


class FunctionCallingAgent:
    """
    the agent that can use an LLM to reason about and call external functions (tools).
    """

    def __init__(self, llm_client: LLMClient):
        self.llm_client = llm_client
        self._tools: Dict[str, Callable] = {}

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

    def _format_prompt(self, user_query: str) -> str:
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

        tool_definitions = [
            tool._tool_metadata for tool in self._tools.values()
        ]

        prompt = (
            f"{console_prompt}\n\n"
            f"Available Tools:\n{json.dumps(tool_definitions, indent=2)}\n\n"
            f"User Query: {user_query}"
        )
        return prompt

    def run(self, user_query: str) -> str:
        pass

