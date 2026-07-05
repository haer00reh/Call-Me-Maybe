import json
from enum import Enum, auto
from typing import Callable, Dict, List
from utils import parse_json
from model import LLMClient


def tool(func: Callable) -> Callable:
    func._is_tool = True
    func._tool_name = func.__name__
    return func


class Phase(Enum):
    START = auto()
    KEY_FUNCTION = auto()
    COLON_1 = auto()
    QUOTE_OPEN_FNAME = auto()
    WRITING_FNAME = auto()
    QUOTE_CLOSE_FNAME = auto()
    COMMA = auto()
    KEY_PARAMETER = auto()
    COLON_2 = auto()
    OPEN_PARAM_OBJ = auto()
    WRITING_PARAM_KEY_OPEN_QUOTE = auto()
    WRITING_PARAM_KEY_CHARS = auto()
    WRITING_PARAM_KEY_CLOSE_QUOTE = auto()
    COLON_PARAM = auto()
    WRITING_PARAM_VALUE_STRING_OPEN = auto()
    WRITING_PARAM_VALUE_STRING_CHARS = auto()
    WRITING_PARAM_VALUE_STRING_CLOSE = auto()
    WRITING_PARAM_VALUE_NUMBER = auto()
    AFTER_PARAM = auto()
    CLOSE_PARAM_OBJ = auto()
    CLOSE_ROOT_OBJ = auto()
    DONE = auto()


class FSMState:
    def __init__(self) -> None:
        self.phase = Phase.START
        self.chosen_function: str | None = None
        self.current_param_index: int = 0
        self.fname_written: str = ""
        self.param_key_written: str = ""
        self.param_value_written: str = ""
        self.key_function_written: str = ""
        self.key_parameter_written: str = ""


class FunctionCallingAgent:
    def __init__(self) -> None:
        self.llm_client = LLMClient()
        self._tools: Dict[str, Callable] = {}
        self.defines = parse_json("io/input/functions_definition.json")
        self.prompts = parse_json("io/input/function_calling_tests.json")
        self.fsm = FSMState()

    def register_tool(self, func: Callable) -> None:
        if not getattr(func, "_is_tool", False):
            raise TypeError("Function must be decorated with @tool to be registered.")
        self._tools[func._tool_name] = func

    def register_tools(self, tools: List[Callable]) -> None:
        for t in tools:
            self.register_tool(t)

    def _format_prompt(self, user_query: str) -> str:
        console_prompt = (
        "Use the correct tool for the user's query. Respond only with JSON.\n\n"
        "Tools:\n"
        + "\n".join(f'- {f["name"]}: {f["description"]}' for f in self.defines)
        + "\n\n"
        "Examples:\n"
        'User query: "divide 10 by 3"\n'
        '{"function":"fn_divide","parameter":{"a":10,"b":3}}\n'
        )
        return (
            f"{console_prompt}\n\n"
            f"User query: {user_query}\n"
        )

    def _ids(self, s: str) -> list:
        return self.llm_client.encode_ids(list(s))

    def _get_allowed_tokens(self) -> list:
        state = self.fsm
        f_names = [f["name"] for f in self.defines]
        LITERAL_FUNCTION = '"function"'
        LITERAL_PARAMETER = '"parameter"'

        if state.phase == Phase.START:
            return self._ids("{")

        if state.phase == Phase.KEY_FUNCTION:
            next_char = LITERAL_FUNCTION[len(state.key_function_written)]
            return self._ids(next_char)

        if state.phase == Phase.COLON_1:
            return self._ids(":")

        if state.phase == Phase.QUOTE_OPEN_FNAME:
            return self._ids('"')

        if state.phase == Phase.WRITING_FNAME:
            candidates = [n for n in f_names if n.startswith(state.fname_written)]
            if state.fname_written in f_names:
                return self._ids('"')
            next_chars = set(
                n[len(state.fname_written)]
                for n in candidates
                if len(n) > len(state.fname_written)
            )
            return self._ids("".join(next_chars))

        if state.phase == Phase.QUOTE_CLOSE_FNAME:
            return self._ids('"')

        if state.phase == Phase.COMMA:
            return self._ids(",")

        if state.phase == Phase.KEY_PARAMETER:
            next_char = LITERAL_PARAMETER[len(state.key_parameter_written)]
            return self._ids(next_char)

        if state.phase == Phase.COLON_2:
            return self._ids(":")

        if state.phase == Phase.OPEN_PARAM_OBJ:
            return self._ids("{")

        if state.phase == Phase.WRITING_PARAM_KEY_OPEN_QUOTE:
            return self._ids('"')

        if state.phase == Phase.WRITING_PARAM_KEY_CHARS:
            func_def = next(f for f in self.defines if f["name"] == state.chosen_function)
            param_names = list(func_def["parameters"].keys())
            current_param = param_names[state.current_param_index]
            written = state.param_key_written
            if written == current_param:
                return self._ids('"')
            return self._ids(current_param[len(written)])

        if state.phase == Phase.WRITING_PARAM_KEY_CLOSE_QUOTE:
            return self._ids('"')

        if state.phase == Phase.COLON_PARAM:
            return self._ids(":")

        if state.phase == Phase.WRITING_PARAM_VALUE_STRING_OPEN:
            return self._ids('"')

        if state.phase == Phase.WRITING_PARAM_VALUE_STRING_CHARS:
            if len(state.param_value_written) >= 15:
                return self._ids('"')
            if len(state.param_value_written) < 2:
                return self._ids("abcdefghijklmnopqrstuvwxyz")
            return self._ids('"abcdefghijklmnopqrstuvwxyz')

        if state.phase == Phase.WRITING_PARAM_VALUE_STRING_CLOSE:
            return self._ids('"')

        if state.phase == Phase.WRITING_PARAM_VALUE_NUMBER:
            func_def = next(f for f in self.defines if f["name"] == state.chosen_function)
            param_names = list(func_def["parameters"].keys())
            total_params = len(param_names)
            if len(state.param_value_written) >= 5:
                if state.current_param_index < total_params - 1:
                    return self._ids(",")
                return self._ids("}")
            if state.current_param_index < total_params - 1:
                return self._ids("0123456789.,")
            return self._ids("0123456789.}")

        if state.phase == Phase.AFTER_PARAM:
            func_def = next(f for f in self.defines if f["name"] == state.chosen_function)
            total_params = len(func_def["parameters"])
            if state.current_param_index < total_params - 1:
                return self._ids(",")
            return self._ids("}")

        if state.phase == Phase.CLOSE_PARAM_OBJ:
            return self._ids("}")

        if state.phase == Phase.CLOSE_ROOT_OBJ:
            return self._ids("}")

        return []

    def _advance_fsm(self, token_str: str) -> None:
        state = self.fsm
        f_names = [f["name"] for f in self.defines]
        LITERAL_FUNCTION = '"function"'
        LITERAL_PARAMETER = '"parameter"'

        if state.phase == Phase.START:
            state.phase = Phase.KEY_FUNCTION
            state.key_function_written = ""

        elif state.phase == Phase.KEY_FUNCTION:
            state.key_function_written += token_str
            if state.key_function_written == LITERAL_FUNCTION:
                state.phase = Phase.COLON_1

        elif state.phase == Phase.COLON_1:
            state.phase = Phase.QUOTE_OPEN_FNAME

        elif state.phase == Phase.QUOTE_OPEN_FNAME:
            state.phase = Phase.WRITING_FNAME
            state.fname_written = ""

        elif state.phase == Phase.WRITING_FNAME:
            if token_str == '"':
                state.phase = Phase.COMMA
            else:
                state.fname_written += token_str
                if state.fname_written in f_names:
                    state.chosen_function = state.fname_written
                    state.phase = Phase.QUOTE_CLOSE_FNAME

        elif state.phase == Phase.QUOTE_CLOSE_FNAME:
            state.phase = Phase.COMMA

        elif state.phase == Phase.COMMA:
            state.phase = Phase.KEY_PARAMETER
            state.key_parameter_written = ""

        elif state.phase == Phase.KEY_PARAMETER:
            state.key_parameter_written += token_str
            if state.key_parameter_written == LITERAL_PARAMETER:
                state.phase = Phase.COLON_2

        elif state.phase == Phase.COLON_2:
            state.phase = Phase.OPEN_PARAM_OBJ

        elif state.phase == Phase.OPEN_PARAM_OBJ:
            state.current_param_index = 0
            state.phase = Phase.WRITING_PARAM_KEY_OPEN_QUOTE

        elif state.phase == Phase.WRITING_PARAM_KEY_OPEN_QUOTE:
            state.param_key_written = ""
            state.phase = Phase.WRITING_PARAM_KEY_CHARS

        elif state.phase == Phase.WRITING_PARAM_KEY_CHARS:
            func_def = next(f for f in self.defines if f["name"] == state.chosen_function)
            param_names = list(func_def["parameters"].keys())
            current_param = param_names[state.current_param_index]
            if token_str == '"':
                state.phase = Phase.COLON_PARAM
            else:
                state.param_key_written += token_str
                if state.param_key_written == current_param:
                    state.phase = Phase.WRITING_PARAM_KEY_CLOSE_QUOTE

        elif state.phase == Phase.WRITING_PARAM_KEY_CLOSE_QUOTE:
            state.phase = Phase.COLON_PARAM

        elif state.phase == Phase.COLON_PARAM:
            func_def = next(f for f in self.defines if f["name"] == state.chosen_function)
            param_names = list(func_def["parameters"].keys())
            current_param = param_names[state.current_param_index]
            param_type = func_def["parameters"][current_param]["type"]
            if param_type == "string":
                state.phase = Phase.WRITING_PARAM_VALUE_STRING_OPEN
            else:
                state.phase = Phase.WRITING_PARAM_VALUE_NUMBER
                state.param_value_written = ""

        elif state.phase == Phase.WRITING_PARAM_VALUE_STRING_OPEN:
            state.phase = Phase.WRITING_PARAM_VALUE_STRING_CHARS
            state.param_value_written = ""

        elif state.phase == Phase.WRITING_PARAM_VALUE_STRING_CHARS:
            if token_str == '"':
                state.phase = Phase.AFTER_PARAM
            else:
                state.param_value_written += token_str

        elif state.phase == Phase.WRITING_PARAM_VALUE_STRING_CLOSE:
            state.phase = Phase.AFTER_PARAM

        elif state.phase == Phase.WRITING_PARAM_VALUE_NUMBER:
            if token_str == "}":
                state.phase = Phase.CLOSE_ROOT_OBJ
            elif token_str == ",":
                state.current_param_index += 1
                state.phase = Phase.WRITING_PARAM_KEY_OPEN_QUOTE
            else:
                state.param_value_written += token_str

        elif state.phase == Phase.AFTER_PARAM:
            func_def = next(f for f in self.defines if f["name"] == state.chosen_function)
            total_params = len(func_def["parameters"])
            if token_str == ",":
                state.current_param_index += 1
                state.phase = Phase.WRITING_PARAM_KEY_OPEN_QUOTE
            elif token_str == "}":
                state.phase = Phase.CLOSE_ROOT_OBJ

        elif state.phase == Phase.CLOSE_PARAM_OBJ:
            state.phase = Phase.CLOSE_ROOT_OBJ

        elif state.phase == Phase.CLOSE_ROOT_OBJ:
            state.phase = Phase.DONE

    def run(self, user_query: str, max_new_tokens: int = 70) -> str:
        self.fsm = FSMState()
        prompt = self._format_prompt(user_query)
        input_ids = self.llm_client.encode(prompt)
        generated = input_ids[:]

        EOS_TOKEN_ID = 151645

        for _ in range(max_new_tokens):
            logits = self.llm_client.get_logits_from_input_ids(generated)
            allowed_ids = self._get_allowed_tokens()

            masked_logits = [float('-inf')] * len(logits)
            for token_id in allowed_ids:
                masked_logits[token_id] = logits[token_id]

            next_token = masked_logits.index(max(masked_logits))
            token_str = self.llm_client.decode([next_token])
            self._advance_fsm(token_str)

            generated.append(next_token)

            if self.fsm.phase == Phase.DONE or next_token == EOS_TOKEN_ID:
                break

        new_tokens = generated[len(input_ids):]
        raw_output = self.llm_client.decode(new_tokens)

        try:
            json.loads(raw_output)
            return raw_output
        except (json.JSONDecodeError, KeyError) as e:
            return f"Error parsing model output: {e}\nRaw output: {raw_output}"