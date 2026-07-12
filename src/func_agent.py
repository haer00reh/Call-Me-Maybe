import json
import re
from typing import Dict, List
from .model import LLMClient
from .utils import _normalize_numeric_values, parse_json, FSMState, Phase
import argparse
from pathlib import Path
from typing import Any


class FunctionCallingAgent:
    """Generate JSON function-call responses from prompts."""

    def __init__(self) -> None:
        """Load tools, prompts, and vocabulary metadata."""
        self.llm_client = LLMClient()
        parser = argparse.ArgumentParser()
        parser.add_argument(
            "--functions_definition",
            type=Path,
            default=Path("src/io/input/functions_definition.json")
        )
        parser.add_argument(
            "--input",
            type=Path,
            default=Path("src/io/input/function_calling_tests.json")
        )
        parser.add_argument(
            "--output",
            type=Path,
            default=Path("src/io/output/function_calling_results.json"))
        args = parser.parse_args()
        self.defines = parse_json(args.functions_definition)
        self.prompts = parse_json(args.input)
        self.output = args.output
        self.fsm = FSMState()

        vocab_path = self.llm_client.model.get_path_to_vocab_file()
        with open(vocab_path) as f:
            raw_vocab = json.load(f)
        self._token_text_by_id = {
            int(token_id): token_text
            for token_text, token_id in raw_vocab.items()
        }
        self._all_token_ids = sorted(self._token_text_by_id.keys())

        self._open_brace_tokens = self._encode_tokens("{")
        self._close_brace_tokens = self._encode_tokens("}")
        self._colon_tokens = self._encode_tokens(":")
        self._comma_tokens = self._encode_tokens(",")
        self._quote_tokens = self._encode_tokens('"')
        self._function_key_tokens = self._encode_tokens('"function"')
        self._parameter_key_tokens = self._encode_tokens('"parameter"')

    def _format_prompt(self, user_query: str) -> str:
        """Build the prompt used to guide the language model."""
        tools_desc = "\n".join(
            f'- {f["name"]}: {f["description"]}' for f in self.defines
        )
        return (
            "You are a function-calling assistant.\n"
            "Return only valid JSON using the schema "
            "{\"function\":...,\"parameter\":...}.\n"
            "Choose exactly one function from the tool list.\n"
            "Use the parameter keys from the tool schema.\n"
            "Copy parameter values as concise spans or numerals.\n\n"
            f"Tools:\n{tools_desc}\n\n"
            "Examples:\n"
            'User query: "divide 10 by 3"\n'
            '{"function":"fn_divide","parameter":{"a":10.0,'
            '"b":3.0}}\n\n'
            'User query: "greet \'Alice\'"\n'
            '{"function":"fn_greet","parameter":{"s":"Alice"}}\n\n'
            f"User query: {user_query}\n"
        )

    def _encode_tokens(self, text: str) -> List[int]:
        """Encode text into token ids, handling tokenizer return shapes."""
        encoded = self.llm_client.encode(text)
        if hasattr(encoded, "tolist"):
            token_ids = encoded.tolist()
            if token_ids and isinstance(token_ids[0], list):
                return [int(token_id) for token_id in token_ids[0]]
            return [int(token_id) for token_id in token_ids]
        return [int(token_id) for token_id in encoded]

    def _token_text(self, token_id: int) -> Any:
        """Return the decoded text for a token id."""
        return self._token_text_by_id.get(
            token_id,
            self.llm_client.decode([token_id]),
        )

    def _append_tokens(
        self,
        generated: List[int],
        tokens: List[int],
    ) -> List[int]:
        """Return a new token list with additional tokens appended."""
        return generated + tokens

    def _sample_greedy_token(
        self,
        generated: List[int],
        allowed_ids: List[int],
    ) -> int:
        """Select the highest-scoring token from the allowed ids."""
        if not allowed_ids:
            raise ValueError(
                "No allowed tokens available for greedy decoding."
            )

        logits = self.llm_client.get_logits_from_input_ids(generated)
        best_token = allowed_ids[0]
        best_score = float("-inf")
        for token_id in allowed_ids:
            score = logits[token_id]
            if score > best_score:
                best_score = score
                best_token = token_id
        return best_token

    def _is_identifier_token(self, token_id: int) -> bool:
        """Check whether a token decodes to an identifier-like string."""
        token_text = self._token_text(token_id).strip()
        return bool(token_text) and (
            re.fullmatch(r"[A-Za-z0-9_]+", token_text) is not None
        )

    def _generate_function_name(
        self,
        generated: List[int],
        max_tokens: int = 20,
    ) -> tuple[List[int], str]:
        """Generate the function name portion of the JSON response."""
        token_ids: List[int] = []

        for _ in range(max_tokens):
            allowed_ids = [
                token_id
                for token_id in self._all_token_ids
                if self._is_identifier_token(token_id)
            ] + self._quote_tokens
            next_token = self._sample_greedy_token(generated, allowed_ids)
            token_text = self._token_text(next_token)

            if token_text == '"' and token_ids:
                break

            if token_text == '"' and not token_ids:
                continue

            token_ids.append(next_token)
            generated.append(next_token)

        return generated, self.llm_client.decode(token_ids).strip()

    def _generate_parameter_object(
        self,
        context: List[int],
        max_tokens: int = 50,
    ) -> tuple[List[int], Dict[str, object]]:
        """Generate the parameter object portion of the JSON response."""
        context = self._append_tokens(context, self._open_brace_tokens)
        generated_tokens: List[int] = self._open_brace_tokens.copy()

        brace_depth = 1

        for _ in range(max_tokens):
            allowed_ids = [
                token_id
                for token_id in self._all_token_ids
                if self._token_text(token_id).strip() != ""
            ]
            allowed_ids += self._open_brace_tokens
            allowed_ids += self._close_brace_tokens

            next_token = self._sample_greedy_token(context, allowed_ids)
            token_text = self._token_text(next_token)

            brace_depth += token_text.count("{")
            brace_depth -= token_text.count("}")

            generated_tokens.append(next_token)
            context.append(next_token)

            if brace_depth <= 0:
                break

        raw_text = self.llm_client.decode(generated_tokens).strip()

        if brace_depth < 0:
            excess = abs(brace_depth)
            if raw_text.endswith("}" * excess):
                raw_text = raw_text[:-excess]

        if not raw_text.endswith("}"):
            raw_text = raw_text + "}"

        try:
            parsed = json.loads(raw_text)
            if isinstance(parsed, dict):
                return generated_tokens, parsed
        except json.JSONDecodeError:
            pass

        return generated_tokens, {}

    def single_run(self, user_query: str,
                   max_new_tokens: int = 70) -> Any:
        """Run a single prompt through the function-calling pipeline."""
        self.fsm = FSMState()
        prompt = self._format_prompt(user_query)
        generated = self._encode_tokens(prompt)

        self.fsm.phase = Phase.FUNCTION_NAME
        generated = self._append_tokens(generated, self._open_brace_tokens)
        generated = self._append_tokens(generated, self._function_key_tokens)
        generated = self._append_tokens(generated, self._colon_tokens)
        generated = self._append_tokens(generated, self._quote_tokens)

        generated, fname = self._generate_function_name(
            generated,
            max_tokens=max_new_tokens,
        )
        if not fname:
            return {
                "error": "model failed to select a function"
            }
        func_def = next(
            (f for f in self.defines if f["name"] == fname),
            None,
        )
        if func_def is None:
            return {
                "error": f"unknown function '{fname}'"
            }
        generated = self._append_tokens(generated, self._quote_tokens)
        generated = self._append_tokens(generated, self._comma_tokens)
        generated = self._append_tokens(generated, self._parameter_key_tokens)
        generated = self._append_tokens(generated, self._colon_tokens)

        self.fsm.phase = Phase.PARAMETERS
        _, output_params = self._generate_parameter_object(
            generated,
            max_tokens=max_new_tokens,
        )

        self.fsm.phase = Phase.DONE

        result = {
            "prompt": user_query,
            "function": fname,
            "parameter": output_params,
        }
        return _normalize_numeric_values(result)

    def run(self, max_new_tokens: int = 70) -> None:
        """Execute the agent over all loaded prompts and write the results."""
        all_results = []
        execution_result: dict[str, Any]

        for prompt_data in self.prompts:
            user_query = prompt_data.get("prompt")
            if user_query:
                execution_result = self.single_run(user_query, max_new_tokens)

                if "error" in execution_result:
                    record = {
                        "prompt": user_query,
                        "name": "error",
                        "parameters": {"message": execution_result["error"]},
                    }
                else:
                    record = {
                        "prompt": user_query,
                        "name": execution_result.get("function"),
                        "parameters": execution_result.get("parameter"),
                    }

                all_results.append(record)

        with open(self.output, "w") as f:
            json.dump(all_results, f, indent=4)
