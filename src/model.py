from typing import Any, Iterable, List, Optional, Sequence
from .llm_sdk import Small_LLM_Model


class LLMClient:
    def __init__(self, model: Optional[Any] = None):
        self.model = model if model is not None else Small_LLM_Model()

    def encode_ids(self, text: str) -> List[int]:
        if not isinstance(text, str):
            raise TypeError("text must be a str")
        ids = self.model.encode(text)
        try:
            lst = ids.tolist()
        except Exception:
            raise TypeError("model.encode returned an unsupported type")
        if lst and isinstance(lst[0], (list, tuple)):
            lst = lst[0]
        return [int(x) for x in lst]

    def decode_ids(self, tokens: Sequence[int]) -> str:
        if not hasattr(tokens, "__iter__"):
            raise TypeError("tokens must be iterable of ints")
        tokens_list = [int(x) for x in tokens]
        return self.model.decode(tokens_list)
    
