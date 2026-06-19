from typing import Any, Iterable, List, Optional, Sequence
from llm_sdk import Small_LLM_Model


class LLMClient:
    def __init__(self, model: Optional[Any] = None):
        self.model = model if model is not None else Small_LLM_Model()

    def encode_ids(self, lst: list) -> List[int]:
        if not isinstance(lst, list):
            raise TypeError("lst must be a list")
        ids = []
        for char in lst:
            ids.append(self.model.encode(char)[0].tolist()[0])
        return ids


    def decode_ids(self, tokens: Sequence[int]) -> str:
        if not hasattr(tokens, "__iter__"):
            raise TypeError("tokens must be iterable of ints")
        tokens_list = [int(x) for x in tokens]
        return self.model.decode(tokens_list)
    
