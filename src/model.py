from typing import Any, List, Optional, Sequence
from pydantic import BaseModel, ConfigDict, Field
from llm_sdk import Small_LLM_Model


class LLMClient(BaseModel):
    """Thin wrapper around the small language model client."""

    model_config = ConfigDict(
        arbitrary_types_allowed=True,
        validate_assignment=True,
    )

    model: Any = Field(default_factory=Small_LLM_Model)

    def __init__(self, model: Optional[Any] = None):
        super().__init__(
            model=model if model is not None else Small_LLM_Model()
        )

    def encode(self, text: str) -> List[int]:
        """Encode a full string into a flat list of token IDs."""
        return self.model.encode(text)[0].tolist()

    def encode_ids(self, lst: list) -> List[int]:
        """Encode a list of characters into a flat list of token IDs."""
        if not isinstance(lst, list):
            raise TypeError("lst must be a list")
        ids = []
        for char in lst:
            ids.extend(self.model.encode(char)[0].tolist())
        return ids

    def decode(self, tokens: Sequence[int]) -> str:
        """Decode a list of token IDs back to a string."""
        if not hasattr(tokens, "__iter__"):
            raise TypeError("tokens must be iterable of ints")
        return self.model.decode([int(x) for x in tokens])

    def get_logits_from_input_ids(self, input_ids: List[int]) -> List[float]:
        """Get logits for the next token given a list of token IDs."""
        return self.model.get_logits_from_input_ids(input_ids)
