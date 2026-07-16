from typing import Any, List, Optional, Sequence
from pydantic import BaseModel, ConfigDict, Field
from .llm_sdk import Small_LLM_Model


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

    def encode(self, text: str) -> Any:
        """Encode a full string into a flat list of token IDs."""
        return self.model.encode(text)[0].tolist()

    def encode_ids(self, lst: list) -> Any:
        """Encode a list of characters into a flat list of token IDs."""
        if not isinstance(lst, list):
            raise TypeError("lst must be a list")
        ids = []
        for char in lst:
            ids.extend(self.model.encode(char)[0].tolist())
        return ids

    def decode(self, tokens: Sequence[int]) -> Any:
        """Decode a list of token IDs back to a string."""
        if not hasattr(tokens, "__iter__"):
            raise TypeError("tokens must be iterable of ints")
        return self.model.decode([int(x) for x in tokens])
