"""LLM SDK helpers for local model inference."""

from typing import cast

from huggingface_hub import hf_hub_download
import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    PreTrainedModel,
    PreTrainedTokenizer,
    logging,
)


logging.set_verbosity_error()


class Small_LLM_Model:
    """Wrap a lightweight Hugging Face causal language model."""

    def __init__(
        self,
        model_name: str = "Qwen/Qwen3-0.6B",
        *,
        device: str | None = None,
        dtype: torch.dtype | None = None,
        trust_remote_code: bool = True,
    ) -> None:
        """Load the tokenizer and model on the selected device."""
        self._model_name = model_name

        if device is None:
            if torch.backends.mps.is_available():
                device = "mps"
            elif torch.cuda.is_available():
                device = "cuda"
            else:
                device = "cpu"
        self._device = device

        if dtype is None:
            dtype = (
                torch.float16
                if self._device in ["cuda", "mps"]
                else torch.float32
            )
        self._dtype = dtype

        self._tokenizer: PreTrainedTokenizer = cast(
            PreTrainedTokenizer,
            AutoTokenizer.from_pretrained(
                model_name,
                trust_remote_code=trust_remote_code,
            ),
        )
        if self._tokenizer.pad_token_id is None:
            self._tokenizer.pad_token_id = self._tokenizer.eos_token_id

        self._model: PreTrainedModel = cast(
            PreTrainedModel,
            AutoModelForCausalLM.from_pretrained(
                model_name,
                torch_dtype=self._dtype,
                device_map="auto" if self._device == "cuda" else None,
                trust_remote_code=trust_remote_code,
            ),
        )
        self._model.to(device=self._device)  # type: ignore[call-arg]
        self._model.eval()

        for parameter in self._model.parameters():
            parameter.requires_grad = False

    def encode(self, text: str) -> torch.Tensor:
        """Tokenize text and return a 2-D input_ids tensor.

        The tensor is created on the selected target device.
        """
        ids = self._tokenizer.encode(text, add_special_tokens=False)
        return torch.tensor(
            [ids],
            device=self._device,
            dtype=torch.long,
        )

    def decode(self, ids: torch.Tensor | list[int]) -> str:
        """Decode token ids back into text without special tokens."""
        if isinstance(ids, torch.Tensor):
            ids = ids.tolist()
        return str(self._tokenizer.decode(ids, skip_special_tokens=True))

    def get_logits_from_input_ids(
        self,
        input_ids: list[int],
    ) -> list[float]:
        """Return raw next-token logits for the provided token ids."""
        input_tensor = torch.tensor(
            [input_ids],
            device=self._device,
            dtype=torch.long,
        )
        with torch.no_grad():
            output = self._model(input_ids=input_tensor)
        logits = output.logits[0, -1].tolist()
        return [float(x) for x in logits]

    def get_path_to_vocab_file(self) -> str:
        """Download and return the tokenizer vocab file path."""
        vocab_file_name = self._tokenizer.vocab_files_names.get(
            "vocab_file",
            "vocab.json",
        )
        vocab_path = hf_hub_download(
            repo_id=self._model_name,
            filename=vocab_file_name,
        )
        return vocab_path

    def get_path_to_merges_file(self) -> str:
        """Download and return the tokenizer merges file path."""
        merges_file_name = self._tokenizer.vocab_files_names.get(
            "merges_file",
            "merges.txt",
        )
        merges_path = hf_hub_download(
            repo_id=self._model_name,
            filename=merges_file_name,
        )
        return merges_path

    def get_path_to_tokenizer_file(self) -> str:
        """Download and return the tokenizer JSON file path."""
        tokenizer_file_name = self._tokenizer.vocab_files_names.get(
            "tokenizer_file",
            "tokenizer.json",
        )
        tokenizer_path = hf_hub_download(
            repo_id=self._model_name,
            filename=tokenizer_file_name,
        )
        return tokenizer_path
