from .llm_sdk import Small_LLM_Model
from . import model
llm_client = model.LLMClient()
tokens = llm_client.encode_ids("gegegeeggg 021 kkam")

print(tokens)
