from typing import get_args

from .types import ModelType

EMBEDDINGS_MODEL_NAMES = [
    "openai:text-embedding-ada-002",
    "azure:text-embedding-ada-002",
    "hf:WhereIsAI/UAE-Large-V1",
    "ollama:nomic-embed-text",
]
EMBEDDINGS_MODEL_NAME_HELP = (
    "Name of embeddings model. Prefix the name with one of the model types ('"
    + "', '".join(get_args(ModelType))
    + "'), otherwise model type will be inferred based on the set environmant "
    "variables. E.g.: '" + "', '".join(EMBEDDINGS_MODEL_NAMES) + "'"
)
LLM_NAMES = ["openai:gpt-3.5-turbo", "openai:gpt-4-turbo", "azure:gpt-4", "hf:google/flan-t5-base", "ollama:llama3"]
LLM_NAME_HELP = (
    "Name of LLM. Prefix the name with one of the model types ('"
    + "', '".join(get_args(ModelType))
    + "'), otherwise model type will be inferred based on the set environmant "
    "variables. E.g.: '" + "', '".join(LLM_NAMES) + "'"
)
