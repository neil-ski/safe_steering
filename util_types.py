from typing import Literal

PoolingStrategy = Literal["mean", "max", "finaltoken"]
DataSplit = Literal["train", "test", "eval"]
ModelName = Literal["HuggingFaceTB/SmolLM3-3B", "mistralai/Mistral-7B-Instruct-v0.3", "meta-llama/Llama-3.2-3B-Instruct", "meta-llama/Llama-3.2-1B-Instruct", "meta-llama/Llama-3.2-1B", "huihui-ai/Llama-3.2-3B-Instruct-abliterated"]