import torch
from typing import List, Dict, Tuple
from transformers import AutoTokenizer, AutoModelForCausalLM  #, BitsAndBytesConfig
from transformers import PreTrainedTokenizer, PreTrainedModel


def load_model(model_name: str) -> Tuple[PreTrainedTokenizer, PreTrainedModel]:
    """
    Load a model and tokenizer with proper padding token handling, using half-precision.
    Args:
        model_name (str): Name of the model to load.
    Returns:
        tokenizer, model: The tokenizer and half-precision model loaded.
    """
    # Load the tokenizer
    tokenizer: PreTrainedTokenizer = AutoTokenizer.from_pretrained(model_name)

    # Check if the tokenizer has a pad_token; if not, assign one
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Load the model in half-precision
    model: PreTrainedModel = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16,
        device_map="auto"
    )

    print(f"Loaded model in half-precision: {model_name}")

    return tokenizer, model


def load_all_models(model_names: List[str]) -> Tuple[Dict[str, PreTrainedTokenizer], Dict[str, PreTrainedModel]]:
    """
    Load all models specified in the model_names list.

    Args:
        model_names (List[str]): List of model names.

    Returns:
        Tuple[Dict[str, PreTrainedTokenizer], Dict[str, PreTrainedModel]]: Dictionaries mapping model names to tokenizers and models.
    """
    tokenizers: Dict[str, PreTrainedTokenizer] = {}
    models: Dict[str, PreTrainedModel] = {}


    for model_name in model_names:
        tokenizer, model = load_model(model_name)
        tokenizers[model_name] = tokenizer
        models[model_name] = model

    return tokenizers, models
