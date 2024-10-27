import torch
from typing import List, Dict, Tuple
from transformers import AutoTokenizer, AutoModelForCausalLM  #, BitsAndBytesConfig
from transformers import PreTrainedTokenizer, PreTrainedModel



def load_model(model_name: str):
    from transformers import AutoTokenizer, AutoModelForCausalLM
    import torch

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.padding_side = 'left'
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Define maximum memory per device
    max_memory = {
        0: '14GB',    # Adjust based on your GPU's available memory
        'cpu': '30GB'  # Adjust based on your system's RAM
    }

    # Load the model with specified max memory
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        device_map='auto',
        max_memory=max_memory,
        torch_dtype=torch.float16,
        low_cpu_mem_usage=True,
    )

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
