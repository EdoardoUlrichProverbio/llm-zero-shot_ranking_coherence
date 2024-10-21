# load_model.py

from typing import List, Dict, Tuple, Union
from transformers import AutoTokenizer, AutoModelForCausalLM, PreTrainedTokenizer, PreTrainedModel




def load_model(model_name: str) -> Tuple[PreTrainedTokenizer, PreTrainedModel]:
    """
    Load a model and tokenizer with 8-bit quantization.
    Args:
        model_name (str): Name of the model to load.
    Returns:
        tokenizer, model: The tokenizer and model loaded.
    """

    print(f"Loading model {model_name}...")
    tokenizer: PreTrainedTokenizer = AutoTokenizer.from_pretrained(model_name)
    model: PreTrainedModel = AutoModelForCausalLM.from_pretrained(
        model_name,
        load_in_8bit=True,         # Set quantization to 8 bits
        device_map="auto"          # Automatically map model to available devices
    )
    return tokenizer, model



def load_all_models(model_names: Union[List[str]|str]) -> Tuple[Dict[str, PreTrainedTokenizer], Dict[str, PreTrainedModel]]:
    """
    Load all models specified in the model_names list.

    Args:
        model_names (List[str]): List of model names.

    Returns:
        Tuple[Dict[str, PreTrainedTokenizer], Dict[str, PreTrainedModel]]: Dictionaries mapping model names to tokenizers and models.
    """
    tokenizers: Dict[str, PreTrainedTokenizer] = {}
    models: Dict[str, PreTrainedModel] = {}

    if isinstance(model_names, List[str]):
        for model_name in model_names:
            tokenizer, model = load_model(model_name)
            tokenizers[model_name] = tokenizer
            models[model_name] = model
    else: 
        tokenizer, model = load_model(model_name)
        tokenizers[model_name] = tokenizer
        models[model_name] = model

    print("All models have been loaded with 8-bit quantization.")
    return tokenizers, models
