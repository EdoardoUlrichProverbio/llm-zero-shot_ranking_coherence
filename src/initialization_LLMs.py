# load_model.py

from typing import List, Dict, Tuple
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from transformers import PreTrainedTokenizer, PreTrainedModel

def load_model(model_name: str) -> Tuple[PreTrainedTokenizer, PreTrainedModel]:
    """
    Load a model and tokenizer with 8-bit quantization.
    Args:
        model_name (str): Name of the model to load.
    Returns:
        tokenizer, model: The tokenizer and model loaded.
    """
    # Create a quantization config to replace deprecated load_in_8bit
    quantization_config = BitsAndBytesConfig(
        load_in_8bit=True  # or load_in_4bit=True if you want 4-bit quantization
    )

    print(f"Loading model {model_name}...")

    # Load the tokenizer
    tokenizer: PreTrainedTokenizer = AutoTokenizer.from_pretrained(model_name)

    # Load the model using the quantization config
    model: PreTrainedModel = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=quantization_config,  # Pass the quantization config
        device_map="auto"  # Automatically map the model to available devices (GPU/CPU)
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

    print("All models have been loaded with 8-bit quantization.")
    return tokenizers, models
