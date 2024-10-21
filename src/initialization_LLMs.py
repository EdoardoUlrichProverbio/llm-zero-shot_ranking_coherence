# load_model.py

from typing import List, Dict, Tuple
from transformers import AutoTokenizer, AutoModelForCausalLM#, BitsAndBytesConfig
from transformers import PreTrainedTokenizer, PreTrainedModel

def load_model(model_name: str) -> Tuple[PreTrainedTokenizer, PreTrainedModel]:
    """
    Load a model and tokenizer with proper padding token handling.
    Args:
        model_name (str): Name of the model to load.
    Returns:
        tokenizer, model: The tokenizer and model loaded.
    """
    print(f"Loading model {model_name}...")

    # Load the tokenizer
    tokenizer: PreTrainedTokenizer = AutoTokenizer.from_pretrained(model_name)

    # Check if the tokenizer has a pad_token; if not, assign one
    if tokenizer.pad_token is None:
        print("Tokenizer does not have a pad_token. Setting pad_token to eos_token.")
        tokenizer.pad_token = tokenizer.eos_token  # Option 1: Use eos_token as pad_token
        # Option 2: Add a special padding token (uncomment the next two lines for this approach)
        # tokenizer.add_special_tokens({'pad_token': '[PAD]'})
        # model.resize_token_embeddings(len(tokenizer))  # Resize model embeddings if a new token is added

    # Load the model
    model: PreTrainedModel = AutoModelForCausalLM.from_pretrained(
        model_name,
        device_map="auto"  # Automatically map model to available devices (GPU/CPU)
    )
    
    print(f"Loaded model: {model_name}")
    
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
