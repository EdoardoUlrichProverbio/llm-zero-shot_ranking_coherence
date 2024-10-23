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

    # Set padding side to 'left' for decoder-only models
    tokenizer.padding_side = 'left'

    # Check if the tokenizer has a pad_token; if not, assign one
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Load the model in half-precision
    model: PreTrainedModel = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16,
        device_map="auto"
    )

#    print(f"Loaded model in half-precision: {model_name}")
#
#    return tokenizer, model
#
#
#def load_model(model_name: str) -> Tuple[PreTrainedTokenizer, PreTrainedModel]:
#    from transformers import AutoTokenizer, AutoConfig, AutoModelForCausalLM
#    from accelerate import init_empty_weights, load_checkpoint_and_dispatch
#    import torch
#
#    # Load the tokenizer
#    tokenizer = AutoTokenizer.from_pretrained(model_name)
#    tokenizer.padding_side = 'left'
#    if tokenizer.pad_token is None:
#        tokenizer.pad_token = tokenizer.eos_token
#
#    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#
#    if device.type == 'cuda':
#        # Use accelerate for efficient loading with CPU offloading
#        with init_empty_weights():
#            config = AutoConfig.from_pretrained(model_name)
#            model = AutoModelForCausalLM.from_config(config)
#
#        model = load_checkpoint_and_dispatch(
#            model,
#            model_name,
#            device_map='auto',
#            no_split_module_classes=['GPTJBlock'],  # Adjust based on model architecture
#            dtype=torch.float16,
#        )
#    else:
#        # Load model in full precision on CPU
#        model = AutoModelForCausalLM.from_pretrained(
#            model_name,
#            low_cpu_mem_usage=True,
#        )
#
#    return tokenizer, model
#
#


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
