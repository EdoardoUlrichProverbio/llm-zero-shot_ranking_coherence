import os
from typing import List, Dict
from ctransformers import AutoModelForCausalLM
import torch

def load_model(model_path: str, model_type: str, device: str = "cuda") -> AutoModelForCausalLM:
    """
    Load a quantized model using ctransformers' AutoModelForCausalLM on the specified device.

    Args:
        model_path (str): Path to the quantized model file.
        model_type (str): Type of the model (e.g., 'gptj', 'gpt2', 'opt', 'bloom', 'llama').
        device (str): The device to load the model on. Defaults to "cuda".

    Returns:
        AutoModelForCausalLM: A ctransformers model instance on the specified device.
    """
    if device == "cuda" and not torch.cuda.is_available():
        print("Warning: CUDA is not available. Falling back to CPU.")
        device = "cpu"
    
    llm = AutoModelForCausalLM.from_pretrained(
        model_path,
        model_type=model_type,
        device=device
    )
    return llm

def load_all_models(model_info_list: List[Dict[str, str]]) -> Dict[str, AutoModelForCausalLM]:
    """
    Load all models specified in the model_info_list.

    Args:
        model_info_list (List[Dict[str, str]]): List of dictionaries with 'name', 'path', and 'type' keys.

    Returns:
        Dict[str, AutoModelForCausalLM]: Dictionary mapping model names to AutoModelForCausalLM instances.
    """
    models: Dict[str, AutoModelForCausalLM] = {}

    for model_info in model_info_list:
        model_name = model_info['name']
        model_path = model_info['path']
        model_type = model_info['type']
        print(f"Loading model: {model_name}")
        llm = load_model(model_path, model_type)
        models[model_name] = llm

    return models