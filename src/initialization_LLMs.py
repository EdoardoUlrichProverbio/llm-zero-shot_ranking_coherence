import torch
from typing import List, Dict, Tuple
from transformers import AutoTokenizer, AutoModelForCausalLM  #, BitsAndBytesConfig
from transformers import PreTrainedTokenizer, PreTrainedModel

import onnx
import onnxruntime
from onnxruntime.quantization import quantize_dynamic, QuantType
from onnxconverter_common.float16 import convert_float_to_float16


#def load_model(model_name: str) -> Tuple[PreTrainedTokenizer, PreTrainedModel]:
#    """
#    Load a model and tokenizer with proper padding token handling, using half-precision.
#    Args:
#        model_name (str): Name of the model to load.
#    Returns:
#        tokenizer, model: The tokenizer and half-precision model loaded.
#    """
#    # Load the tokenizer
#    tokenizer: PreTrainedTokenizer = AutoTokenizer.from_pretrained(model_name)
#
#    # Set padding side to 'left' for decoder-only models
#    tokenizer.padding_side = 'left'
#
#    # Check if the tokenizer has a pad_token; if not, assign one
#    if tokenizer.pad_token is None:
#        tokenizer.pad_token = tokenizer.eos_token
#
#    # Load the model in half-precision
#    model: PreTrainedModel = AutoModelForCausalLM.from_pretrained(
#        model_name,
#        torch_dtype=torch.float16,
#        device_map="auto"
#    )
#
#    print(f"Loaded model in half-precision: {model_name}")
#
#    return tokenizer, model


def load_model(model_name: str, quantization_type: str = "int8") -> Tuple[PreTrainedTokenizer, str]:
    """
    Load a model and tokenizer, export the model to ONNX format, and apply optional quantization.
    
    Args:
        model_name (str): Name of the model to load.
        quantization_type (str): Type of quantization to apply ('fp16' for float16 or 'int8' for int8 quantization).
        
    Returns:
        tokenizer, onnx_file_path: The tokenizer and the path to the ONNX model.
    """
    # Load the tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    # Set padding side to 'left' for decoder-only models
    tokenizer.padding_side = 'left'
    
    # Check if the tokenizer has a pad_token; if not, assign one
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Load the model in full precision (initial step before ONNX export)
    model = AutoModelForCausalLM.from_pretrained(model_name)

    # Export the model to ONNX
    onnx_file_path = f"{model_name.replace('/', '_')}.onnx"
    dummy_input = tokenizer("Hello", return_tensors="pt").input_ids.to("cuda")

    torch.onnx.export(
        model,
        dummy_input,
        onnx_file_path,
        opset_version=13,
        input_names=['input_ids'],
        output_names=['logits'],
        dynamic_axes={'input_ids': {0: 'batch_size', 1: 'sequence_length'}}
    )
    
    print(f"Model exported to {onnx_file_path}")

    # Apply quantization if specified
    if quantization_type == 'fp16':
        print(f"Applying FP16 quantization to {onnx_file_path}...")
        onnx_model = onnx.load(onnx_file_path)
        model_fp16 = convert_float_to_float16(onnx_model)
        quantized_model_path = onnx_file_path.replace(".onnx", "_fp16.onnx")
        onnx.save(model_fp16, quantized_model_path)
        print(f"Model quantized to FP16 and saved at {quantized_model_path}")
        return tokenizer, quantized_model_path

    elif quantization_type == 'int8':
        print(f"Applying INT8 quantization to {onnx_file_path}...")
        quantized_model_path = onnx_file_path.replace(".onnx", "_int8.onnx")
        quantize_dynamic(
            model_input=onnx_file_path,
            model_output=quantized_model_path,
            weight_type=QuantType.QInt8  # Use INT8 for weights
        )
        print(f"Model quantized to INT8 and saved at {quantized_model_path}")
        return tokenizer, quantized_model_path

    return tokenizer, onnx_file_path



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
