
from transformers import PreTrainedTokenizer, PreTrainedModel
from itertools import combinations
from typing import List, Tuple, Dict
import asyncio
import torch
import csv
import ast
import os


def _save_results(model_name: str, ranking_window: str) -> str:
    # Sanitize the model name to avoid special characters in filenames
    safe_model_name = model_name.replace("/", "_")
    results_dir = "results"
    if not os.path.exists(results_dir):
        try:
            os.makedirs(results_dir)
            print(f"Created results directory: {results_dir}")
        except Exception as e:
            print(f"Error creating results directory: {e}")
            raise

    csv_filename = f"{results_dir}/{safe_model_name}_Tresult_window={ranking_window}.csv"
    
    # Check if file exists, and if not, create it with headers
    if not os.path.exists(csv_filename):
        try:
            with open(csv_filename, mode='w', newline='') as file:
                writer = csv.DictWriter(file, fieldnames=["batch_idx", "batch", "batch_genres", "transitivity_check"])
                writer.writeheader()
            print(f"Created CSV file: {csv_filename}")
        except Exception as e:
            print(f"Error creating CSV file: {csv_filename} - {e}")
            raise
    
    return csv_filename


def _update_results(csv_filename:str, batch:List[str], batch_genres:List[str], batch_idx:int, transitivity_check:int):
        with open(csv_filename, mode='a', newline='') as file:
            writer = csv.DictWriter(file, fieldnames=["batch_idx", "batch", "batch_genres","transitivity_violations"])
            writer.writerow({"batch_idx": batch_idx, "batch": batch, "batch_genres": batch_genres, "transitivity_check": transitivity_check})



def transitivity_check(rankings: List[Dict[int, int]]):
    """
    Perform pairwise comparison of rankings to detect transitivity violations.
    Args:
        rankings (List[Dict[int, int]]): A list of dictionaries, where each dictionary contains 
                                         {index: rank} pairs for a given partial ranking.
    Returns:
        int: The number of transitivity violations found.
    """
    # Dictionary to track the relative order of each pair of indices
    pairwise_order = {}

    for ranking in rankings:
        # Generate all possible pairs (i, j) from the partial ranking
        for (i, rank_i), (j, rank_j) in combinations(ranking.items(), 2):
            # Determine the relative order for this pair
            if rank_i < rank_j:
                pair = (i, j)
                relation = 1  # i is ranked above j
            else:
                pair = (j, i)
                relation = -1  # j is ranked above i

            # If the pair already exists, check for contradictions
            if pair in pairwise_order:
                # Check if the new relationship contradicts previous ones
                if pairwise_order[pair] != relation:
                    print(f"Transitivity violation detected for pair {pair}: conflicting rankings")
                    return 1  # Return early if a contradiction is found
            else:
                # Store the relationship
                pairwise_order[pair] = relation

    print("No transitivity violations found")
    return 0  # No violations detected


def _construct_prompts(
    batch_combinations: List[List[Tuple[int, str]]],
    batch_paragon: str
) -> List[str]:
    """
    Constructs prompts for the given combinations of descriptions.
    Args:
        batch_combinations (List[List[Tuple[int, str]]]): List of combinations, where each combination is a list of (index, description) tuples.
        batch_paragon (str): The genre or paragon to compare against.
    Returns:
        List[str]: A list of constructed prompts.
    """
    prompts = []
    example = """
        "Genre: Mystery\n"
        "Descriptions:\n"
        "1: A detective solving a complex case.\n"
        "2: A romantic tale in a small town.\n"
        "3: An astronaut exploring space.\n"
        "Output: {{1: 1, 2: 3, 3: 2}}"""
    for combination in batch_combinations:
        # Unpack indices and descriptions
        indices, descriptions = zip(*combination)

        # Use a list to accumulate prompt components
        prompt_lines = [
            f"Rank the following descriptions based on their similarity to the genre '{batch_paragon}'. ",
            "Return the result as a single dictionary where the keys are the indices of the descriptions and the values are their ranks (1 being most similar):\n"
        ]
        # Add each description with its index
        prompt_lines.extend(f"{indices[idx] + 1}: {desc}\n" for idx, desc in enumerate(descriptions))
        # Conclude the prompt with the expected format
        prompt_lines.append("\nReturn the rankings in the following format: {index_of_description: rank_of_description}.")
        prompt_lines.append("example: {example}")
        prompt_lines.append("Output:")

        # Join all components into a single string
        prompt = '\n'.join(prompt_lines)
        prompts.append(prompt)
    return prompts



def _process_prompts_in_batches(
    prompts: List[str],
    tokenizer: PreTrainedTokenizer,
    model: PreTrainedModel,
    device: torch.device,
    batch_size: int = 16,
    max_new_tokens: int = 50
) -> List[Dict[int, int]]:
    """
    Processes the prompts in batches and returns the parsed results.
    Args:
        prompts (List[str]): List of prompts to process.
        tokenizer (PreTrainedTokenizer): The tokenizer used to tokenize the prompts.
        model (PreTrainedModel): The model used for inference.
        device (torch.device): The device to run the model on.
        batch_size (int, optional): Number of prompts to process in each batch. Defaults to 16.
        max_new_tokens (int, optional): Maximum number of tokens to generate. Defaults to 50.
    Returns:
        List[Dict[int, int]]: A list of parsed results from the model output.
    """
    results = []
    for i in range(0, len(prompts), batch_size):
        batch_prompts = prompts[i:i+batch_size]

        # Tokenize and process prompts in batch
        inputs = tokenizer(
            batch_prompts,
            return_tensors="pt",
            padding=True,
            truncation=True
        ).to(device)

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False
            )
        batch_outputs = tokenizer.batch_decode(outputs, skip_special_tokens=True)

        # Parse and store results
        for output in batch_outputs:
            try:
                parsed_result = ast.literal_eval(output)
                results.append(parsed_result)
            except Exception as e:
                print(f"Error parsing result: {output}\nException: {e}")

    return results



async def process_model(
    model_name: str,
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizer,
    batches: List[List[str]],
    batches_info: List[List[Tuple[str, str]]],
    batches_paragon: Tuple[str],
    ranking_window: int 
) -> None:
    """
    Asynchronous function to process sentences with a single model and perform transitivity check.

    Args:
        model_name (str): Name of the model.
        model: The model object.
        tokenizer: The tokenizer object.
        batches (Tuple[List[str]]): Tuple of batches of movie descriptions to process.
        batches_paragon (Tuple[str]): Tuple of genres (paragon) to rank each batch against.
        ranking_window (int): Number of descriptions to process at a time (window size).

    Returns:
        None: Results are saved to a CSV file.
    """
    device = next(model.parameters()).device
    n_batches = len(batches)
    results = []

    csv_filename =_save_results(model_name=model_name, ranking_window=ranking_window)

    print(f"Starting processing with model {model_name}...")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    for batch_idx, batch in enumerate(batches):
        print(f"Processing batch: {batch_idx}")
        batch_paragon = batches_paragon[batch_idx]

        # Assign indices to descriptions
        indexed_descriptions = {i: desc for i, desc in enumerate(batch)}

        # Generate all combinations of descriptions of size 'ranking_window'
        ranking_window = 3  # Example value
        all_combinations = list(combinations(indexed_descriptions.items(), ranking_window))

        # Calculate dynamic batch size
        total_combinations = len(all_combinations)
        batch_size = max(int(total_combinations / 2), 1)
        max_batch_size = 40  # Set an upper limit if needed
        batch_size = min(batch_size, max_batch_size)

        # Construct prompts for all combinations
        prompts = _construct_prompts(all_combinations, batch_paragon)

        # Process prompts in batches and get results
        batch_results = _process_prompts_in_batches(
            prompts,
            tokenizer,
            model,
            device,
            batch_size=batch_size,
            max_new_tokens=50
        )

        transitivity = transitivity_check(rankings=batch_results)

        batch_genres = [genre for _, genre in batches_info[batch_idx]] 
        _update_results(csv_filename=csv_filename, batch=batch, batch_genres=batch_genres,
                        batch_idx=batch_idx, transitivity_check=transitivity)

        results.extend({"batch_idx": batch_idx, "batch": batch, "batch_genres": batch_genres, "transitivity_check": transitivity})

    #Progress update
    if (batch_idx) % 10 == 0:
        print(f"{model_name}: Processed {batch_idx} batch out of {n_batches} sentences")

    print(f"Processing completed with model {model_name}.")
    return results
