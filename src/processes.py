
from transformers import PreTrainedTokenizer, PreTrainedModel
from itertools import combinations
from typing import List, Tuple, Dict
import asyncio
import torch
import csv
import os

def _save_results(model_name: str, ranking_window:str):

    results_dir = "results"
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)
    
    csv_filename = f"{results_dir}/{model_name}_Tresult_window={ranking_window}.csv"
    
    # Check if file exists, and if not, create it with headers
    if not os.path.exists(csv_filename):
        with open(csv_filename, mode='w', newline='') as file:
            writer = csv.DictWriter(file, fieldnames=["batch_idx", "batch", "batch_genres", "transitivity_check"])
            writer.writeheader()
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

    for batch_idx, batch in enumerate(batches):
        batch_results = []
        batch_paragon = batches_paragon[batch_idx]
        transitivity_violations = 0

        # Assign a fixed index to each description in the batch
        indexed_descriptions = {i: desc for i, desc in enumerate(batch)}

        # Create all combinations of indexed descriptions of size ranking_window
        all_combinations = list(combinations(indexed_descriptions.items(), ranking_window))

        for combination in all_combinations:
            # Unpack the indices and descriptions for each combination
            indices, descriptions = zip(*combination)

            # Construct the prompt for the model
            prompt = f"Rank the following descriptions based on their similarity to the genre '{batch_paragon}'. Return the result as a dictionary where the keys are the indices of the descriptions and the values are their ranks (1 being most similar):\n"
            for idx, desc in enumerate(descriptions):
                prompt += f"Description {indices[idx] +1 }: {desc}\n"
            prompt += "\nReturn the rankings in the following format: {index_of_description: rank_of_description}."


            # Tokenize and process the prompt with the model
            inputs = tokenizer([prompt], return_tensors="pt", padding=True, truncation=True).to(device)
            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=50,    # Maximum number of tokens to generate
                    do_sample=False       # Disable sampling for deterministic results
                )

            batch_results = tokenizer.batch_decode(outputs, skip_special_tokens=True)
            batch_results.extend({indices:batch_results})

        transitivity = transitivity_check()

        batch_genres = [genre for _, genre in batches_info[batch_idx]] 
        _update_results(csv_filename=csv_filename, batch=batch, batch_genres=batch_genres,
                        batch_idx=batch_idx, transitivity_check=transitivity)

        results.extend({"batch_idx": batch_idx, "batch": batch, "batch_genres": batch_genres, "transitivity_check": transitivity})

    #Progress update
    if (batch_idx) % 10 == 0:
        print(f"{model_name}: Processed {batch_idx} batch out of {n_batches} sentences")

    print(f"Processing completed with model {model_name}.")
    return results
