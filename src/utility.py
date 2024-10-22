from typing import List, Dict


def convert_to_dict_list(batch_indices: List[List[int]], batch_results: List[List[int]]) -> List[Dict[int, int]]:
    # List to hold the resulting dictionaries
    result_dicts = []
    
    # Iterate through both lists simultaneously
    for indices, results in zip(batch_indices, batch_results):
        # Create a dictionary mapping items from indices to results
        result_dict = {index: result for index, result in zip(indices, results)}
        # Append the resulting dictionary to the list
        result_dicts.append(result_dict)
    
    return result_dicts