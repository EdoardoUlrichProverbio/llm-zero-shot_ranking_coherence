import pandas as pd
import random
from typing import List, Tuple
def prepare_batches(
    df: pd.DataFrame, batch_size: int = 10, seed: int = 42
) -> Tuple[List[List[str]], List[List[Tuple[str, str]]], List[str]]:
    """
    Prepare deterministic batches of data with no repeated title-genre pairs in a single batch.
    One batch will contain only the descriptions, while the corresponding titles and genres are stored separately.
    Additionally, for each batch, one genre is selected randomly from the genres in that batch.

    Args:
        df (pd.DataFrame): The DataFrame containing 'title', 'genre' (renamed 'listed_in'), and 'description'.
        batch_size (int): Number of rows per batch (default is 10).
        seed (int): Seed for deterministic shuffling (default is 42).

    Returns:
        Tuple[List[List[str]], List[List[Tuple[str, str]]], List[str]]:
            A tuple with three elements:
            - A list of batches where each batch contains only the descriptions (List[List[str]]).
            - A corresponding list of batches where each batch contains (title, genre) tuples (List[List[Tuple[str, str]]]).
            - A list of randomly selected genres (one genre per batch) (List[str]).
    """
    # Ensure the operation is deterministic by setting the random seed
    random.seed(seed)
    
    # Convert the dataframe rows to a list of tuples: (title, genre, description)
    data_tuples: List[Tuple[str, str, str]] = list(df.itertuples(index=False, name=None))  # No index, just the raw data as tuples
    
    # Shuffle the data deterministically based on the seed
    random.shuffle(data_tuples)
    
    # Prepare batches
    description_batches: List[List[str]] = []  # Will store only the descriptions
    title_genre_batches: List[List[Tuple[str, str]]] = []  # Will store (title, genre) pairs
    genre_choices: List[str] = []  # Will store the randomly selected genre for each batch
    
    current_desc_batch: List[str] = []
    current_title_genre_batch: List[Tuple[str, str]] = []
    seen_pairs_in_batch: set[Tuple[str, str]] = set()  # To track title-genre pairs in each batch
    
    for row in data_tuples:
        title, genre, description = row
        
        # If the title-genre pair has already been added to the current batch, start a new batch
        if (title, genre) in seen_pairs_in_batch:
            # Finalize the current batch
            description_batches.append(current_desc_batch)
            title_genre_batches.append(current_title_genre_batch)
            
            # Randomly select a genre from the current batch
            random_genre = random.choice([g for _, g in current_title_genre_batch])
            genre_choices.append(random_genre)
            
            # Start new batch
            current_desc_batch = []
            current_title_genre_batch = []
            seen_pairs_in_batch = set()
        
        # Add the current row to the batches
        current_desc_batch.append(description)
        current_title_genre_batch.append((title, genre))
        seen_pairs_in_batch.add((title, genre))
        
        # If the current batch reaches the specified batch size, finalize it and start a new one
        if len(current_desc_batch) == batch_size:
            # Finalize the current batch
            description_batches.append(current_desc_batch)
            title_genre_batches.append(current_title_genre_batch)
            
            # Randomly select a genre from the current batch
            random_genre = random.choice([g for _, g in current_title_genre_batch])
            genre_choices.append(random_genre)
            
            # Start new batch
            current_desc_batch = []
            current_title_genre_batch = []
            seen_pairs_in_batch = set()
    
    # Append any leftover batch (less than batch size)
    if current_desc_batch:
        description_batches.append(current_desc_batch)
        title_genre_batches.append(current_title_genre_batch)
        
        # Randomly select a genre from the final batch
        random_genre = random.choice([g for _, g in current_title_genre_batch])
        genre_choices.append(random_genre)
    
    return description_batches, title_genre_batches, genre_choices