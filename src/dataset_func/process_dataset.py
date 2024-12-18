# dataset_func/process_dataset.py

import os
import pandas as pd
import numpy as np
import random
from typing import Tuple, List, Dict
from collections import Counter

# Define the category renaming rules based on the provided mapping
rename_mapping = {
    'TV Dramas': 'Dramas',
    'TV Comedies': 'Comedies',
    'Docuseries': 'Documentaries',
    'TV Action & Adventure': 'Action & Adventure',
    'Children & Family Movies': 'Children & Family',
    'Kids\' TV': 'Children & Family',
    'Teen TV Shows': 'Children & Family',
    'Romantic Movies': 'Romantic',
    'Romantic TV Shows': 'Romantic',
    'TV Thrillers': 'Thrillers',
    'Crime TV Shows': 'Crime',
    'Horror Movies': 'Horror',
    'TV Horror': 'Horror',
    'Stand-Up Comedy & Talk Shows': 'Stand-Up Comedy',
    'TV Sci-Fi & Fantasy': 'Sci-Fi & Fantasy'
}

# List of categories to remove
categories_to_remove = [
    "International Movies", "International TV Shows", "Independent Movies", "British TV Shows",
    "Spanish-Language TV Shows", "Sports Movies", "Classic Movies", "Korean TV Shows",
    "TV Mysteries", "LGBTQ Movies", "Science & Nature TV", "Cult Movies", 
    "Faith & Spirituality", "Classic & Cult TV", "TV Shows", "Movies", "Reality TV", "Anime Series",
    "Anime Features"
]

def _tag_removal(df: pd.DataFrame) -> pd.DataFrame:
    """
    Remove unwanted categories and split the 'listed_in' column into separate categories.

    Args:
        df (pd.DataFrame): The input DataFrame.

    Returns:
        pd.DataFrame: Filtered DataFrame with unwanted categories removed and 'listed_in' exploded into rows.
    """
    # Extract the columns: 'title', 'listed_in', 'description'
    df_selected = df[['title', 'listed_in', 'description']]

    # Split the 'listed_in' column values into separate categories and explode them into rows
    df_filtered = df_selected.assign(listed_in=df_selected['listed_in'].str.split(',')).explode('listed_in')

    # Strip whitespace around the category names
    df_filtered['listed_in'] = df_filtered['listed_in'].str.strip()

    # Remove rows where 'listed_in' contains categories from the removal list
    df_filtered = df_filtered[~df_filtered['listed_in'].isin(categories_to_remove)]

    # Drop rows where 'listed_in' is NaN or empty
    df_filtered = df_filtered.dropna(subset=['listed_in'])
    df_filtered = df_filtered[df_filtered['listed_in'] != '']

    return df_filtered

def _tag_remap(df_filtered: pd.DataFrame) -> pd.DataFrame:
    """
    Remap categories based on the provided renaming rules.

    Args:
        df_filtered (pd.DataFrame): The filtered DataFrame.

    Returns:
        pd.DataFrame: DataFrame with categories renamed.
    """
    # Rename categories based on the provided mapping
    df_filtered['listed_in'] = df_filtered['listed_in'].replace(rename_mapping)

    # Drop rows where 'listed_in' is NaN or empty (again after renaming)
    df_filtered = df_filtered.dropna(subset=['listed_in'])
    df_filtered = df_filtered[df_filtered['listed_in'] != '']

    return df_filtered

def _retrieve_dataset(dataset_dir: str) -> str:
    """
    Retrieve the CSV file from the dataset directory.

    Args:
        dataset_dir (str): Path to the dataset directory.

    Returns:
        str: The path to the CSV file.
    """
    # List all files in the dataset directory
    files_in_dir = os.listdir(dataset_dir)
    csv_files = [f for f in files_in_dir if f.endswith('.csv')]
    
    if not csv_files:
        raise FileNotFoundError(f"No CSV file found in directory {dataset_dir}")
    elif len(csv_files) > 1:
        raise ValueError(f"Multiple CSV files found in directory {dataset_dir}. Please specify the file to use.")
    else:
        csv_file = os.path.join(dataset_dir, csv_files[0])
        print(f"Found CSV file: {csv_file}")

    return csv_file

def _balance_categories(df: pd.DataFrame, random_seed: int = 42) -> pd.DataFrame:
    """
    Balance categories by downsampling over-represented categories to the median count.

    Args:
        df (pd.DataFrame): The DataFrame to balance.
        random_seed (int): Seed for random operations to ensure reproducibility.

    Returns:
        pd.DataFrame: The balanced DataFrame.
    """
    np.random.seed(random_seed)
    
    # Get category counts
    category_counts = df['listed_in'].value_counts()
    
    # Find the median count
    median_count = int(category_counts.median()*0.7)
    
    # List to store DataFrames
    balanced_df_list = []
    
    # Process each category
    for category in category_counts.index:
        category_df = df[df['listed_in'] == category]
        current_count = len(category_df)
        
        if current_count > median_count:
            # Downsample to median count
            category_df = category_df.sample(n=median_count, random_state=random_seed)
        
        # Append the (sampled) category DataFrame to the list
        balanced_df_list.append(category_df)
    
    # Concatenate all categories into a single DataFrame
    balanced_df = pd.concat(balanced_df_list).reset_index(drop=True)
    
    return balanced_df



def process_dataset(dataset_dir: str, seed: int) -> Tuple[pd.DataFrame, Dict[str, int]]:
    """
    Retrieve the CSV file from the dataset directory, process it, and return a DataFrame.

    Args:
        dataset_dir (str): Path to the dataset directory.

    Returns:
        pd.DataFrame: Processed DataFrame.
    """
    
    csv_file = _retrieve_dataset(dataset_dir=dataset_dir)
    df = pd.read_csv(csv_file)

    # Step 1: Remove unwanted tags
    df_filtered = _tag_removal(df=df)

    # Step 2: Remap categories
    df_remapped = _tag_remap(df_filtered=df_filtered)

    # Step 3: Balance the categories by dropping some over-represented rows
    df_balanced = _balance_categories(df=df_remapped, random_seed=seed)

    # Count occurrences of the remaining categories (optional)
    genre_occurrencies = df_balanced['listed_in'].value_counts().to_dict()
    print("Category counts after balancing:")
    print(genre_occurrencies)
    
    return df_balanced, genre_occurrencies
