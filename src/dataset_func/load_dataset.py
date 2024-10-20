# dataset_func/load_dataset.py

import os
import kagglehub

def load_dataset():
    """
    Check if the dataset exists in the 'resources' folder.
    If not, download it using kagglehub.

    Returns:
        str: Path to the dataset files.
    """
    # Define the path to the resources directory and dataset
    resources_dir = os.path.join(os.path.dirname(__file__), '..', 'resources')
    dataset_dir = os.path.join(resources_dir, 'dataset-netflix-shows')

    # Check if the dataset directory exists
    if not os.path.exists(dataset_dir):
        print("Dataset not found. Downloading from Kaggle...")
        # Ensure the resources directory exists
        if not os.path.exists(resources_dir):
            os.makedirs(resources_dir)
        # Download the latest version of the dataset
        path = kagglehub.dataset_download("infamouscoder/dataset-netflix-shows", path=resources_dir)
        print("Dataset downloaded to:", path)
    else:
        print("Dataset already exists at:", dataset_dir)

    return dataset_dir


load_dataset()