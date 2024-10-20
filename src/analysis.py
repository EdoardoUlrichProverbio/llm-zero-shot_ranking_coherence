import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from typing import Dict
from sklearn.preprocessing import MultiLabelBinarizer
from scipy.stats import chi2_contingency
import os

# Occurrences of genres (from the provided data)
genre_occurrences = {
    "Dramas": 1803,
    "Comedies": 1208,
    "Documentaries": 1028,
    "Children & Family": 782,
    "Action & Adventure": 727,
    "Romantic": 615,
    "Thrillers": 532,
    "Horror": 396,
    "Stand-Up Comedy": 377,
    "Crime": 360,
    "Music & Musicals": 283,
    "Sci-Fi & Fantasy": 257
}

# Set the directory where your CSV files are located
input_dir = "results"
output_dir = "analysis_results"
os.makedirs(output_dir, exist_ok=True)


def _generate_plot(chi_square_df:pd.DataFrame, model_name:str, ranking_window:int):
    plt.figure(figsize=(10, 6))
    sns.heatmap(chi_square_df[['weighted_chi2']].T, annot=True, cmap="Blues", cbar=True)
    
    # Customize the title and labels
    plt.title(f'Chi-Square Test Results by Genre\nModel: {model_name}, Window: {ranking_window}')
    plt.xlabel('Genres')
    plt.ylabel('')  # Remove 'weighted_chi2' label from y-axis
    plt.xticks(rotation=90)
    plt.tight_layout()

    # Save the plot for the current model and ranking window
    output_path = os.path.join(output_dir, f"{model_name}_window_{ranking_window}_heatmap.png")
    plt.savefig(output_path)
    plt.close()  # Close the plot to avoid overlapping in future iterations


def _weighted_chi_square(genre_weights: Dict[str,float], df:pd.DataFrame):
    # One-hot encode the genres
    mlb = MultiLabelBinarizer()
    genres_encoded = pd.DataFrame(mlb.fit_transform(df['batch_genres']), columns=mlb.classes_)
    df_encoded = pd.concat([df, genres_encoded], axis=1)

    # Perform weighted Chi-Square test for each genre
    chi_square_results = {}
    for genre in mlb.classes_:
        # Contingency table between genre and transitivity
        contingency_table = pd.crosstab(df_encoded[genre], df_encoded['transitivity_check'])
        # Perform the Chi-Square test
        chi2, p, _, expected = chi2_contingency(contingency_table)
        # Apply the weight to the observed Chi-Square statistic
        weighted_chi2 = chi2 * genre_weights.get(genre, 1)  # Multiply chi2 by the weight of the genre
        # Store the weighted chi-square result and p-value
        chi_square_results[genre] = {'weighted_chi2': weighted_chi2, 'p_value': p}

    chi_square_df = pd.DataFrame(chi_square_results).T

    return chi_square_df


def perform_result_analysis(genre_occurrences: Dict[str,int]):

    # Step 1: Calculate weights based on genre occurrence (inverse of frequency)
    max_occurrence = max(genre_occurrences.values())  # Normalize against the max
    genre_weights = {genre: max_occurrence / occurrence for genre, occurrence in genre_occurrences.items()}

    # Iterate over all CSV files in the directory
    for filename in os.listdir(input_dir):
        if filename.endswith(".csv"):
            # Extract model_name and ranking_window from filename
            parts = filename.split("_Tresult_window=")
            model_name = parts[0]
            ranking_window = parts[1].replace(".csv", "")

            # Load the CSV file
            file_path = os.path.join(input_dir, filename)
            df = pd.read_csv(file_path)

            chi_square_df =_weighted_chi_square(genre_weights=genre_weights, df=df)

            #Generate heatmap for weighted chi-square statistics
            _generate_plot(chi_square_df=chi_square_df, model_name=model_name, ranking_window=ranking_window)


    print(f"Heatmaps saved in {output_dir}")
