import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import ast
import numpy as np

file_path = r"output\\avg_of_rating_per_movieId.csv"
movies_df = pd.read_csv(file_path)
movies_metadata_file_path = r"input\\archive\\movies_metadata.csv"
movies_metadata = pd.read_csv(movies_metadata_file_path)
# Data Cleaning

# Checking for missing values
missing_values = movies_df.isnull().sum()

# Checking for duplicate rows
duplicate_rows = movies_df.duplicated().sum()

# Checking data types
data_types = movies_df.dtypes

# List of numerical variables to plot
numerical_columns = [
    "adult",
    "budget",
    "genres",
    "original_language",
    "release_date",
    "revenue",
    "spoken_languages",
    "runtime",
    "production_companies",
    "production_countries",
]
numerical_values = {column: movies_metadata[column] for column in numerical_columns}
list_of_genres_per_movie = movies_metadata["genres"].to_list()
temp_list_of_genres_per_movie = []
temp_list_of_genres_per_movie.extend(list_of_genres_per_movie)


for i in range(len(movies_metadata["genres"])):
    if temp_list_of_genres_per_movie[i] != "[]":
        temp_list_of_genres_per_movie[i] = ast.literal_eval(
            list_of_genres_per_movie[i]
        )[0]["id"]
    else:
        temp_list_of_genres_per_movie[i] = None

movies_metadata["genres"] = temp_list_of_genres_per_movie

list_of_spoken_languages_per_movie = movies_metadata["spoken_languages"].to_list()
temp_list_of_spoken_languages_per_movie = []
temp_list_of_spoken_languages_per_movie.extend(list_of_spoken_languages_per_movie)
x = temp_list_of_spoken_languages_per_movie[19729]


for i in range(len(movies_metadata["spoken_languages"])):
    if temp_list_of_spoken_languages_per_movie[i] != "[]" and not np.isnan(
        temp_list_of_spoken_languages_per_movie[i]
    ):
        temp_list_of_spoken_languages_per_movie[i] = ast.literal_eval(
            list_of_spoken_languages_per_movie[i]
        )[0]["name"]
    else:
        temp_list_of_spoken_languages_per_movie[i] = None

movies_metadata["spoken_languages"] = temp_list_of_spoken_languages_per_movie

# Plotting histograms for the numerical variables
plt.figure(figsize=(15, 10))
for i, col in enumerate(numerical_columns, 1):
    plt.subplot(3, 4, i)
    sns.histplot(numerical_values[col], bins=20, kde=True)
    plt.title(f"Distribution of {col}")
    plt.xlabel(col)
    plt.ylabel("Frequency")

plt.tight_layout()
plt.show()
