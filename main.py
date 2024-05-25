import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import ast
import numpy as np

file_path = r"output\\avg_of_rating_per_movieId.csv"
movies_df = pd.read_csv(file_path)
movies_metadata_file_path = r"input\\archive\\movies_metadata.csv"
movies_metadata = pd.read_csv(movies_metadata_file_path,low_memory=False)
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


for i in range(len(movies_metadata["spoken_languages"])):
    if temp_list_of_spoken_languages_per_movie[i] != "[]":
        temp_list_of_spoken_languages_per_movie[i] = ast.literal_eval(
            list_of_spoken_languages_per_movie[i]
        )[0]["iso_639_1"]
    else:
        temp_list_of_spoken_languages_per_movie[i] = None

movies_metadata["spoken_languages"] = temp_list_of_spoken_languages_per_movie

list_of_companies_per_movie = movies_metadata["production_companies"].to_list()
temp_list_of_companies_per_movie = []
temp_list_of_companies_per_movie.extend(list_of_companies_per_movie)

for i in range(len(movies_metadata["production_companies"])):
    if temp_list_of_companies_per_movie[i] != "[]":
        temp_list_of_companies_per_movie[i] = ast.literal_eval(
            list_of_companies_per_movie[i]
        )[0]["id"]
    else:
        temp_list_of_spoken_languages_per_movie[i] = None

movies_metadata["production_companies"] = temp_list_of_companies_per_movie

list_of_production_countries_per_movie = movies_metadata["production_countries"].to_list()
temp_list_of_production_countries_per_movie = []
temp_list_of_production_countries_per_movie.extend(list_of_production_countries_per_movie)

for i in range(len(movies_metadata["production_countries"])):
    if temp_list_of_production_countries_per_movie[i] != "[]":
        temp_list_of_production_countries_per_movie[i] = ast.literal_eval(
            list_of_production_countries_per_movie[i]
        )[0]["iso_3166_1"]
    else:
        temp_list_of_spoken_languages_per_movie[i] = None

movies_metadata["production_countries"] = temp_list_of_production_countries_per_movie

output_data = {
    "movieId": movies_df["movieId"],
    "avg_of_rating": movies_df["avg_of_rating"],
    "adult":movies_metadata["adult"],
    "budget":movies_metadata["budget"],
    "genres":movies_metadata["genres"],
    "original_language":movies_metadata["original_language"],
    "release_date":movies_metadata["release_date"],
    "revenue":movies_metadata["revenue"],
    "spoken_languages":movies_metadata["spoken_languages"],
    "runtime":movies_metadata["runtime"],
    "production_companies":movies_metadata["production_companies"],
    "production_countries":movies_metadata["production_countries"],
}
output_df = pd.DataFrame(output_data)
# omiting the movies with no rating
output_df = output_df[output_df["avg_of_rating"] != -1]
output_file_path = "output/movies_relevant_data.csv"


output_df.to_csv(output_file_path, index=False)
print("done")
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
