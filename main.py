import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import ast
import numpy as np

country_codes = ['', 'mn', 'nz', 'dk', 'fr', 'mm', 'fy', 'qu', 'be', 'vn', 'me', 'as', 'tg', 'ny', 'au', 'hi', 'zh', 'it', 'wo', 'na', 'sw', 'zu', 'th', 'ps', 'mk', 'tw', 'ki', 'us', 'so', 'bn', 'ka', 'ky', 'bm', 'ms', 'jo', 'rw', 'mq', 'tn', 'pl', 'ja', 'nl', 'tt', 'pa', 'az', 'ln', 'kn', 'et', 'fa', 'sk', 'yi', 'eu', 'ec', 'es', 'ba', 'sg', 'gu', 'cy', 'te', 've', 'iq', 'kh', 'cm', 'ar', 'ua', 'ci', 'gd', 'ly', 'ni', 'ge', 'tr', 'gb', 'hk', 'ws', 'ae', 'gh', 'eg', 'sh', 'gn', 'ne', 'ko', 'ir', 'he', 'af', 'yu', 'ha', 'is', 'sn', 'uy', 'ug', 'da', 'um', 'hu', 'pk', 'bs', 'ml', 'mt', 'li', 'ce', 'my', 'ao', 'en', 'kw', 'tj', 'xg', 'rs', 'ee', 'ro', 'do', 'vi', 'pr', 'km', 'jv', 'il', 'ph', 'ur', 'lu', 'ng', 'cu', 'bg', 'ga', 'pt', 'uk', 'co', 'la', 'tl', 'pe', 'aw', 'in', 'cd', 'ma', 'kr', 'el', 'ab', 'ku', 'lo', 'si', 'eo', 'hy', 'bf', 'lk', 'pg', 'cz', 'sv', 'id', 'kg', 'am', 'xh', 'lt', 'ay', 'de', 'cr', 'tz', 'lv', 'kp', 'dz', 'cl', 'td', 'bo', 'jp', 'bt', 'mi', 'cn', 'nb', 'lb', 'se', 'bw', 'su', 'bi', 'sq', 'fi', 'iu', 'ff', 'cs', 'xx', 'no', 'py', 'at', 'mc', 'sl', 'za', 'lr', 'by', 'al', 'jm', 'mx', 'np', 'br', 'bd', 'gt', 'sa', 'hr', 'ru', 'kk', 'ch', 'sm', 'mr', 'ie', 'xc', 'ta', 'kz', 'ca', 'sr', '[]', 'gr', 'qa', 'gl', 'st', 'uz', 'sy']

file_path = r"output\\avg_of_rating_per_movieId.csv"
movies_df = pd.read_csv(file_path)
movies_metadata_file_path = r"input\\archive\\movies_metadata.csv"
movies_metadata = pd.read_csv(movies_metadata_file_path,low_memory=False)

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
movie_ids = movies_df["movieId"].to_list()
movie_ids2 = movies_metadata["id"].to_list()
#print(movies_metadata.isnull().sum())
#setup wartosci nienumerycznych na liczby

list_of_genres_per_movie = movies_metadata["genres"].to_list()
temp_list_of_genres_per_movie = []
temp_list_of_genres_per_movie.extend(list_of_genres_per_movie)

for i in range(len(movies_metadata["genres"])):
    if temp_list_of_genres_per_movie[i] != "[]":
        temp_list_of_genres_per_movie[i] = ast.literal_eval(
            list_of_genres_per_movie[i]
        )[0]["id"]
    else:
        temp_list_of_genres_per_movie[i] = 0

movies_metadata["genres"] = temp_list_of_genres_per_movie

list_of_spoken_languages_per_movie = movies_metadata["spoken_languages"].to_list()
temp_list_of_spoken_languages_per_movie = []
temp_list_of_spoken_languages_per_movie.extend(list_of_spoken_languages_per_movie)


for i in range(len(movies_metadata["spoken_languages"])):
    if temp_list_of_spoken_languages_per_movie[i] != "[]":
        temp_list_of_spoken_languages_per_movie[i] = country_codes.index((ast.literal_eval(
            list_of_spoken_languages_per_movie[i])[0]["iso_639_1"]).lower())
    else:
        temp_list_of_spoken_languages_per_movie[i] = 0

movies_metadata["spoken_languages"] = temp_list_of_spoken_languages_per_movie

list_of_adult_per_movie = movies_metadata["adult"].to_list()
for i in range(len(movies_metadata["adult"])):
    list_of_adult_per_movie[i]  = int(list_of_adult_per_movie[i])
movies_metadata["adult"] = list_of_adult_per_movie

list_of_production_countries_per_movie = movies_metadata["production_countries"].to_list()
temp_list_of_production_countries_per_movie = []
temp_list_of_production_countries_per_movie.extend(list_of_production_countries_per_movie)

for i in range(len(movies_metadata["production_countries"])):
    if temp_list_of_production_countries_per_movie[i] != "[]" and not temp_list_of_production_countries_per_movie[i] in country_codes:
        temp_list_of_production_countries_per_movie[i] = country_codes.index((ast.literal_eval(
            list_of_production_countries_per_movie[i])[0]["iso_3166_1"]).lower())
    else:
        temp_list_of_production_countries_per_movie[i] = 0
movies_metadata["production_countries"] = temp_list_of_production_countries_per_movie

list_of_release_date_per_movie = movies_metadata["release_date"].to_list()
temp_list_of_release_date_per_movie = []
temp_list_of_release_date_per_movie.extend(list_of_release_date_per_movie)

for i in range(len(movies_metadata["release_date"])):
    if temp_list_of_release_date_per_movie[i] != "" and not pd.isnull(temp_list_of_release_date_per_movie[i]):
        temp_list_of_release_date_per_movie[i] = temp_list_of_release_date_per_movie[i].replace("-","")[:-2]
    else:
        temp_list_of_release_date_per_movie[i] = 190001

movies_metadata["release_date"] = temp_list_of_release_date_per_movie

list_of_original_language_per_movie = movies_metadata["original_language"].to_list()
for i in range(len(movies_metadata["original_language"])):
    if list_of_original_language_per_movie[i] != "[]" and not pd.isnull(list_of_original_language_per_movie[i]):
        list_of_original_language_per_movie[i] = country_codes.index(list_of_original_language_per_movie[i].lower())
    else:
        list_of_original_language_per_movie[i] = 0

movies_metadata["original_language"] = list_of_original_language_per_movie

# koniec setupu

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
    "production_countries":movies_metadata["production_countries"],
    "vote_count":movies_metadata["vote_count"]
}
output_df = pd.DataFrame(output_data)
# omiting the movies with no rating
output_df = output_df[output_df["avg_of_rating"] != -1]
output_file_path = "output/movies_relevant_data.csv"

output_df.to_csv(output_file_path, index=False)
print("done")

