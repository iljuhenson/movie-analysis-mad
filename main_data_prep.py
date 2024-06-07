import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import ast
import numpy as np
from constants import COUNTRY_CODES


def process_genres(movies_metadata):
    temp_list_of_genres_per_movie = []
    temp_list_of_genres_per_movie.extend(movies_metadata["genres"].to_list())

    for i in range(len(movies_metadata["genres"])):
        if temp_list_of_genres_per_movie[i] != "[]":
            temp_list_of_genres_per_movie[i] = ast.literal_eval(
                movies_metadata["genres"].to_list()[i]
            )[0]["id"]
        else:
            temp_list_of_genres_per_movie[i] = 0

    movies_metadata["genres"] = temp_list_of_genres_per_movie
    return movies_metadata


def process_spoken_languages(movies_metadata):
    list_of_spoken_languages_per_movie = movies_metadata["spoken_languages"].to_list()
    temp_list_of_spoken_languages_per_movie = []
    temp_list_of_spoken_languages_per_movie.extend(list_of_spoken_languages_per_movie)
    amount_of_spoken_languages_per_movie = [0] * len(
        movies_metadata["spoken_languages"]
    )
    for i in range(len(movies_metadata["spoken_languages"])):
        if temp_list_of_spoken_languages_per_movie[i] != "[]":
            # temp_list_of_spoken_languages_per_movie[i] = COUNTRY_CODES.index(
            #     (
            #         ast.literal_eval(list_of_spoken_languages_per_movie[i])[0][
            #             "iso_639_1"
            #         ]
            #     ).lower()
            # )

            amount_of_spoken_languages_per_movie[i] = len(
                ast.literal_eval(list_of_spoken_languages_per_movie[i])
            )

        else:
            # temp_list_of_spoken_languages_per_movie[i] = 0
            amount_of_spoken_languages_per_movie[i] = 0

    movies_metadata["spoken_languages"] = amount_of_spoken_languages_per_movie
    return movies_metadata


def _process_adult(movies_metadata):
    list_of_adult_per_movie = movies_metadata["adult"].to_list()
    for i in range(len(movies_metadata["adult"])):
        list_of_adult_per_movie[i] = int(list_of_adult_per_movie[i])
    movies_metadata["adult"] = list_of_adult_per_movie
    return movies_metadata


def process_production_countries(movies_metadata):
    list_of_production_countries_per_movie = movies_metadata[
        "production_countries"
    ].to_list()
    temp_list_of_production_countries_per_movie = []
    temp_list_of_production_countries_per_movie.extend(
        list_of_production_countries_per_movie
    )

    for i in range(len(movies_metadata["production_countries"])):
        if (
            temp_list_of_production_countries_per_movie[i] != "[]"
            and not temp_list_of_production_countries_per_movie[i] in COUNTRY_CODES
        ):
            temp_list_of_production_countries_per_movie[i] = COUNTRY_CODES.index(
                (
                    ast.literal_eval(list_of_production_countries_per_movie[i])[0][
                        "iso_3166_1"
                    ]
                ).lower()
            )
        else:
            temp_list_of_production_countries_per_movie[i] = 0

    movies_metadata["production_countries"] = (
        temp_list_of_production_countries_per_movie
    )
    return movies_metadata


def process_release_date(movies_metadata):
    list_of_release_date_per_movie = movies_metadata["release_date"].to_list()
    temp_list_of_release_date_per_movie = []
    temp_list_of_release_date_per_movie.extend(list_of_release_date_per_movie)

    for i in range(len(movies_metadata["release_date"])):
        if temp_list_of_release_date_per_movie[i] != "" and not pd.isnull(
            temp_list_of_release_date_per_movie[i]
        ):
            temp_list_of_release_date_per_movie[i] = (
                temp_list_of_release_date_per_movie[i].replace("-", "")[:-2]
            )
        else:
            temp_list_of_release_date_per_movie[i] = 0

    movies_metadata["release_date"] = temp_list_of_release_date_per_movie
    return movies_metadata


def process_original_language(movies_metadata):
    list_of_original_language_per_movie = movies_metadata["original_language"].to_list()
    for i in range(len(movies_metadata["original_language"])):
        if list_of_original_language_per_movie[i] != "[]" and not pd.isnull(
            list_of_original_language_per_movie[i]
        ):
            list_of_original_language_per_movie[i] = COUNTRY_CODES.index(
                list_of_original_language_per_movie[i].lower()
            )
        else:
            list_of_original_language_per_movie[i] = 0

    movies_metadata["original_language"] = list_of_original_language_per_movie
    return movies_metadata


def process_cast_director(
    movies_metadata: pd.DataFrame, credits_df: pd.DataFrame
) -> list:
    top_actors_list: list[int] = []
    directors_list: list[int] = []
    num_of_misses_cast = 0
    num_of_misses_director = 0
    for entry in movies_metadata.itertuples():
        tmp = credits_df[credits_df["id"] == int(entry.id)]

        if len(tmp.cast.to_list()) != 0:
            cast = ast.literal_eval(tmp.cast.to_list()[0])
        else:
            cast = []
        if len(tmp.crew.to_list()) != 0:
            crew = ast.literal_eval(tmp.crew.to_list()[0])
        else:
            crew = []
        actor_id = cast[0]["id"] if len(cast) > 0 else -1

        director_id = -1
        for i in range(len(cast)):
            if "job" in cast[i]:
                if cast[i]["job"] == "Director":
                    director_id = cast[i]["id"]
                    break
        # only one director per movie

        if director_id == -1:
            for i in range(len(crew)):
                if crew[i]["job"] == "Director":
                    director_id = crew[i]["id"]
                    break
        if actor_id == 1:
            num_of_misses_cast += 1
        if director_id == 1:
            num_of_misses_director += 1
        top_actors_list.append(actor_id)
        directors_list.append(director_id)

    print(f"Cast misses: {num_of_misses_cast}")
    print(f"Director misses: {num_of_misses_director}")
    movies_metadata["top_actor_id"] = top_actors_list
    movies_metadata["director_id"] = directors_list
    return movies_metadata


def _process_keywords(movies_metadata: pd.DataFrame, keywords_df: pd.DataFrame):
    list_keywords_per_movie = []
    num_of_misses = 0
    for entry in keywords_df.itertuples():
        if len(entry.keywords) != 0:
            keywords = ast.literal_eval(entry.keywords)
            tmdbId = entry.id
            if len(keywords) > 0:
                temp = {"tmdbId": tmdbId, "keyword": keywords[0]["id"]}
                list_keywords_per_movie.append(temp)
                continue

        num_of_misses += 1
    movies_metadata["keyword_id"] = list_keywords_per_movie
    return movies_metadata


credits_file_path = r"input/archive/credits.csv"
credits_df = pd.read_csv(credits_file_path)
keywords_file_path = r"input/archive/keywords.csv"
keywords_df = pd.read_csv(keywords_file_path)
file_path = r"output/avg_of_rating_per_movieId.csv"
movies_df = pd.read_csv(file_path)
movies_metadata_file_path = r"input/archive/movies_metadata.csv"
movies_metadata = pd.read_csv(movies_metadata_file_path, low_memory=False)


movie_ids = movies_df["movieId"].to_list()
movie_ids2 = movies_metadata["id"].to_list()

movies_metadata = process_cast_director(movies_metadata, credits_df)
print("Director and top actor DONE")
movies_metadata = process_genres(movies_metadata)
print("Genres DONE")
movies_metadata = process_spoken_languages(movies_metadata)
print("Spoken languages DONE")
movies_metadata = process_production_countries(movies_metadata)
print("Production countries DONE")
movies_metadata = process_release_date(movies_metadata)
print("Release date DONE")
movies_metadata = process_original_language(movies_metadata)
print("Original language DONE")

print("Maching ids metadata and ratings...")

ids = movies_metadata["imdb_id"].to_list()
ids2 = movies_df["imdbId"].to_list()
ratings = movies_df["avg_of_rating"].to_list()
result = []
resultIds = []
for i in range(len(ids)):
    found = False
    for j in range(len(ids2)):
        if ids[i] == ids2[j]:
            found = True
            result.append(ratings[j])
            resultIds.append(ids2[j])
            break
    if found == False:
        result.append(0)
        resultIds.append(ids[i])
new_avg = {"imdbId": resultIds, "avg_of_rating": result}

print("Maching ids metadata and ratings DONE")

output_data = {
    "matched_ids_avg": new_avg["imdbId"],
    "movieId_movies_metadata": movies_metadata["imdb_id"],
    "avg_of_rating": new_avg["avg_of_rating"],
    "budget": movies_metadata["budget"],
    "director_id": movies_metadata["director_id"],
    "top_actor_id": movies_metadata["top_actor_id"],
    "genres": movies_metadata["genres"],
    "original_language": movies_metadata["original_language"],
    "release_date": movies_metadata["release_date"],
    "revenue": movies_metadata["revenue"],
    "spoken_languages": movies_metadata["spoken_languages"],
    "runtime": movies_metadata["runtime"],
    "production_countries": movies_metadata["production_countries"],
    "vote_count": movies_metadata["vote_count"],
}
output_file = pd.DataFrame(output_data)

output_file = output_file[output_file["budget"] != 0]
output_file = output_file[output_file["genres"] != 0]
output_file = output_file[output_file["original_language"] != 0]
output_file = output_file[output_file["release_date"] != 0]
output_file = output_file[output_file["revenue"] != 0]
output_file = output_file[output_file["spoken_languages"] != 0]
output_file = output_file[output_file["runtime"] != 0]
output_file = output_file[output_file["production_countries"] != 0]
output_file = output_file[output_file["vote_count"] != 0]
output_file = output_file[output_file["avg_of_rating"] != -1]
output_file = output_file[output_file["top_actor_id"] != -1]
output_file = output_file[output_file["director_id"] != -1]

print("Omiting movies with not all data DONE")
output_file_path = "output/movies_relevant_data.csv"
output_file.to_csv(output_file_path, index=False)
print("File saved. All DONE!")
