# Importing necessary libraries
import pandas as pd

# Loading the dataset
file_path = r"input/archive/ratings.csv"
reviews_df = pd.read_csv(file_path)

movie_rating_sum_and_rating_counter: dict[int, [int, int]] = {}

for review_field in reviews_df.itertuples():
    if review_field.movieId in movie_rating_sum_and_rating_counter:
        rating, amount = movie_rating_sum_and_rating_counter[review_field.movieId]

        movie_rating_sum_and_rating_counter[review_field.movieId] = [
            rating + review_field.rating,
            amount + 1,
        ]
    else:
        movie_rating_sum_and_rating_counter[review_field.movieId] = [
            review_field.rating,
            1,
        ]

average_rating: list[int] = []

for movie_id, [rating, amount] in movie_rating_sum_and_rating_counter.items():
    if amount > 0:
        average_rating.append(rating / amount)
    else:
        average_rating.append(-1)

# Adding additional ids for the dataset
file_path_for_ids = r"input/archive/links.csv"
links_df = pd.read_csv(file_path_for_ids)

tmdbIds: list[str] = []
imdbIds: list[str] = []

for movie_id in movie_rating_sum_and_rating_counter.keys():
    item = links_df[links_df["movieId"] == movie_id]
    if not pd.isna(item.imdbId.values[0]):
        value = str(int((item.imdbId.values[0])))
        while len(value) < 7:
            value = "0" + value
        value = "tt" + value
        imdbIds.append(value)
    else:
        imdbIds.append("-1")  # -1 Dla braku wartości

    if not pd.isna(item.tmdbId.values[0]):

        value = str(int((item.tmdbId.values[0])))
        tmdbIds.append(value)
    else:
        tmdbIds.append("-1")  # -1 Dla braku wartości
output_data = {
    "movieId": movie_rating_sum_and_rating_counter.keys(),
    "avg_of_rating": average_rating,
    "imdbId": imdbIds,
    "tmdbId": tmdbIds,
}

output_df = pd.DataFrame(output_data)
output_df = output_df[output_df["avg_of_rating"] != -1]
output_file_path = "output/avg_of_rating_per_movieId.csv"

output_df.to_csv(output_file_path, index=False)
