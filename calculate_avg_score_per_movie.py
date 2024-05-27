# Importing necessary libraries
import pandas as pd


# Loading the dataset
file_path = r"input\\archive\\ratings.csv"
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

output_data = {
    "movieId": movie_rating_sum_and_rating_counter.keys(),
    "avg_of_rating": average_rating,
}

output_df = pd.DataFrame(output_data)
output_df = output_df[output_df["avg_of_rating"] != -1]
output_file_path = "output/avg_of_rating_per_movie_Id.csv"

output_df.to_csv(output_file_path, index=False)
