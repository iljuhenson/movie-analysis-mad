# Importing necessary libraries
import pandas as pd

# Loading the dataset
file_path = r"input\\archive\\ratings.csv"
movies_df = pd.read_csv(file_path)

movie_sum_of_rating = [0.0] * len(movies_df["movieId"])
movie_sum_of_ratings_records = [0] * len(movies_df["movieId"])
for idx, row in movies_df.iterrows():
    movie_sum_of_rating[int(row["movieId"])] += float(row["rating"])
    movie_sum_of_ratings_records[int(row["movieId"])] += 1
# Calculating the average of rating for each movie
movie_avg_of_rating = [0.0] * len(movies_df["movieId"])
for i in range(len(movie_sum_of_rating)):
    movie_avg_of_rating[i] = movie_sum_of_rating[i] / movie_sum_of_ratings_records[i]


output_file_path = "output/avg_of_rating_per_movieId.csv"
output_data = {
    "movieId": movies_df["movieId"],
    "avg_of_rating": movie_avg_of_rating,
}
output_df = pd.DataFrame(output_data)
output_df.to_csv(output_file_path, index=False)
