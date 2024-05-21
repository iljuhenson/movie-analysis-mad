# Importing necessary libraries
import pandas as pd
import concurrent.futures


def calculate_avg_rating(start_idx, end_idx):
    for idx in range(start_idx, end_idx):
        movie_sum_of_rating[int(movies_df.loc[idx, "movieId"])] += float(
            movies_df.loc[idx, "rating"]
        )
        movie_amount_of_ratings_records[int(movies_df.loc[idx, "movieId"])] += 1
        if idx % 100000 == 0:
            print(f"Processed {idx} records")


# Loading the dataset
file_path = r"input\\archive\\ratings.csv"
movies_df = pd.read_csv(file_path)

# amount_of_movies = len(movies_df["movieId"].unique())
amount_of_reviews = len(movies_df)

movie_sum_of_rating = [0.0] * amount_of_reviews
movie_amount_of_ratings_records = [0] * amount_of_reviews

# Split the range into chunks
chunk_size = 1000000
num_chunks = amount_of_reviews // chunk_size
ranges = [(i * chunk_size, (i + 1) * chunk_size) for i in range(num_chunks)]

# Create a thread pool
with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
    # Submit the tasks to the thread pool
    futures = [
        executor.submit(calculate_avg_rating, start_idx, end_idx)
        for start_idx, end_idx in ranges
    ]

    # Wait for all tasks to complete
    concurrent.futures.wait(futures)

# Calculating the average of rating for each movie

movie_avg_of_rating = [0.0] * amount_of_reviews

for i in range(amount_of_reviews):
    # If there is no rating for a movie, set the average rating to -1
    try:
        movie_avg_of_rating[i] = (
            movie_sum_of_rating[i] / movie_amount_of_ratings_records[i]
        )
    except ZeroDivisionError:
        movie_avg_of_rating[i] = -1

output_data = {
    "movieId": movies_df["movieId"],
    "avg_of_rating": movie_avg_of_rating,
}
output_df = pd.DataFrame(output_data)
# omiting the movies with no rating
output_df = output_df[output_df["avg_of_rating"] != -1]
output_file_path = "output/avg_of_rating_per_movieId.csv"


output_df.to_csv(output_file_path, index=False)
