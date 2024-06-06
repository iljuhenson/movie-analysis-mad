import pandas as pd
import numpy as np


def generate_random_values(num_samples=10):
    df = pd.read_csv("./output/movies_relevant_data_num_ids.csv")
    df.drop("avg_of_rating", axis=1, inplace=True)  # Drop the target column
    df.drop(
        "movieId_movies_metadata", axis=1, inplace=True
    )  # Drop the movieId column (it's not a feature
    random_movies = []
    columns = df.columns

    for _ in range(num_samples):
        movie = {}
        for column in columns:
            if column == "avg_of_rating":
                movie[column] = np.random.uniform(df[column].min(), df[column].max())
            else:
                movie[column] = int(np.random.uniform(min(df[column]), max(df[column])))
        random_movies.append(movie)

    return pd.DataFrame(random_movies)


if __name__ == "__main__":
    random_movies_df = generate_random_values(num_samples=10)

    print(generate_random_values(10).tail())
