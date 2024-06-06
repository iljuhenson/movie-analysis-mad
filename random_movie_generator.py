import pandas as pd
import numpy as np


def generate_random_values(num_samples=10):
    df = pd.read_csv('./output/movies_relevant_data_num_ids.csv')
    random_movies = []
    columns = df.columns

    for _ in range(num_samples):
        movie = {}
        for column in columns:
            if df[column].dtype in [np.int64, np.int32]:
                movie[column] = np.random.randint(df[column].min(), df[column].max())
            elif df[column].dtype in [np.float64, np.float32]:
                movie[column] = np.random.uniform(df[column].min(), df[column].max())
        random_movies.append(movie)
    
    return pd.DataFrame(random_movies)


if __name__ == "__main__":
    random_movies_df = generate_random_values(num_samples=10)

    print(generate_random_values(10).tail())
