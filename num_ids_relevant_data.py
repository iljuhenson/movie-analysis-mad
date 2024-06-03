import pandas as pd

movies_metadata_file_path = r"output\movies_relevant_data.csv"
movies_metadata = pd.read_csv(movies_metadata_file_path, low_memory=False)
movies_metadata.drop("matched_ids_avg", axis=1, inplace=True)
movies_metadata["movieId_movies_metadata"] = movies_metadata[
    "movieId_movies_metadata"
].str[2:]


output_file_path = r"output\movies_relevant_data_num_ids.csv"
movies_metadata.to_csv(output_file_path, index=False)
print(f"File saved to {output_file_path}")
