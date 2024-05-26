import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

file_path = "output/movies_relevant_data.csv"
movies_df = pd.read_csv(file_path)

numerical_columns = [
    #"movieId",
    "avg_of_rating",
    "adult",
    "budget",
    "genres",
    "original_language",
    "release_date",
    "revenue",
    "spoken_languages",
    "runtime",
    "production_countries",
    "vote_count",
]
plt.figure(figsize=(15, 10))
for i, col in enumerate(numerical_columns, 1):
    plt.subplot(3, 4, i)
    sns.histplot(movies_df[col])
    plt.title(f'Distribution of {col}')
    plt.xlabel(col)
    plt.ylabel('Frequency')

plt.tight_layout()
plt.show()
print('done')
