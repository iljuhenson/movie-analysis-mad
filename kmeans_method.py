import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans

file_path = "output/movies_relevant_data_num_ids.csv"
movies_df = pd.read_csv(file_path)
clustering_features_simple = [
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

X_clustering_simple = movies_df[clustering_features_simple]

wcss = []
for i in range(1, 11):
    kmeans = KMeans(n_clusters=i, init="k-means++", random_state=42)
    kmeans.fit(X_clustering_simple)
    wcss.append(kmeans.inertia_)

second_derivative = (
    [0] + [wcss[i] - 2 * wcss[i + 1] + wcss[i + 2] for i in range(len(wcss) - 2)] + [0]
)

# The optimal k is where the second derivative is maximum
optimal_k = second_derivative.index(max(second_derivative)) + 1

print(f"The optimal number of clusters (k) is: {optimal_k}")

# Plotting the Elbow method graph (optional)
plt.figure(figsize=(10, 6))
plt.plot(range(1, 11), wcss, marker="o")
plt.title("Elbow Method")
plt.xlabel("Number of clusters")
plt.ylabel("WCSS")
plt.show()

# optimal_k = 2

# Fitting k-means with best num of clasters
kmeans_simple = KMeans(n_clusters=optimal_k, init="k-means++", random_state=42)
movies_df["cluster"] = kmeans_simple.fit_predict(X_clustering_simple)

# Grouping the data by clusters to analyze the characteristics
cluster_summary_simple = movies_df.groupby("cluster").mean()[clustering_features_simple]
cluster_summary_simple["number_of_movies"] = movies_df["cluster"].value_counts()
print(cluster_summary_simple)
pd.save(cluster_summary_simple, "output/cluster_summary_simple.txt")


# Understanding the results of the k-means
# Selecting the features used for clustering

movies_df_clustering = movies_df[clustering_features_simple + ["cluster"]]

# Replacing the cluster labels with string labels for better visualization
movies_df_clustering["cluster"] = movies_df_clustering["cluster"].astype(str)

# Creating a pairplot to visualize all possible XY graphs of every k-means group
sns.pairplot(movies_df_clustering, hue="cluster", palette="viridis", diag_kind="kde")
# plt.show()
plt.savefig("output/kmeans_pairplot.png")
