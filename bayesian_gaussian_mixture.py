import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.mixture import BayesianGaussianMixture
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
from constants import NUMERICAL_COLUMNS

file_path = "output/movies_relevant_data_num_ids.csv"
movies_df = pd.read_csv(file_path)

# Selecting relevant features for clustering
clustering_features = NUMERICAL_COLUMNS

X = movies_df[clustering_features]

# Standardize the features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Splitting the data into training and testing sets
X_train, X_test = train_test_split(X_scaled, test_size=0.2, random_state=42)

# Bayesian Gaussian Mixture Model
bgmm = BayesianGaussianMixture(n_components=2, covariance_type="full", random_state=42)
bgmm.fit(X_train)  # Train the model on the training set

# Predicting clusters for the testing set
test_clusters = bgmm.predict(X_test)

# Evaluate the model using silhouette score
silhouette_avg = silhouette_score(X_test, test_clusters)
print("Silhouette Score:", silhouette_avg)

# Save the cluster labels to a CSV file
cluster_labels = pd.DataFrame(test_clusters, columns=["cluster"])
cluster_labels["movieId"] = movies_df["movieId_movies_metadata"]
cluster_labels["avg_of_rating"] = movies_df["avg_of_rating"]
cluster_labels.to_csv("output/cluster_labels.csv", index=False)

