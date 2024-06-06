from sklearn.cluster import DBSCAN
from sklearn.metrics import adjusted_rand_score
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from constants import NUMERICAL_COLUMNS
import pandas as pd

file_path = "output/movies_relevant_data_num_ids.csv"
movies_df = pd.read_csv(file_path)

# Selecting relevant features for clustering
clustering_features = NUMERICAL_COLUMNS

# Define features (X) and target (y)
X = movies_df[clustering_features].drop(columns=["avg_of_rating"])
y = movies_df["avg_of_rating"]

# Standardize the features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Splitting the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42
)
if __name__ == "__main__":
    # Train the DBSCAN model
    dbscan = DBSCAN(eps=0.5, min_samples=5)

    dbscan.fit(X_train)

    # Predicting clusters for the testing set
    test_clusters = dbscan.fit_predict(X_test)

    # Validation: Create a binary classification for `avg_of_rating`
    threshold = y_test.median()
    y_test_clusters = (y_test > threshold).astype(int)

    # Validate the clustering result using Adjusted Rand Index
    ari_score = adjusted_rand_score(y_test_clusters, test_clusters)

    print(f"Adjusted Rand Index: {ari_score}")
