from dbscan import *


# Training the Model
for i in range(1, 10):
    for j in range(1, 30):
        # Train the DBSCAN model
        dbscan = DBSCAN(eps=i / 10, min_samples=j)

        dbscan.fit(X_train)

        # Predicting clusters for the testing set
        test_clusters = dbscan.fit_predict(X_test)

        # Validation: Create a binary classification for `avg_of_rating`
        threshold = y_test.median()
        y_test_clusters = (y_test > threshold).astype(int)

        # Validate the clustering result using Adjusted Rand Index
        ari_score = adjusted_rand_score(y_test_clusters, test_clusters)

        print(f"Adjusted Rand Index: {ari_score}")
        with open(r"output/dbscan_opti.txt", "a") as file:
            file.write(
                f"Eps: {i/10}, min_samples: {j},  "
                + f"Adjusted Rand Index: {ari_score:.5f}\n"
            )
