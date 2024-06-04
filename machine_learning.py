import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error


file_path = "output/movies_relevant_data_num_ids.csv"
movies_df = pd.read_csv(file_path)

# Data Preparation
encoder = OneHotEncoder(sparse_output=False)
scaler = MinMaxScaler()


# One-hot encoding categorical features
categorical_features_reg = encoder.fit_transform(
    movies_df[
        [
            "adult",
            "genres",
            "original_language",
            "spoken_languages",
            "production_countries",
        ]
    ]
)
scaled_features_reg = scaler.fit_transform(
    movies_df[["budget", "release_date", "runtime", "vote_count", "revenue"]]
)
X = np.hstack([scaled_features_reg, categorical_features_reg])
y = movies_df["avg_of_rating"].values

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Neural Network for Regression
model_reg = tf.keras.Sequential(
    [
        tf.keras.layers.Dense(64, activation="relu", input_shape=(X_train.shape[1],)),
        tf.keras.layers.Dense(32, activation="relu"),
        tf.keras.layers.Dense(1),
    ]
)

model_reg.compile(optimizer="adam", loss="mean_squared_error")
model_reg.summary()

if __name__ == "__main__":
    # Training the Model
    history_reg = model_reg.fit(
        X_train, y_train, epochs=100, batch_size=32, validation_split=0.2, verbose=1
    )

    # Evaluating the Model
    y_pred = model_reg.predict(X_test)
    mae = mean_absolute_error(y_test, y_pred)
    print(f"Mean Absolute Error for Average Rating: {mae:.5f}")
