import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report

# Załaduj dane
data = pd.read_csv('output/movies_relevant_data_num_ids.csv')

# Wyznaczenie etykiet (0 = zły, 1 = dobry)
threshold = data['avg_of_rating'].median()
data['label'] = (data['avg_of_rating'] >= threshold).astype(int)

# Wybór cech do modelu
features = ['adult', 'budget', 'genres', 'original_language', 'release_date', 
            'revenue', 'spoken_languages', 'runtime', 'production_countries', 'vote_count']
X = data[features]
y = data['label']

# Standaryzacja cech
scaler = StandardScaler()
X = scaler.fit_transform(X)

# Podział na zbiory treningowy i testowy
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=62)

# Trenowanie klasyfikatora k-NN
knn = KNeighborsClassifier(n_neighbors=10)
knn.fit(X_train, y_train)

# Predykcja na zbiorze testowym
y_pred = knn.predict(X_test)

# Ocena modelu
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Classification Report:\n", classification_report(y_test, y_pred))
