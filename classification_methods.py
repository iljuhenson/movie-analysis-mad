import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.linear_model import LogisticRegression
from constants import CLUSTER_COLUMNS
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import load_iris
from sklearn.svm import LinearSVC

def knn_best_params(X_train, y_train, X_test, y_test):
    best_k = 0
    best_score = 0
    for k in range(1, 100):
        knn = KNeighborsClassifier(n_neighbors=k)
        knn.fit(X_train, y_train)
        y_pred = knn.predict(X_test)
        score = accuracy_score(y_test, y_pred)
        if score > best_score:
            best_score = score
            best_k = k

    return best_k, best_score




def hybrid_method_predict(X_vals) -> np.ndarray:
 
    data = pd.read_csv('output/movies_relevant_data_num_ids.csv')

    threshold = data['avg_of_rating'].median()
    data['label'] = (data['avg_of_rating'] >= threshold).astype(int)

    features = CLUSTER_COLUMNS
    X = data[features]
    y = data['label']

    
    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    svc = LinearSVC()
    svc.fit(X_train, y_train)
    d1 = svc.predict(X_vals)

    logistic = LogisticRegression()
    logistic.fit(X_train, y_train)
    d2 = logistic.predict(X_vals)

    knn = KNeighborsClassifier(n_neighbors=8)
    knn.fit(X_train, y_train)
    d3 = knn.predict(X_vals)

    ans = []
    for val1, val2, val3 in zip(d1, d2, d3):
        ans.append(round((val1 + val2 + val3) / 3))
    
    return np.array(ans)


data = pd.read_csv('output/movies_relevant_data_num_ids.csv')
threshold = data['avg_of_rating'].median()
data['label'] = (data['avg_of_rating'] >= threshold).astype(int)
data = data.drop('avg_of_rating', axis=1)



features = CLUSTER_COLUMNS
X = data[features]
y = data['label']


scaler = StandardScaler()
X = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


knn = KNeighborsClassifier(n_neighbors=8)
knn.fit(X_train, y_train)


logistic = LogisticRegression()
logistic.fit(X_train, y_train)
y_pred = knn.predict(X_test)


print("Accuracy:", accuracy_score(y_test, y_pred))
print("Classification Report:\n", classification_report(y_test, y_pred))
y_pred = logistic.predict(X_test)

print("Accuracy:", accuracy_score(y_test, y_pred))
print("Classification Report:\n", classification_report(y_test, y_pred))

# creating linear svc classifier
svc = LinearSVC()
svc.fit(X_train, y_train)
y_pred = svc.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Classification Report:\n", classification_report(y_test, y_pred))
y_pred = hybrid_method_predict(X_test)
print("Accuracy: ", accuracy_score(y_test, y_pred))
print("Classification Report:\n", classification_report(y_test, y_pred))

labels = np.array(y_test)
preds = np.array(y_pred)

cm = confusion_matrix(labels, preds)

plt.figure(figsize=(10, 7))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Predicted Negative', 'Predicted Positive'], yticklabels=['Actual Negative', 'Actual Positive'])
plt.xlabel('Prediction')
plt.ylabel('Actual')
plt.title('Confusion Matrix')
plt.savefig("output/confusion_matrix.png")
scores = cross_val_score(knn, X, y, cv=10)

# Wyświetlenie wyników
print(f"Średni wynik walidacji krzyżowej: {scores.mean():.2f}")
print(f"Odchylenie standardowe wyników: {scores.std():.2f}")


# Generacja losowych dannyc i pokazanie wyników
import random_movie_generator
rand_values = random_movie_generator.generate_random_values(5)[CLUSTER_COLUMNS]
rand_values = scaler.fit_transform(rand_values)

print("\nPredykcja dla losowych danych:")
print(knn.predict(rand_values))
print(logistic.predict(rand_values))
print(svc.predict(rand_values))