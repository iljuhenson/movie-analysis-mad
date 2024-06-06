import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.linear_model import LogisticRegression
from constants import NUMERICAL_COLUMNS
import seaborn as sns
import matplotlib.pyplot as plt

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

data = pd.read_csv('output/movies_relevant_data_num_ids.csv')

threshold = data['avg_of_rating'].median()
data['label'] = (data['avg_of_rating'] >= threshold).astype(int)

features = NUMERICAL_COLUMNS
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
from sklearn.svm import LinearSVC
svc = LinearSVC()
svc.fit(X_train, y_train)
y_pred = svc.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))
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

