from machine_learning import y_test, y_pred
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score, classification_report
import numpy as np
import matplotlib.pyplot as plt

labels = np.array(y_test)
preds = np.array(y_pred)

cm = confusion_matrix(labels, preds)

plt.figure(figsize=(10, 7))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Predicted Negative', 'Predicted Positive'], yticklabels=['Actual Negative', 'Actual Positive'])
plt.xlabel('Prediction')
plt.ylabel('Actual')
plt.title('Confusion Matrix')
plt.show()