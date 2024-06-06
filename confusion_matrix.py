from machine_learning import y_test, y_pred
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score, classification_report

labels = np.array(y_test)
preds = np.array(y_pred)

cm = confusion_matrix(labels, preds)

plt.figure(figsize=(10, 7))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Predicted Negative', 'Predicted Positive'], yticklabels=['Actual Negative', 'Actual Positive'])
plt.xlabel('Prediction')
plt.ylabel('Actual')
plt.title('Confusion Matrix')
plt.show()

precision = round(precision_score(preds, labels), 3)
recall = round(recall_score(preds, labels), 3)
f1 = round(f1_score(preds, labels), 3)

print("Precision:", precision)
print("Recall:", recall)
print("F1 Score:", f1)
print(classification_report(preds, labels))

print('hi')