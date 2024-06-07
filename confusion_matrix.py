from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt 
import seaborn as sns
import numpy as np

def plot_confusion_matrix(y_test, y_pred, method_name=''):
    labels = np.array(y_test)
    preds = np.array(y_pred)
    cm = confusion_matrix(labels, preds)
    plt.figure(figsize=(10, 7))
    sns.heatmap(
        cm,
        annot=cm,
        fmt="d",
        xticklabels=["Przewidziany zły", "Przewidziany dobry"],
        yticklabels=["Faktycznie zły", "Faktycznie dobry"],
    )
    plt.xlabel("Przewidziany stan")
    plt.ylabel("Stan faktyczny")
    plt.title("Macierz błędu")
    file_name = "confusion_matrix_" + method_name + ".png"
    plt.savefig(file_name)
    plt.show()