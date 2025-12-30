from sklearn.metrics import accuracy_score, confusion_matrix
import matplotlib.pyplot as plt

# Evaluate model performance
def evaluate_model(predictions, true_labels):
    accuracy = accuracy_score(true_labels, predictions)
    cm = confusion_matrix(true_labels, predictions)
    print(f"Accuracy: {accuracy}")
    print(f"Confusion Matrix: {cm}")

    # ROC curve
    fpr, tpr, thresholds = roc_curve(true_labels, predictions)
    plt.plot(fpr, tpr, label="ROC curve")
    plt.show()
