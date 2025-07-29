import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier

# Load and split the data
df = pd.read_csv('classification_data.csv')
X = df[['feature1', 'feature2']].values
y = df['class'].values
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# A8: Vary k from 1 to 11 and plot accuracy

def plot_knn_accuracy_vs_k(features_train, labels_train, features_test, labels_test, max_k):
    """
    Trains KNN for k=1 to max_k and plots the accuracy.
    """
    k_values = range(1, max_k + 1)
    accuracies = []

    for k in k_values:
        classifier = KNeighborsClassifier(n_neighbors=k)
        classifier.fit(features_train, labels_train)
        accuracy = classifier.score(features_test, labels_test)
        accuracies.append(accuracy)

    return list(k_values), accuracies

# --- Main Program ---
if __name__ == '__main__':
    k_vals, accuracy_scores = plot_knn_accuracy_vs_k(X_train, y_train, X_test, y_test, 11)

    print("--- Accuracy vs. k Analysis ---")
    for k, acc in zip(k_vals, accuracy_scores):
        print(f"k = {k}, Accuracy = {acc:.2f}")

    # Plot the accuracy
    plt.figure(figsize=(10, 6))
    plt.plot(k_vals, accuracy_scores, marker='o', linestyle='-', color='b')
    plt.title('k-NN Test Accuracy vs. Number of Neighbors (k)')
    plt.xlabel('k (Number of Neighbors)')
    plt.ylabel('Accuracy')
    plt.xticks(k_vals)
    plt.grid(True)
    plt.show()
