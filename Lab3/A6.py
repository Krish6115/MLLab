import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier

# Load, split, and train the model
df = pd.read_csv('classification_data.csv')
X = df[['feature1', 'feature2']].values
y = df['class'].values
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
knn_classifier_k3 = KNeighborsClassifier(n_neighbors=3)
knn_classifier_k3.fit(X_train, y_train)

# A6: Test the accuracy of the KNN

def evaluate_accuracy(classifier, features_test, labels_test):
    """
    Calculates the accuracy of a trained classifier.
    """
    accuracy = classifier.score(features_test, labels_test)
    return accuracy

# --- Main Program ---
if __name__ == '__main__':
    test_accuracy = evaluate_accuracy(knn_classifier_k3, X_test, y_test)

    print("--- k-NN Classifier Accuracy Test (k=3) ---")
    print(f"Accuracy on the test set: {test_accuracy:.2f}")
