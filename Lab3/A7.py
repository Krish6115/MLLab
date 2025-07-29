import numpy as np
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

# A7: Study the prediction behavior of the classifier

def get_predictions(classifier, features_test):
    """
    Gets predictions for a set of test vectors.
    """
    return classifier.predict(features_test)

def predict_single_vector(classifier, vector):
    """
    Predicts the class for a single feature vector.
    """
    # Reshape the vector to be 2D
    return classifier.predict(vector.reshape(1, -1))

# --- Main Program ---
if __name__ == '__main__':
    # Get predictions for the entire test set
    all_test_predictions = get_predictions(knn_classifier_k3, X_test)

    # Predict a single vector from the test set
    test_vector_instance = X_test[0]
    single_prediction = predict_single_vector(knn_classifier_k3, test_vector_instance)

    print("--- Prediction Behavior Analysis ---")
    print(f"Predictions for the entire test set: {all_test_predictions}")
    print(f"Actual labels for the test set:   {y_test}\n")

    print(f"Test vector: {test_vector_instance}")
    print(f"Predicted class for the vector: {single_prediction[0]}")
    print(f"Actual class for the vector: {y_test[0]}")
