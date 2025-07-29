import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix, classification_report

# Load and split the data
df = pd.read_csv('classification_data.csv')
X = df[['feature1', 'feature2']].values
y = df['class'].values
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# A9: Evaluate confusion matrix and other performance metrics

def evaluate_performance(classifier, features_train, labels_train, features_test, labels_test):
    """
    Generates confusion matrix and classification reports for train and test data.
    """
    # Predictions for training data
    y_train_pred = classifier.predict(features_train)

    # Predictions for testing data
    y_test_pred = classifier.predict(features_test)

    # Confusion matrices
    cm_train = confusion_matrix(labels_train, y_train_pred)
    cm_test = confusion_matrix(labels_test, y_test_pred)

    # Classification reports
    report_train = classification_report(labels_train, y_train_pred)
    report_test = classification_report(labels_test, y_test_pred)

    return cm_train, report_train, cm_test, report_test

# --- Main Program ---
if __name__ == '__main__':
    # Train a k-NN classifier (e.g., k=3)
    knn_classifier = KNeighborsClassifier(n_neighbors=3)
    knn_classifier.fit(X_train, y_train)

    # Get performance metrics
    conf_matrix_train, class_report_train, conf_matrix_test, class_report_test = evaluate_performance(
        knn_classifier, X_train, y_train, X_test, y_test
    )

    print("--- Performance Evaluation (k=3) ---")

    # --- Training Data Performance ---
    print("\n--- Training Data ---")
    print("Confusion Matrix:")
    print(conf_matrix_train)
    print("\nClassification Report:")
    print(class_report_train)

    # --- Test Data Performance ---
    print("\n--- Test Data ---")
    print("Confusion Matrix:")
    print(conf_matrix_test)
    print("\nClassification Report:")
    print(class_report_test)

    # --- Inference on Model Fit ---
    train_accuracy = knn_classifier.score(X_train, y_train)
    test_accuracy = knn_classifier.score(X_test, y_test)

    print("\n--- Model Fit Inference ---")
    print(f"Training Accuracy: {train_accuracy:.2f}")
    print(f"Test Accuracy: {test_accuracy:.2f}")

    if train_accuracy > test_accuracy and (train_accuracy - test_accuracy) > 0.1:
        print("Inference: The model might be OVERFITTING.")
        print("Reason: Training accuracy is significantly higher than test accuracy.")
    elif train_accuracy < 0.8 and test_accuracy < 0.8:
        print("Inference: The model might be UNDERFITTING.")
        print("Reason: Both training and test accuracies are low.")
    else:
        print("Inference: The model has a REGULAR FIT (Good Fit).")
        print("Reason: Training and test accuracies are close and reasonably high.")
