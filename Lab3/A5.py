import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier

# Load and split the data
df = pd.read_csv('classification_data.csv')
X = df[['feature1', 'feature2']].values
y = df['class'].values
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# A5: Train a KNN classifier

def train_knn(features_train, labels_train, n_neighbors=3):
    """
    Trains a KNeighborsClassifier.
    """
    classifier = KNeighborsClassifier(n_neighbors=n_neighbors)
    classifier.fit(features_train, labels_train)
    return classifier

# --- Main Program ---
if __name__ == '__main__':
    knn_classifier_k3 = train_knn(X_train, y_train, n_neighbors=3)

    print("--- k-NN Classifier Training (k=3) ---")
    print("Classifier trained successfully!")
    print(f"Model details: {knn_classifier_k3}")
