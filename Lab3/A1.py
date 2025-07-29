import numpy as np
import pandas as pd

# Load the dataset
df = pd.read_csv('classification_data.csv')

# Separate the classes
class_0 = df[df['class'] == 0].iloc[:, :-1].values
class_1 = df[df['class'] == 1].iloc[:, :-1].values

# A1: Calculate intraclass spread and interclass distances

def evaluate_class_separation(class_a_features, class_b_features):
    """
    Calculates and returns the centroids, spreads, and interclass distance.
    """
    # Calculate centroids (mean vectors) for each class
    centroid_a = np.mean(class_a_features, axis=0)
    centroid_b = np.mean(class_b_features, axis=0)

    # Calculate spread (standard deviation) for each class
    spread_a = np.std(class_a_features, axis=0)
    spread_b = np.std(class_b_features, axis=0)

    # Calculate the distance between the centroids
    interclass_dist = np.linalg.norm(centroid_a - centroid_b)

    return centroid_a, centroid_b, spread_a, spread_b, interclass_dist

# --- Main Program ---
if __name__ == '__main__':
    # Get the evaluation metrics
    centroid1, centroid2, spread1, spread2, distance = evaluate_class_separation(class_0, class_1)

    print("--- Class Separation Analysis ---")
    print(f"Class 0 Centroid: {centroid1}")
    print(f"Class 1 Centroid: {centroid2}\n")

    print(f"Class 0 Spread (Std Dev): {spread1}")
    print(f"Class 1 Spread (Std Dev): {spread2}\n")

    print(f"Distance between Class Centroids (Interclass Distance): {distance:.2f}")
