import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Load the dataset
df = pd.read_csv('classification_data.csv')
X = df.iloc[:, :-1].values

# A3: Calculate and plot Minkowski distance

def calculate_minkowski_series(vector1, vector2, max_r):
    """
    Calculates Minkowski distance for r from 1 to max_r.
    """
    distances = []
    r_values = range(1, max_r + 1)
    for r in r_values:
        distance = np.linalg.norm(vector1 - vector2, ord=r)
        distances.append(distance)
    return list(r_values), distances

# --- Main Program ---
if __name__ == '__main__':
    # Take two feature vectors from the dataset
    vec1 = X[0]  # First data point
    vec2 = X[10] # Eleventh data point

    r_vals, minkowski_distances = calculate_minkowski_series(vec1, vec2, 10)

    print("--- Minkowski Distance Analysis ---")
    print(f"Vector 1: {vec1}")
    print(f"Vector 2: {vec2}\n")
    for r, dist in zip(r_vals, minkowski_distances):
        print(f"Minkowski distance for r={r}: {dist:.2f}")

    # Plot the distances
    plt.figure(figsize=(8, 6))
    plt.plot(r_vals, minkowski_distances, marker='o', linestyle='--')
    plt.title('Minkowski Distance vs. r')
    plt.xlabel('r value')
    plt.ylabel('Minkowski Distance')
    plt.grid(True)
    plt.xticks(r_vals)
    plt.show()
