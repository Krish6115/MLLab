import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Load the dataset
df = pd.read_csv('classification_data.csv')

# A2: Observe the density pattern of a feature

def analyze_feature_density(feature_vector):
    """
    Calculates histogram data, mean, and variance for a feature.
    """
    # Calculate histogram data
    hist_data, bin_edges = np.histogram(feature_vector, bins=5)

    # Calculate mean and variance
    mean_val = np.mean(feature_vector)
    variance_val = np.var(feature_vector)

    return hist_data, bin_edges, mean_val, variance_val

# --- Main Program ---
if __name__ == '__main__':
    feature1 = df['feature1'].values
    hist, bins, mean, variance = analyze_feature_density(feature1)

    print("--- Feature Density Analysis for 'feature1' ---")
    print(f"Histogram Data (Counts per bin): {hist}")
    print(f"Bin Edges: {bins}")
    print(f"Mean: {mean:.2f}")
    print(f"Variance: {variance:.2f}\n")

    # Plot the histogram
    plt.figure(figsize=(8, 6))
    plt.hist(feature1, bins=5, edgecolor='black', alpha=0.7)
    plt.title('Histogram of Feature 1')
    plt.xlabel('Feature 1 Value')
    plt.ylabel('Frequency')
    plt.grid(True)
    plt.show()
