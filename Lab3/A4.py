import pandas as pd
from sklearn.model_selection import train_test_split

# Load the dataset
df = pd.read_csv('classification_data.csv')

# Separate features (X) and target (y)
X = df[['feature1', 'feature2']].values
y = df['class'].values

# A4: Split dataset into train and test sets

def split_data(feature_matrix, target_vector, test_size=0.3, random_state=42):
    """
    Splits the dataset into training and testing sets.
    """
    X_train, X_test, y_train, y_test = train_test_split(
        feature_matrix, target_vector, test_size=test_size, random_state=random_state
    )
    return X_train, X_test, y_train, y_test

# --- Main Program ---
if __name__ == '__main__':
    X_train, X_test, y_train, y_test = split_data(X, y)

    print("--- Dataset Splitting ---")
    print(f"Total samples: {len(X)}")
    print(f"Training samples: {len(X_train)}")
    print(f"Testing samples: {len(X_test)}")
