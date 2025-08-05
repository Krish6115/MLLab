import random
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, classification_report, mean_squared_error, mean_absolute_percentage_error, r2_score
from sklearn.metrics import accuracy_score

#============ Utility Functions ===================
def get_data_csv(csv):
    d = pd.read_csv(csv)
    return d

def get_data_excel(excel, sheet):
    d = pd.read_excel(excel, sheet_name=sheet)
    return d

#============ #A1 =================================
print("\n\n==========A1===========")
csv = "classification_data.csv"
data = get_data_csv(csv)
print(data)

X = data[['feature1']]
y = data['class']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

n = KNeighborsClassifier(n_neighbors=3)
n.fit(X_train, y_train)

y_pred_test = n.predict(X_test)
y_pred_train = n.predict(X_train)

cm_test = confusion_matrix(y_test, y_pred_test)
cm_train = confusion_matrix(y_train, y_pred_train)

print("Confusion Matrix (k=3):\n")
print(cm_test, "\n", cm_train)


def sensitivityRecall(cm):
    return cm[1][1] / (cm[1][1] + cm[1][0])

def specificity(cm):
    return cm[0][0] / (cm[0][0] + cm[0][1])

def precision(cm):
    return cm[1][1] / (cm[1][1] + cm[0][1])

def accuracy(cm):
    return (cm[1][1] + cm[0][0]) / (cm[1][1] + cm[0][1] + cm[0][0] + cm[1][0])

def FbScore(cm, b):
    precision_v = precision(cm)
    sensitivityRecall_v = sensitivityRecall(cm)
    return ((1 + b ** 2) * (precision_v * sensitivityRecall_v)) / ((b ** 2) * precision_v + sensitivityRecall_v)


print("F1Score", FbScore(cm_test, 1),
      "\nsensitivityRecall", sensitivityRecall(cm_test),
      "\nspecificity", specificity(cm_test),
      "\nprecision", precision(cm_test),
      "\naccuracy", accuracy(cm_test))

#============ #A2 =================================
print("\n\n==========A2===========")
def get_data_csv(csv):
    d = pd.read_csv(csv)
    return d

# Load the new CSV file
csv = "classification_data.csv"
data = get_data_csv(csv)

# Prepare features and labels
X = data[['feature1', 'feature2']]
y = data['class']

# Split data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)  

# Create and train the KNN classifier with k=3
knn = KNeighborsClassifier(n_neighbors=3)
knn.fit(X_train, y_train)

# Predictions
y_pred = knn.predict(X_test)

# Confusion matrix
cm = confusion_matrix(y_test, y_pred)

# Print confusion matrix and classification metrics
print("Confusion Matrix:\n", cm)
print("\nClassification Report:\n", classification_report(y_test, y_pred))
print("Accuracy:", accuracy_score(y_test, y_pred))
#============ #A3 =================================
print("\n\n==========A3===========")
# Simulated data demo scatter plot
X1 = [7, 10, 10, 7, 3, 8, 5, 4, 8, 6, 3, 6, 7, 1, 2, 4, 8, 5, 3, 3]
X2 = [7, 2, 7, 1, 2, 10, 10, 5, 4, 3, 1, 1, 2, 4, 6, 2, 1, 4, 7, 4]
y = [1 if (X1[i] + X2[i]) > 10 else 0 for i in range(len(X1))]

plt.figure(figsize=(8, 6))
for x1, x2, label in zip(X1, X2, y):
    color = 'red' if label == 0 else 'blue'
    plt.scatter(x1, x2, c=color)
plt.title("A3: Example Binary Classification Scatter Plot")
plt.grid(True)
plt.show()

#============ #A4 is not present (skipped) ========

#============ #A5 =================================
print("\n\n==========A5===========")
# Large random sample as per your paste for demo
np.random.seed(0)
X1 = np.random.uniform(0, 10, 1000)
X2 = np.random.uniform(0, 10, 1000)
y = [1 if (X1[i] + X2[i]) > 10 else 0 for i in range(len(X1))]

X = list(zip(X1, X2))
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

n_list = [KNeighborsClassifier(n_neighbors=i) for i in range(1, 5)]
for i in range(4):
    n_list[i].fit(X_train, y_train)

y_pred = [n_list[i].predict(X_test) for i in range(4)]

plt.figure(figsize=(8, 6))
for i in range(4):
    for (x1, x2), label in zip(X_test, y_pred[i]):
        color = 'red' if label == 0 else 'blue'
        plt.scatter(x1, x2, c=color, edgecolor='k', alpha=0.15)
plt.title("A5: KNN Classifications for k=1 to 4")
plt.grid(True)
plt.show()

#============ #A6 =================================
print("\n\n==========A6===========")
csv = "classification_data.csv"
try:
    dataF = get_data_csv(csv)

    X1 = dataF.iloc[340:360, 4]
    X2 = dataF.iloc[340:360, 5]
    y = dataF.iloc[340:360, 129]
    plt.figure(figsize=(8, 6))
    for x1, x2, label in zip(X1, X2, y):
        color = 'red' if label == 0 else 'blue'
        plt.scatter(x1, x2, c=color)
    plt.title("A6: fMRI Feature Group Scatter (Example)")
    plt.show()

    X1 = dataF.iloc[1:780, 4]
    X2 = dataF.iloc[1:780, 5]
    y = dataF.iloc[1:780, 129]

    X = list(zip(X1, X2))
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)
    n = KNeighborsClassifier(n_neighbors=3)
    n.fit(X_train, y_train)
    y_pred = n.predict(X_test)

    plt.figure(figsize=(8, 6))
    for (x1, x2), label in zip(X_test, y_pred):
        color = 'red' if label == 0 else 'blue'
        plt.scatter(x1, x2, c=color, edgecolor='k', alpha=0.6)
    plt.title("A6: KNN Results on fMRI Features")
    plt.grid(True)
    plt.show()
except Exception as e:
    print(f"Could not load A6 csv data: {e}")

#============ #A7 =================================
print("\n\n==========A7===========")
csv = "classification_data.csv"
try:
    dataF = get_data_csv(csv)
    X1 = dataF.iloc[1:780, 4]
    X2 = dataF.iloc[1:780, 5]
    y = dataF.iloc[1:780, 129]
    X = list(zip(X1, X2))

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    # Grid Search
    param_grid = {'n_neighbors': list(range(1, 21))}
    grid_search = GridSearchCV(KNeighborsClassifier(), param_grid, cv=5, scoring='accuracy')
    grid_search.fit(X_train, y_train)
    best_k = grid_search.best_params_['n_neighbors']
    best_score = grid_search.best_score_
    best_model = KNeighborsClassifier(n_neighbors=best_k)
    best_model.fit(X_train, y_train)
    y_pred = best_model.predict(X_test)
    test_accuracy = accuracy_score(y_test, y_pred)
    print(f"Best k gridsearch: {best_k}")
    print(f"Best Cross-Validation Accuracy: {best_score:.4f}")
    print(f"Test Accuracy with k={best_k}: {test_accuracy:.4f}")

    # Randomized Search
    param_dist = {'n_neighbors': np.arange(1, 21)}
    random_search = RandomizedSearchCV(
        KNeighborsClassifier(),
        param_distributions=param_dist,
        n_iter=10, cv=5,
        scoring='accuracy', random_state=42)
    random_search.fit(X_train, y_train)
    best_k_random = random_search.best_params_['n_neighbors']
    print(f"Best k (Random Search): {best_k_random}")
except Exception as e:
    print(f"Could not load A7 csv data: {e}")

