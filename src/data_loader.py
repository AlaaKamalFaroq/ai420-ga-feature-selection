import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from config import RANDOM_STATE, TEST_SIZE, NORMALIZE, KNN_NEIGHBORS

def load_data():
    data = load_breast_cancer()
    X, y = data.data, data.target
    names = data.feature_names
    print(f"Dataset Loaded: {X.shape[0]} samples, {X.shape[1]} features")
    return X, y, names    
    import kagglehub
    import os
    
    download_path = "data/malaria-dataset"
    os.makedirs(download_path, exist_ok=True)
    
    # تحميل البيانات
    dataset = kagglehub.dataset_download(
        "iarunava/cell-images-for-detecting-malaria",
        path=download_path
    )
    
    print(f"done {dataset}")

def run_eda(X, y, names):
    df = pd.DataFrame(X, columns=names)
    plt.figure(figsize=(12, 10))
    sns.heatmap(df.corr(), cmap='coolwarm')
    plt.title('Feature Correlation')
    plt.tight_layout()
    plt.savefig('results/plots/eda_correlation.png')
    plt.close()
    print("Correlation plot saved.")

def preprocess(X, y):
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )
    if NORMALIZE:
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)
    return X_train, X_test, y_train, y_test

def get_baseline_accuracy(X_train, X_test, y_train, y_test):
    model = KNeighborsClassifier(n_neighbors=KNN_NEIGHBORS)
    model.fit(X_train, y_train)
    acc = accuracy_score(y_test, model.predict(X_test))
    print(f"Baseline Accuracy: {acc:.4f}")
    return acc

def evaluate_subset(X_train, X_test, y_train, y_test, indices):
    if len(indices) == 0: return 0.0
    model = KNeighborsClassifier(n_neighbors=KNN_NEIGHBORS)
    model.fit(X_train[:, indices], y_train)
    return model.score(X_test[:, indices], y_test)

if __name__ == "__main__":
    X, y, names = load_data()
    run_eda(X, y, names)
    X_train, X_test, y_train, y_test = preprocess(X, y)
    get_baseline_accuracy(X_train, X_test, y_train, y_test)