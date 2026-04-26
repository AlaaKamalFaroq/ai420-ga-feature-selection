import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report

from src.config import RANDOM_STATE, TEST_SIZE, NORMALIZE, KNN_NEIGHBORS

# ── Paths ────────────────────────────────────────────────────────────────────
# This is the corrected path based on the standard Kaggle structure for this dataset
DATA_DIR     = Path("/content/ai420-ga-feature-selection/data/tb_data/TB_Chest_Radiography_Database")
CACHE_FILE   = Path("/content/ai420-ga-feature-selection/data/tb_features.npz")


# ── Feature extraction ───────────────────────────────────────────────────────

def _extract_features_from_image(img_array):
    """
    Extract a 36-element feature vector from one image (numpy array, RGB).

    Features (36 total):
      - 8  HOG orientation bins (via manual gradient histogram)
      - 24 HSV colour histogram (8 bins per channel: H, S, V)
      - 4  basic statistics (mean R, G, B, overall std)
    """
    import cv2

    # Resize to 128x128 for better detail in X-ray images
    img = cv2.resize(img_array, (128, 128))

    # ── HOG-like gradient histogram (8 bins) ──────────────────────────────
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY).astype(np.float32)
    gx   = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    gy   = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    mag, angle = cv2.cartToPolar(gx, gy, angleInDegrees=True)
    hog_hist, _ = np.histogram(angle.ravel(), bins=8, range=(0, 360),
                                weights=mag.ravel())
    hog_hist = hog_hist / (hog_hist.sum() + 1e-8)   # normalise

    # ── HSV colour histogram (8 bins × 3 channels = 24) ──────────────────
    hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
    hsv_feats = []
    for ch in range(3):
        hist, _ = np.histogram(hsv[:, :, ch], bins=8, range=(0, 256))
        hist = hist / (hist.sum() + 1e-8)
        hsv_feats.extend(hist.tolist())

    # ── Basic statistics (4) ─────────────────────────────────────────────
    stats = [
        img[:, :, 0].mean() / 255,   # mean R
        img[:, :, 1].mean() / 255,   # mean G
        img[:, :, 2].mean() / 255,   # mean B
        img.std() / 255,             # overall std
    ]

    return np.array(hog_hist.tolist() + hsv_feats + stats, dtype=np.float32)


def extract_all_features(data_dir=DATA_DIR, max_per_class=700):
    """
    Walk the Normal / Tuberculosis folders, extract features, return X, y.
    """
    try:
        import cv2
    except ImportError:
        raise ImportError("Install opencv-python:  pip install opencv-python")

    # Updated classes for TB dataset
    classes = {"Normal": 0, "Tuberculosis": 1}
    X_list, y_list = [], []

    for class_name, label in classes.items():
        folder = data_dir / class_name
        if not folder.exists():
            print(f"Warning: Folder {folder} not found. Skipping...")
            continue
            
        # Support both .png and .jpg
        files = sorted(list(folder.glob("*.png")) + list(folder.glob("*.jpg")))[:max_per_class]
        
        print(f"  Extracting {len(files)} images from {class_name}/...")
        for fpath in files:
            img = cv2.imread(str(fpath))
            if img is None:
                continue
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            feats = _extract_features_from_image(img)
            X_list.append(feats)
            y_list.append(label)

    if not X_list:
        raise ValueError(f"No images found in {data_dir}. Please check your path and unzip process.")

    X = np.array(X_list, dtype=np.float32)
    y = np.array(y_list, dtype=np.int32)
    print(f"Feature matrix: {X.shape}  |  labels: {y.shape}")
    return X, y


# ── Public API ───────────────────────────────────────────────────────────────

def load_data():
    if CACHE_FILE.exists():
        print(f"Loading cached features from {CACHE_FILE} ...")
        npz = np.load(CACHE_FILE)
        X, y = npz["X"], npz["y"]
    else:
        print("Cache not found. Extracting features from images...")
        os.makedirs(CACHE_FILE.parent, exist_ok=True)
        X, y = extract_all_features()
        np.savez(CACHE_FILE, X=X, y=y)
        print(f"Features cached to {CACHE_FILE}")

    feature_names = (
        [f"hog_bin_{i}"  for i in range(8)] +
        [f"hue_bin_{i}"  for i in range(8)] +
        [f"sat_bin_{i}"  for i in range(8)] +
        [f"val_bin_{i}"  for i in range(8)] +
        ["mean_R", "mean_G", "mean_B", "std_all"]
    )

    print(f"Dataset: {X.shape[0]} samples | {X.shape[1]} features | "
          f"classes: {np.bincount(y)}")
    return X, y, feature_names


def preprocess(X, y):
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y
    )

    if NORMALIZE:
        scaler  = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test  = scaler.transform(X_test)

    print(f"Train: {X_train.shape} | Test: {X_test.shape}")
    return X_train, X_test, y_train, y_test


def get_baseline_accuracy(X_train, X_test, y_train, y_test):
    model = KNeighborsClassifier(n_neighbors=KNN_NEIGHBORS,n_jobs=-1)
    model.fit(X_train, y_train)
    acc = accuracy_score(y_test, model.predict(X_test))
    print(f"\nBaseline accuracy (all {X_train.shape[1]} features): {acc:.4f}")
    print(classification_report(y_test, model.predict(X_test),
                                 target_names=["Normal", "Tuberculosis"]))
    return acc


def evaluate_subset(X_train, X_test, y_train, y_test, indices):
    if len(indices) == 0:
        return 0.0
    model = KNeighborsClassifier(n_neighbors=KNN_NEIGHBORS,n_jobs=-1)
    model.fit(X_train[:, indices], y_train)
    return model.score(X_test[:, indices], y_test)


def run_eda(X, y, feature_names):
    os.makedirs("results/plots", exist_ok=True)
    df = pd.DataFrame(X, columns=feature_names)
    plt.figure(figsize=(14, 12))
    sns.heatmap(df.corr(), cmap="coolwarm", center=0, linewidths=0.3)
    plt.title("Chest X-Ray features — correlation heatmap")
    plt.tight_layout()
    plt.savefig("results/plots/eda_correlation.png", dpi=150)
    plt.close()
    print("Saved: results/plots/eda_correlation.png")


if __name__ == "__main__":
    X, y, names = load_data()
    run_eda(X, y, names)
    X_train, X_test, y_train, y_test = preprocess(X, y)
    get_baseline_accuracy(X_train, X_test, y_train, y_test)
