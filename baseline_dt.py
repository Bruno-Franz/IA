# Baseline Decision Tree classifiers for Bank, Flowers, and Books datasets
# Implements simple preprocessing and evaluation for each dataset

from __future__ import annotations

import numpy as np
from pathlib import Path
from typing import List, Dict, Tuple

import pandas as pd
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.feature_extraction.text import TfidfVectorizer

try:
    import tensorflow as tf
    import tensorflow_datasets as tfds
except Exception:  # TensorFlow might not be installed
    tf = None
    tfds = None


# --------------------- Bank Marketing ---------------------

BANK_URL = "https://archive.ics.uci.edu/ml/machine-learning-databases/00222/bank.zip"


def load_bank_dataset() -> Path:
    """Download bank dataset if necessary and return CSV path."""
    csv_path = Path("bank-full.csv")
    if csv_path.exists():
        return csv_path

    import zipfile
    import subprocess

    zip_path = Path("bank.zip")
    subprocess.run(["wget", "-q", BANK_URL, "-O", str(zip_path)], check=True)
    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extract("bank-full.csv")
    return csv_path


def preprocess_bank(csv_file: Path) -> Tuple[pd.DataFrame, pd.Series]:
    df = pd.read_csv(csv_file, sep=";")
    binary_map = {"yes": 1, "no": 0}
    for col in ["default", "housing", "loan", "y"]:
        df[col] = df[col].map(binary_map)

    categorical_cols = ["job", "marital", "education", "contact", "month", "poutcome"]
    df = pd.get_dummies(df, columns=categorical_cols, drop_first=True)

    X = df.drop("y", axis=1)
    y = df["y"]

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    return X_scaled, y


# --------------------- Books Reviews ---------------------

BOOKS_PATH = Path("books_reviews.csv")


def preprocess_books() -> Tuple[pd.DataFrame, pd.Series]:
    df = pd.read_csv(BOOKS_PATH)
    df = df.dropna(subset=["review_text", "label"]).copy()
    y = df["label"]

    vectorizer = TfidfVectorizer(max_features=5000)
    X = vectorizer.fit_transform(df["review_text"].astype(str))
    return X, y


# --------------------- Flowers Recognition ---------------------


def preprocess_flowers() -> Tuple[pd.DataFrame, pd.Series]:
    if tfds is None:
        raise RuntimeError("TensorFlow Datasets not available")

    (ds_train, ds_val, ds_test), info = tfds.load(
        "tf_flowers",
        split=["train[:80%]", "train[80%:90%]", "train[90%:]"],
        as_supervised=True,
        with_info=True,
    )

    # combine splits
    ds = ds_train.concatenate(ds_val).concatenate(ds_test)

    images = []
    labels = []
    for img, label in tfds.as_numpy(ds):
        # Resize to smaller size for speed
        img_resized = tf.image.resize(img, (64, 64)).numpy().astype("uint8")
        # Compute color histogram (16 bins per channel -> 48 features)
        hist = []
        for i in range(3):
            h, _ = np.histogram(img_resized[:, :, i], bins=16, range=(0, 255))
            hist.extend(h)
        images.append(hist)
        labels.append(label)

    X = pd.DataFrame(images)
    y = pd.Series(labels)
    return X, y


# --------------------- Evaluation ---------------------

PARAMS_LIST = [
    {"criterion": "gini", "max_depth": None},
    {"criterion": "entropy", "max_depth": 5},
    {"criterion": "gini", "max_depth": 10, "min_samples_split": 10},
]


def evaluate_dataset(X, y, dataset_name: str) -> List[Dict[str, object]]:
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    results = []
    for i, params in enumerate(PARAMS_LIST, 1):
        clf = DecisionTreeClassifier(random_state=42, **params)
        clf.fit(X_train, y_train)
        preds = clf.predict(X_test)
        report = classification_report(y_test, preds, output_dict=True, zero_division=0)
        results.append(
            {
                "Dataset": dataset_name,
                "Config": f"Config {i}",
                "Params": params,
                "Accuracy": accuracy_score(y_test, preds),
                "Precision": report["weighted avg"]["precision"],
                "Recall": report["weighted avg"]["recall"],
                "F1": report["weighted avg"]["f1-score"],
            }
        )
    return results


# --------------------- Main ---------------------

results_dt: List[Dict[str, object]] = []


def run_all() -> List[Dict[str, object]]:
    # Bank
    bank_csv = load_bank_dataset()
    X_bank, y_bank = preprocess_bank(bank_csv)
    results_dt.extend(evaluate_dataset(X_bank, y_bank, "Bank"))

    # Books
    X_books, y_books = preprocess_books()
    results_dt.extend(evaluate_dataset(X_books, y_books, "Books"))

    # Flowers
    try:
        X_flowers, y_flowers = preprocess_flowers()
    except Exception as exc:
        print(f"Skipping flowers dataset due to error: {exc}")
    else:
        results_dt.extend(evaluate_dataset(X_flowers, y_flowers, "Flowers"))

    return results_dt


if __name__ == "__main__":
    run_all()
    for r in results_dt:
        print(r)
