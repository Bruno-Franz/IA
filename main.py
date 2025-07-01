import os
import zipfile
import subprocess
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report, accuracy_score

# Optional imports for the image dataset
try:
    import tensorflow as tf
    from tensorflow import keras
    from tensorflow.keras import layers
    import tensorflow_datasets as tfds
except Exception:
    tf = None
    tfds = None


# ---------------- Bank Marketing Dataset -----------------

def download_bank_dataset():
    """Download and extract the bank marketing dataset if not present."""
    url = "https://archive.ics.uci.edu/ml/machine-learning-databases/00222/bank.zip"
    zip_path = Path("bank.zip")
    csv_path = Path("bank-full.csv")

    if csv_path.exists():
        return csv_path

    try:
        subprocess.run(["wget", "-q", url, "-O", str(zip_path)], check=True)
        with zipfile.ZipFile(zip_path, "r") as zf:
            zf.extract("bank-full.csv")
    except Exception as exc:
        print(f"Failed to download bank dataset: {exc}")
        return None
    return csv_path


def preprocess_bank(csv_file: Path):
    df = pd.read_csv(csv_file, sep=";")
    df = df.copy()

    binary_map = {"yes": 1, "no": 0}
    for col in ["default", "housing", "loan", "y"]:
        df[col] = df[col].map(binary_map)

    categorical_cols = [
        "job",
        "marital",
        "education",
        "contact",
        "month",
        "poutcome",
    ]
    df = pd.get_dummies(df, columns=categorical_cols, drop_first=True)

    X = df.drop("y", axis=1)
    y = df["y"]

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    return train_test_split(X_scaled, y, test_size=0.2, random_state=42)


def train_bank(X_train, X_test, y_train, y_test):
    clf = DecisionTreeClassifier(random_state=42)
    clf.fit(X_train, y_train)
    preds = clf.predict(X_test)

    print("Bank Marketing dataset results:")
    print(classification_report(y_test, preds, zero_division=0))
    print("Accuracy:", accuracy_score(y_test, preds))


# ---------------- Books Reviews Dataset -----------------

def load_books_dataset(path: str = "books_reviews.csv"):
    return pd.read_csv(path)


def preprocess_books(df: pd.DataFrame):
    df = df.dropna(subset=["review_text", "label"]).copy()
    X = df["review_text"].astype(str)
    y = df["label"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    vectorizer = TfidfVectorizer(max_features=5000)
    X_train_vec = vectorizer.fit_transform(X_train)
    X_test_vec = vectorizer.transform(X_test)

    return X_train_vec, X_test_vec, y_train, y_test


def train_books(X_train, X_test, y_train, y_test):
    clf = LogisticRegression(max_iter=1000)
    clf.fit(X_train, y_train)
    preds = clf.predict(X_test)

    print("Books Reviews dataset results:")
    print(classification_report(y_test, preds, zero_division=0))
    print("Accuracy:", accuracy_score(y_test, preds))


# ---------------- TF Flowers Dataset -----------------

def load_tf_flowers():
    if tfds is None:
        print("TensorFlow datasets is not available.")
        return None
    try:
        return tfds.load(
            "tf_flowers",
            split=["train[:80%]", "train[80%:90%]", "train[90%:]"],
            as_supervised=True,
            with_info=True,
        )
    except Exception as exc:
        print(f"Failed to load tf_flowers: {exc}")
        return None


def preprocess_tf_flowers(ds_train, ds_val, ds_test):
    IMG_SIZE = (180, 180)
    BATCH_SIZE = 32

    def resize_norm(image, label):
        image = tf.image.resize(image, IMG_SIZE)
        image = image / 255.0
        return image, label

    ds_train = (
        ds_train.map(resize_norm).batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)
    )
    ds_val = ds_val.map(resize_norm).batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)
    ds_test = ds_test.map(resize_norm).batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)

    return ds_train, ds_val, ds_test


def train_flowers(ds_train, ds_val, ds_test, num_classes):
    model = keras.Sequential(
        [
            layers.Conv2D(32, 3, activation="relu", input_shape=(180, 180, 3)),
            layers.MaxPooling2D(),
            layers.Conv2D(64, 3, activation="relu"),
            layers.MaxPooling2D(),
            layers.Flatten(),
            layers.Dense(128, activation="relu"),
            layers.Dense(num_classes, activation="softmax"),
        ]
    )

    model.compile(
        optimizer="adam",
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )

    model.fit(ds_train, validation_data=ds_val, epochs=3)
    loss, acc = model.evaluate(ds_test)
    print("TF Flowers dataset results:")
    print("Accuracy:", acc)


# ---------------- Main Script -----------------

def main():
    # Bank Marketing
    csv_file = download_bank_dataset()
    if csv_file is not None:
        X_train, X_test, y_train, y_test = preprocess_bank(csv_file)
        train_bank(X_train, X_test, y_train, y_test)

    # Books Reviews
    books_df = load_books_dataset()
    X_train, X_test, y_train, y_test = preprocess_books(books_df)
    train_books(X_train, X_test, y_train, y_test)

    # TF Flowers
    flowers = load_tf_flowers()
    if flowers is not None and tf is not None:
        (ds_train, ds_val, ds_test), info = flowers
        ds_train, ds_val, ds_test = preprocess_tf_flowers(ds_train, ds_val, ds_test)
        train_flowers(ds_train, ds_val, ds_test, info.features["label"].num_classes)


if __name__ == "__main__":
    main()
