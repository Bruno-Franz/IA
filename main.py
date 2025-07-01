import zipfile
import subprocess
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report, accuracy_score
from neural_models import (
    train_mlp_keras,
    train_cnn_text,
    train_cnn_lstm_text,
    train_gru_text,
    train_cnn_deep,
)
import time

# List to accumulate results for each training block
results_dt = []

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
        import urllib.request

        urllib.request.urlretrieve(url, zip_path)
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
    start = time.time()
    hyper = {"random_state": 42}
    clf = DecisionTreeClassifier(**hyper)
    clf.fit(X_train, y_train)
    preds = clf.predict(X_test)
    elapsed = time.time() - start

    print("Bank Marketing dataset results:")
    print(classification_report(y_test, preds, zero_division=0))
    acc = accuracy_score(y_test, preds)
    print("Accuracy:", acc)

    report = classification_report(y_test, preds, output_dict=True, zero_division=0)
    results_dt.append(
        {
            "dataset": "Bank Marketing",
            "method": "Decision Tree",
            "hyperparameters": hyper,
            "accuracy": acc,
            "precision": report["weighted avg"]["precision"],
            "recall": report["weighted avg"]["recall"],
            "f1": report["weighted avg"]["f1-score"],
            "duration": elapsed,
        }
    )


def train_bank_mlp(X_train, X_test, y_train, y_test):
    """Train scikit-learn MLPClassifier on the bank dataset."""
    start = time.time()
    hyper = {"hidden_layer_sizes": (100,), "max_iter": 300, "random_state": 42}
    clf = MLPClassifier(**hyper)
    clf.fit(X_train, y_train)
    preds = clf.predict(X_test)
    elapsed = time.time() - start

    print("Bank Marketing dataset results (MLP):")
    print(classification_report(y_test, preds, zero_division=0))
    acc = accuracy_score(y_test, preds)
    print("Accuracy:", acc)

    report = classification_report(y_test, preds, output_dict=True, zero_division=0)
    results_dt.append(
        {
            "dataset": "Bank Marketing",
            "method": "MLPClassifier",
            "hyperparameters": hyper,
            "accuracy": acc,
            "precision": report["weighted avg"]["precision"],
            "recall": report["weighted avg"]["recall"],
            "f1": report["weighted avg"]["f1-score"],
            "duration": elapsed,
        }
    )


def train_bank_keras_mlp(X_train, X_test, y_train, y_test):
    metrics = train_mlp_keras(X_train, X_test, y_train, y_test, epochs=50)
    results_dt.append(
        {
            "dataset": "Bank Marketing",
            "method": metrics["method"],
            "hyperparameters": {"epochs": 50},
            "accuracy": metrics["accuracy"],
            "precision": None,
            "recall": None,
            "f1": None,
            "duration": metrics["duration"],
        }
    )

# ---------------- Books Reviews Dataset -----------------

def load_books_dataset(path: str = "books_reviews.csv"):
    """Load the Books Reviews dataset from ``path``.

    The original script expected the CSV inside an ``archive`` folder,
    but the repository already ships ``books_reviews.csv`` at the project
    root. The default path was adjusted so ``run_all()`` works out of the
    box without additional configuration.
    """
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


def preprocess_books_seq(df: pd.DataFrame, num_words: int = 10000,
                         max_len: int = 200):
    """Tokenize text reviews for neural models."""
    df = df.dropna(subset=["review_text", "label"]).copy()
    X = df["review_text"].astype(str)
    y = df["label"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    tokenizer = keras.preprocessing.text.Tokenizer(num_words=num_words,
                                                   oov_token="<OOV>")
    tokenizer.fit_on_texts(X_train)
    train_seq = tokenizer.texts_to_sequences(X_train)
    test_seq = tokenizer.texts_to_sequences(X_test)

    train_seq = keras.preprocessing.sequence.pad_sequences(
        train_seq, maxlen=max_len, padding="post", truncating="post"
    )
    test_seq = keras.preprocessing.sequence.pad_sequences(
        test_seq, maxlen=max_len, padding="post", truncating="post"
    )

    vocab_size = min(num_words, len(tokenizer.word_index) + 1)
    return train_seq, test_seq, y_train, y_test, vocab_size


def train_books(X_train, X_test, y_train, y_test):
    start = time.time()
    hyper = {"max_iter": 1000}
    clf = LogisticRegression(**hyper)
    clf.fit(X_train, y_train)
    preds = clf.predict(X_test)
    elapsed = time.time() - start

    print("Books Reviews dataset results:")
    print(classification_report(y_test, preds, zero_division=0))
    acc = accuracy_score(y_test, preds)
    print("Accuracy:", acc)

    report = classification_report(y_test, preds, output_dict=True, zero_division=0)
    results_dt.append(
        {
            "dataset": "Books Reviews",
            "method": "Logistic Regression",
            "hyperparameters": hyper,
            "accuracy": acc,
            "precision": report["weighted avg"]["precision"],
            "recall": report["weighted avg"]["recall"],
            "f1": report["weighted avg"]["f1-score"],
            "duration": elapsed,
        }
    )


def train_books_cnn(train_seq, test_seq, y_train, y_test, vocab_size):
    metrics = train_cnn_text(train_seq, test_seq, y_train, y_test, vocab_size)
    results_dt.append(
        {
            "dataset": "Books Reviews",
            "method": metrics["method"],
            "hyperparameters": {"vocab_size": vocab_size},
            "accuracy": metrics["accuracy"],
            "precision": None,
            "recall": None,
            "f1": None,
            "duration": metrics["duration"],
        }
    )


def train_books_cnn_lstm(train_seq, test_seq, y_train, y_test, vocab_size):
    metrics = train_cnn_lstm_text(train_seq, test_seq, y_train, y_test, vocab_size)
    results_dt.append(
        {
            "dataset": "Books Reviews",
            "method": metrics["method"],
            "hyperparameters": {"vocab_size": vocab_size},
            "accuracy": metrics["accuracy"],
            "precision": None,
            "recall": None,
            "f1": None,
            "duration": metrics["duration"],
        }
    )


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
    start = time.time()
    hyper = {"epochs": 3}
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

    model.fit(ds_train, validation_data=ds_val, epochs=hyper["epochs"])
    y_true = []
    y_pred = []
    for batch, labels in ds_test:
        preds = model.predict(batch)
        y_pred.extend(preds.argmax(axis=1))
        y_true.extend(labels.numpy())
    elapsed = time.time() - start

    report = classification_report(y_true, y_pred, output_dict=True, zero_division=0)
    acc = accuracy_score(y_true, y_pred)

    print("TF Flowers dataset results:")
    print(classification_report(y_true, y_pred, zero_division=0))
    print("Accuracy:", acc)

    results_dt.append(
        {
            "dataset": "TF Flowers",
            "method": "Simple CNN",
            "hyperparameters": hyper,
            "accuracy": acc,
            "precision": report["weighted avg"]["precision"],
            "recall": report["weighted avg"]["recall"],
            "f1": report["weighted avg"]["f1-score"],
            "duration": elapsed,
        }
    )


def train_flowers_deep(ds_train, ds_val, ds_test, num_classes):
    metrics = train_cnn_deep(ds_train, ds_val, ds_test, num_classes)
    results_dt.append(
        {
            "dataset": "TF Flowers",
            "method": metrics["method"],
            "hyperparameters": {"epochs": 20},
            "accuracy": metrics["accuracy"],
            "precision": None,
            "recall": None,
            "f1": None,
            "duration": metrics["duration"],
        }
    )


# ---------------- Main Script -----------------

def run_all():
    global results_dt
    results_dt = []
    # Bank Marketing
    csv_file = download_bank_dataset()
    if csv_file is not None:
        X_train, X_test, y_train, y_test = preprocess_bank(csv_file)
        train_bank(X_train, X_test, y_train, y_test)
        train_bank_mlp(X_train, X_test, y_train, y_test)
        train_bank_keras_mlp(X_train, X_test, y_train, y_test)

    # Books Reviews
    books_df = load_books_dataset()
    X_train, X_test, y_train, y_test = preprocess_books(books_df)
    train_books(X_train, X_test, y_train, y_test)
    train_seq, test_seq, y_train_seq, y_test_seq, vocab = preprocess_books_seq(books_df)
    train_books_cnn(train_seq, test_seq, y_train_seq, y_test_seq, vocab)
    train_books_cnn_lstm(train_seq, test_seq, y_train_seq, y_test_seq, vocab)

    # TF Flowers
    flowers = load_tf_flowers()
    if flowers is not None and tf is not None:
        (ds_train, ds_val, ds_test), info = flowers
        ds_train, ds_val, ds_test = preprocess_tf_flowers(ds_train, ds_val, ds_test)
        train_flowers(ds_train, ds_val, ds_test, info.features["label"].num_classes)
        train_flowers_deep(ds_train, ds_val, ds_test, info.features["label"].num_classes)

    # Consolidate results and export to CSV
    df = pd.DataFrame(results_dt)
    df.to_csv("results.csv", index=False)
    return df


if __name__ == "__main__":
    run_all()
