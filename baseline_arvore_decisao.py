# %%
from __future__ import annotations
try:
    from google.colab import drive
    drive.mount('/content/drive')
    import sys, pathlib
    project_root = pathlib.Path('/content/drive/MyDrive/IA')
    sys.path.append(str(project_root))
except ModuleNotFoundError:
    import sys, pathlib
    sys.path.append(str(pathlib.Path().resolve()))

# %%
"""Baseline de Árvores de Decisão para os conjuntos Bank, Books e Flowers."""

# %%
import numpy as np
import time
from pathlib import Path
from typing import List, Dict, Tuple

# %%
import pandas as pd
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.feature_extraction.text import TfidfVectorizer

# %%
try:
    import tensorflow as tf
    import tensorflow_datasets as tfds
except Exception:  # TensorFlow might not be installed
    tf = None
    tfds = None


# %% [markdown]
# --------------------- Bank Marketing ---------------------

# %%
BANK_URL = "https://archive.ics.uci.edu/ml/machine-learning-databases/00222/bank.zip"


# %%
def baixar_base_banco() -> Path:
    """Baixar o dataset bancário se necessário e retornar o caminho do CSV."""
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


# %%
def preprocessar_banco(csv_file: Path) -> Tuple[pd.DataFrame, pd.Series]:
    # carrega o CSV e normaliza colunas
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


# %% [markdown]
# --------------------- Books Reviews ---------------------

# %%
BOOKS_PATH = Path("books_reviews.csv")


# %%
def preprocessar_livros() -> Tuple[pd.DataFrame, pd.Series]:
    # prepara as resenhas de livros
    df = pd.read_csv(BOOKS_PATH)
    df = df.dropna(subset=["review_text", "label"]).copy()
    y = df["label"]

    vectorizer = TfidfVectorizer(max_features=5000)
    X = vectorizer.fit_transform(df["review_text"].astype(str))
    return X, y


# %% [markdown]
# --------------------- Flowers Recognition ---------------------


# %%
def preprocessar_flores() -> Tuple[pd.DataFrame, pd.Series]:
    # carrega o dataset de flores e extrai histogramas
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


# %% [markdown]
# --------------------- Evaluation ---------------------

# %%
PARAMS_LIST = [
    {"criterion": "gini", "max_depth": None},
    {"criterion": "entropy", "max_depth": 5},
    {"criterion": "gini", "max_depth": 10, "min_samples_split": 10},
]


# %%
def avaliar_dataset(X, y, dataset_name: str) -> List[Dict[str, object]]:
    # separa dados, treina árvore e avalia
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    results = []
    for i, params in enumerate(PARAMS_LIST, 1):
        start = time.time()
        clf = DecisionTreeClassifier(random_state=42, **params)
        clf.fit(X_train, y_train)
        preds = clf.predict(X_test)
        duration = time.time() - start
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
                "Duration": duration,
            }
        )
    return results


# %% [markdown]
# --------------------- Main ---------------------

# %%
results_dt: List[Dict[str, object]] = []


# %%
def executar_tudo() -> pd.DataFrame:
    # executa o pipeline completo
    global results_dt
    results_dt = []

    # Bank
    bank_csv = baixar_base_banco()
    X_bank, y_bank = preprocessar_banco(bank_csv)
    results_dt.extend(avaliar_dataset(X_bank, y_bank, "Bank"))

    # Books
    X_books, y_books = preprocessar_livros()
    results_dt.extend(avaliar_dataset(X_books, y_books, "Books"))

    # Flowers
    try:
        X_flowers, y_flowers = preprocessar_flores()
    except Exception as exc:
        print(f"Skipping flowers dataset due to error: {exc}")
    else:
        results_dt.extend(avaliar_dataset(X_flowers, y_flowers, "Flowers"))

    df = pd.DataFrame(results_dt)
    df.to_csv("results.csv", index=False)
    return df


# %%
if __name__ == "__main__":
    df = executar_tudo()
    print(df)
