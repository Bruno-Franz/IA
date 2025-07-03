# %%
# Este arquivo faz parte do legado do projeto e não é mais mantido.
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
"""Script principal que executa treinamentos nos três conjuntos de dados."""
import zipfile
import subprocess
from pathlib import Path
from typing import List, Dict, Mapping

# %%
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import (
    classification_report,
    accuracy_score,
    precision_recall_fscore_support,
)
from notebook_loader import load_notebook_as_module


_modelos = load_notebook_as_module("modelos_neurais.ipynb")
treinar_mlp_keras = _modelos.treinar_mlp_keras
treinar_cnn_texto = _modelos.treinar_cnn_texto
treinar_cnn_lstm_texto = _modelos.treinar_cnn_lstm_texto
treinar_gru_texto = _modelos.treinar_gru_texto
treinar_cnn_profundo = _modelos.treinar_cnn_profundo
import time

# %%
# List to accumulate results for each training block
results_dt = []

# %%
# Optional imports for the image dataset
try:
    import tensorflow as tf
    from tensorflow import keras
    from tensorflow.keras import layers
    import tensorflow_datasets as tfds
except Exception:
    tf = None
    tfds = None


# %% [markdown]
# ---------------- Bank Marketing Dataset -----------------

# %%
def baixar_base_banco():
    """Baixar e extrair o dataset de marketing bancário se necessário."""
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


# %%
def preprocessar_banco(csv_file: Path):
    # prepara o dataset bancário
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


# %%
def treinar_banco(X_train, X_test, y_train, y_test):
    # árvore de decisão no dataset bancário
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


# %%
def treinar_banco_mlp(X_train, X_test, y_train, y_test):
    """Train scikit-learn MLPClassifier on the bank dataset."""
    # MLP do scikit-learn
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


# %%
def treinar_banco_keras_mlp(X_train, X_test, y_train, y_test):
    # MLP implementado em Keras
    metrics = treinar_mlp_keras(X_train, X_test, y_train, y_test, epochs=50)
    results_dt.append(
        {
            "dataset": "Bank Marketing",
            "method": metrics["method"],
            "hyperparameters": {"epochs": 50},
            "accuracy": metrics["accuracy"],
            "precision": metrics.get("precision"),
            "recall": metrics.get("recall"),
            "f1": metrics.get("f1"),
            "duration": metrics["duration"],
        }
    )

# %% [markdown]
# ---------------- Books Reviews Dataset -----------------

# %%
def carregar_base_livros(path: str = "books_reviews.csv"):
    """Load the Books Reviews dataset from ``path``.

    The original script expected the CSV inside an ``archive`` folder,
    but the repository already ships ``books_reviews.csv`` at the project
    root. The default path was adjusted so ``executar_tudo()`` works out of the
    box without additional configuration.
    """
    # carrega o CSV de resenhas
    return pd.read_csv(path)


# %%
def preprocessar_livros(df: pd.DataFrame):
    # gera representações TF-IDF
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


# %%
def preprocessar_livros_seq(df: pd.DataFrame, num_words: int = 10000,
                            max_len: int = 200):
    """Tokenize text reviews for neural models."""
    # produz sequências de tokens
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


# %%
def treinar_livros(X_train, X_test, y_train, y_test):
    # regressão logística nas resenhas
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


# %%
def treinar_livros_cnn(train_seq, test_seq, y_train, y_test, vocab_size):
    # CNN simples para texto
    metrics = treinar_cnn_texto(train_seq, test_seq, y_train, y_test, vocab_size)
    results_dt.append(
        {
            "dataset": "Books Reviews",
            "method": metrics["method"],
            "hyperparameters": {"vocab_size": vocab_size},
            "accuracy": metrics["accuracy"],
            "precision": metrics.get("precision"),
            "recall": metrics.get("recall"),
            "f1": metrics.get("f1"),
            "duration": metrics["duration"],
        }
    )


# %%
def treinar_livros_cnn_lstm(train_seq, test_seq, y_train, y_test, vocab_size):
    # CNN combinada com LSTM
    metrics = treinar_cnn_lstm_texto(train_seq, test_seq, y_train, y_test, vocab_size)
    results_dt.append(
        {
            "dataset": "Books Reviews",
            "method": metrics["method"],
            "hyperparameters": {"vocab_size": vocab_size},
            "accuracy": metrics["accuracy"],
            "precision": metrics.get("precision"),
            "recall": metrics.get("recall"),
            "f1": metrics.get("f1"),
            "duration": metrics["duration"],
        }
    )


# %% [markdown]
# ---------------- TF Flowers Dataset -----------------

# %%
def carregar_tf_flores():
    # carrega dataset de flores do TensorFlow
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


# %%
def preprocessar_tf_flores(ds_train, ds_val, ds_test):
    # normaliza e redimensiona imagens
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


# %%
def treinar_flores(ds_train, ds_val, ds_test, num_classes):
    # CNN simples para o dataset de flores
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


# %%
def treinar_flores_profundo(ds_train, ds_val, ds_test, num_classes):
    # versão mais profunda da CNN
    metrics = treinar_cnn_profundo(ds_train, ds_val, ds_test, num_classes)
    results_dt.append(
        {
            "dataset": "TF Flowers",
            "method": metrics["method"],
            "hyperparameters": {"epochs": 20},
            "accuracy": metrics["accuracy"],
            "precision": metrics.get("precision"),
            "recall": metrics.get("recall"),
            "f1": metrics.get("f1"),
            "duration": metrics["duration"],
        }
    )


# %% [markdown]
# ---------------- Main Script -----------------

# %%
def executar_tudo():
    # orquestra todos os experimentos
    global results_dt
    results_dt = []
    # Bank Marketing
    csv_file = baixar_base_banco()
    if csv_file is not None:
        X_train, X_test, y_train, y_test = preprocessar_banco(csv_file)
        treinar_banco(X_train, X_test, y_train, y_test)
        treinar_banco_mlp(X_train, X_test, y_train, y_test)
        treinar_banco_keras_mlp(X_train, X_test, y_train, y_test)

    # Books Reviews
    books_df = carregar_base_livros()
    X_train, X_test, y_train, y_test = preprocessar_livros(books_df)
    treinar_livros(X_train, X_test, y_train, y_test)
    train_seq, test_seq, y_train_seq, y_test_seq, vocab = preprocessar_livros_seq(books_df)
    treinar_livros_cnn(train_seq, test_seq, y_train_seq, y_test_seq, vocab)
    treinar_livros_cnn_lstm(train_seq, test_seq, y_train_seq, y_test_seq, vocab)

    # TF Flowers
    flowers = carregar_tf_flores()
    if flowers is not None and tf is not None:
        (ds_train, ds_val, ds_test), info = flowers
        ds_train, ds_val, ds_test = preprocessar_tf_flores(ds_train, ds_val, ds_test)
        treinar_flores(ds_train, ds_val, ds_test, info.features["label"].num_classes)
        treinar_flores_profundo(ds_train, ds_val, ds_test, info.features["label"].num_classes)

    # Consolidate results and export to CSV
    df = pd.DataFrame(results_dt)
    df.to_csv("results.csv", index=False)
    return df


# %%
def avaliar_dataset(df: pd.DataFrame, target: str) -> List[Dict[str, float]]:
    """Train simple models on *df* and return metrics."""
    X = df.drop(columns=[target])
    y = df[target]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    models = {
        "Logistic Regression": LogisticRegression(max_iter=1000),
        "Decision Tree": DecisionTreeClassifier(random_state=42),
        "GaussianNB": GaussianNB(),
    }

    results = []
    for name, model in models.items():
        model.fit(X_train, y_train)
        preds = model.predict(X_test)
        acc = accuracy_score(y_test, preds)
        precision, recall, f1, _ = precision_recall_fscore_support(
            y_test, preds, average="weighted", zero_division=0
        )
        results.append(
            {
                "method": name,
                "accuracy": acc,
                "precision": precision,
                "recall": recall,
                "f1": f1,
            }
        )
    return results


# %%
def agregar_resultados(csv_path: str) -> Mapping[str, pd.DataFrame]:
    """Group ``results.csv`` by dataset and method and average metrics."""
    df = pd.read_csv(csv_path)
    metrics = ["accuracy", "precision", "recall", "f1", "duration"]
    grouped = df.groupby(["dataset", "method"])[metrics].mean().reset_index()

    result: Dict[str, pd.DataFrame] = {}
    for dataset, table in grouped.groupby("dataset"):
        result[dataset] = table.drop(columns="dataset").reset_index(drop=True)
    return result


# %%
def gerar_tabelas(csv_path: str, out_dir: str = ".", prefix: str = "table") -> None:
    """Create per-dataset tables in CSV and Markdown format."""
    tables = agregar_resultados(csv_path)
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    for dataset, table in tables.items():
        slug = dataset.lower().replace(" ", "_")
        table.to_csv(out / f"{prefix}_{slug}.csv", index=False)
        try:
            markdown = table.to_markdown(index=False)
        except ImportError:
            markdown = table.to_csv(index=False)
        with open(out / f"{prefix}_{slug}.md", "w", encoding="utf-8") as fh:
            fh.write(markdown)


# %%
if __name__ == "__main__":
    executar_tudo()
