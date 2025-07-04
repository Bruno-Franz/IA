# %%
"""Funções para avaliar e agregar resultados de modelos."""
from __future__ import annotations

# %%
from typing import List, Dict, Mapping

# %%
from pathlib import Path

# %%
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, precision_recall_fscore_support


# %%
def avaliar_dataset(df: pd.DataFrame, target: str) -> List[Dict[str, float]]:
    """Train simple models on *df* and return metrics.

    Parameters
    ----------
    df : pandas.DataFrame
        DataFrame containing features and the target column.
    target : str
        Name of the target column in ``df``.

    Returns
    -------
    List[Dict[str, float]]
        List with one metrics dictionary per model. Each dictionary contains the
        keys ``method``, ``accuracy``, ``precision``, ``recall`` and ``f1``.
    """
    # separa dados e treina modelos simples
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
    """Group ``results.csv`` by dataset and method and average metrics.

    Parameters
    ----------
    csv_path : str
        Path to the CSV generated by the training scripts.

    Returns
    -------
    Mapping[str, pandas.DataFrame]
        A dictionary mapping each dataset name to a table with the aggregated
        metrics for every method tested.
    """
    # agrupa métricas por dataset e método
    df = pd.read_csv(csv_path)
    metrics = ["accuracy", "precision", "recall", "f1", "duration"]
    grouped = df.groupby(["dataset", "method"])[metrics].mean().reset_index()

    result: Dict[str, pd.DataFrame] = {}
    for dataset, table in grouped.groupby("dataset"):
        result[dataset] = table.drop(columns="dataset").reset_index(drop=True)
    return result


# %%
def gerar_tabelas(csv_path: str, out_dir: str = ".", prefix: str = "table") -> None:
    """Create per-dataset tables in CSV and Markdown format.

    Parameters
    ----------
    csv_path : str
        Source results CSV.
    out_dir : str, optional
        Directory where the tables will be written.
    prefix : str, optional
        Prefix used for the output filenames.
    """
    # gera arquivos com tabelas sumarizadas
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
    import argparse

    parser = argparse.ArgumentParser(description="Aggregate training results")
    parser.add_argument("csv", nargs="?", default="results.csv")
    parser.add_argument("--out-dir", default=".")
    parser.add_argument("--prefix", default="table")
    args = parser.parse_args()

    gerar_tabelas(args.csv, out_dir=args.out_dir, prefix=args.prefix)
