from __future__ import annotations

from typing import List, Dict

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, precision_recall_fscore_support


def evaluate_dataset(df: pd.DataFrame, target: str) -> List[Dict[str, float]]:
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
