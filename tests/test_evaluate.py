import pandas as pd
from sklearn.datasets import make_classification

from avaliacao import avaliar_dataset


def test_avaliar_dataset_retorna_tres_dicts_com_chaves():
    X, y = make_classification(n_samples=30, n_features=4, n_informative=2, random_state=0)
    df = pd.DataFrame(X, columns=[f"f{i}" for i in range(X.shape[1])])
    df["label"] = y

    results = avaliar_dataset(df, target="label")

    assert isinstance(results, list)
    assert len(results) == 3
    expected_keys = {"method", "accuracy", "precision", "recall", "f1"}
    for res in results:
        assert expected_keys == set(res.keys())
