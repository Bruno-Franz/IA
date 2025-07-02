"""Testes para a função de agregação de resultados."""
import pandas as pd
import pytest

from avaliacao import agregar_resultados


def test_agregar_resultados_grupos_e_medias(tmp_path):
    data = [
        {
            "dataset": "A",
            "method": "M1",
            "accuracy": 0.8,
            "precision": 0.7,
            "recall": 0.8,
            "f1": 0.75,
            "duration": 1.0,
        },
        {
            "dataset": "A",
            "method": "M1",
            "accuracy": 0.9,
            "precision": 0.8,
            "recall": 0.9,
            "f1": 0.85,
            "duration": 2.0,
        },
        {
            "dataset": "B",
            "method": "M2",
            "accuracy": 1.0,
            "precision": 1.0,
            "recall": 1.0,
            "f1": 1.0,
            "duration": 3.0,
        },
    ]
    df = pd.DataFrame(data)
    csv_path = tmp_path / "res.csv"
    df.to_csv(csv_path, index=False)

    results = agregar_resultados(str(csv_path))

    assert set(results.keys()) == {"A", "B"}
    a_table = results["A"]
    assert a_table.shape[0] == 1
    row = a_table.iloc[0]
    assert row["method"] == "M1"
    assert row["accuracy"] == pytest.approx(0.85)
    b_table = results["B"]
    assert b_table.iloc[0]["method"] == "M2"
