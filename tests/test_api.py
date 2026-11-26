import pandas as pd
import numpy as np
from fastapi.testclient import TestClient
import main

client = TestClient(main.app)


class FakeModel:
    def __init__(self, probability):
        self.probability = probability

    def predict_proba(self, X):
        return np.array([[1 - self.probability, self.probability]])


def make_transaction():
    required_cols = (
        set(main.freq_maps.keys())
        | set(main.l_enc.keys())
        | {'card1', 'addr1', 'P_emaildomain'}
    )

    for group_column, mapping_data in main.agg_maps.items():
      for main_column, agg_type in mapping_data["columns"]:
        required_cols.add(main_column)
        required_cols.add(group_column)

    required_cols -= {
        'card1_addr1',
        'card1_addr1_P_emaildomain'
    }

    transaction = {}

    for col in required_cols:
        if main.feature_dtype.get(col) == 'float64':
            transaction[col] = 0.0
        else:
            transaction[col] = "test"

    transaction['TransactionAmt'] = 100.0

    return transaction


def prepare_mocked_api(monkeypatch, probability):
    def fake_preprocess(*args):
        return pd.DataFrame(
            [[0] * len(main.feature_colum)],
            columns=main.feature_colum
        )

    monkeypatch.setattr(main, 'preprocess_data', fake_preprocess)
    monkeypatch.setattr(main, 'model', FakeModel(probability))


def test_empty_transaction():
    response = client.post("/predict", json={})

    assert response.status_code == 400
    assert response.json()["detail"] == "Transaction data cannot be empty."


def test_missing_required_fields():
    response = client.post("/predict", json={"card1": 1234})

    assert response.status_code == 400
    assert "Missing required fields" in response.json()["detail"]


def test_invalid_transaction_data():
    transaction = make_transaction()
    transaction['TransactionAmt'] = "hello"

    response = client.post("/predict", json=transaction)

    assert response.status_code == 400
    assert "Invalid transaction data" in response.json()["detail"]


def test_legitimate_transaction(monkeypatch):
    prepare_mocked_api(monkeypatch, 0.10)

    transaction = make_transaction()
    response = client.post("/predict", json=transaction)

    assert response.status_code == 200
    assert response.json()['prediction'] == 0
    assert response.json()['result'] == "legitimate"
    assert response.json()['fraud_probability'] == 0.10


def test_fraudulent_transaction(monkeypatch):
    prepare_mocked_api(monkeypatch, 0.90)

    transaction = make_transaction()
    response = client.post("/predict", json=transaction)

    assert response.status_code == 200
    assert response.json()['prediction'] == 1
    assert response.json()['result'] == "fraudulent"
    assert response.json()['fraud_probability'] == 0.90
