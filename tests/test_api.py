import json
from pathlib import Path
from fastapi.testclient import TestClient 
from main import app

client = TestClient(app)

TEST_DIR = Path(__file__).parent

def load_transaction(filename):
    file_path = TEST_DIR/filename
    with open(file_path, "r") as file:
       return json.load(file)

def test_empty_transaction():
    response = client.post("/predict", json={})
    assert response.status_code == 400
    assert response.json()["detail"] == "Transaction data cannot be empty."


def test_missing_required_fields():
    response = client.post("/predict", json={"card1": 1234})
    assert response.status_code == 400
    assert "Missing required fields" in response.json()["detail"]

def test_invalid_transaction_data():
    transaction = load_transaction('legitimate transaction.json')
    transaction['TransactionAmt'] = "hello"
    response = client.post("/predict", json=transaction)
    assert response.status_code == 400
    assert "Invalid transaction data" in response.json()['detail']

def test_legitimate_transaction():
    transaction = load_transaction('legitimate transaction.json')
    response = client.post("/predict", json=transaction)
    assert response.status_code == 200
    assert response.json()['prediction'] == 0
    assert response.json()['result'].lower() == "legitimate"

def test_fraudulent_transaction():
    transaction = load_transaction('fraudulent transaction.json')
    response = client.post("/predict", json=transaction)
    assert response.status_code == 200
    assert response.json()['prediction'] == 1
    assert response.json()['result'].lower() == "fraudulent"