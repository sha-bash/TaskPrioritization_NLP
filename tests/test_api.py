import pytest
from fastapi.testclient import TestClient
from services.fastapi_endpoint.fastapi_app import app

# Создаем клиент FastAPI для синхронных запросов
@pytest.fixture
def client():
    return TestClient(app)

def test_status(client):
    response = client.get("/status")
    assert response.status_code == 200  
    assert response.json() == {"status": "API is running"}  

def test_all_predict(client):
    payload = {"text": "Пример текста для анализа"}
    response = client.post("/v1/all_predict", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert "logistic_regression_prediction" in data
    assert "svm_prediction" in data
    assert "random_forest_prediction" in data
    assert "CNN_prediction" in data

def test_logistic_regression_prediction(client):
    payload = {"text": "Пример текста для анализа"}
    response = client.post("/v1/logistic_regression_prediction", json=payload)
    assert response.status_code == 200
    assert "logistic_regression_prediction" in response.json()

def test_svm_prediction(client):
    payload = {"text": "Пример текста для анализа"}
    response = client.post("/v1/svm_prediction", json=payload)
    assert response.status_code == 200
    assert "svm_prediction" in response.json()

def test_random_forest_prediction(client):
    payload = {"text": "Пример текста для анализа"}
    response = client.post("/v1/random_forest_prediction", json=payload)
    assert response.status_code == 200
    assert "random_forest_prediction" in response.json()

def test_predict(client):
    payload = {"text": "Пример текста для анализа"}
    response = client.post("/v1/predict", json=payload)
    assert response.status_code == 200
    assert "predict" in response.json()

def test_cnn_prediction(client):
    payload = {"text": "Пример текста для анализа"}
    response = client.post("/v1/CNN_prediction", json=payload)
    assert response.status_code == 200
    assert "CNN_prediction" in response.json()

def test_bert_prediction(client):
    payload = {"text": "Пример текста для анализа"}
    response = client.post("/v1/bert_prediction", json=payload)
    assert response.status_code == 200
    assert "BERT_predict" in response.json()
