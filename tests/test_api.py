"""
Pruebas de integración para la API FastAPI.

Usan TestClient de FastAPI para invocar los endpoints sin
levantar un servidor real.
"""

import pytest
from fastapi.testclient import TestClient

from app.api import app


# Cliente de pruebas reutilizable.
client = TestClient(app)


# Cliente de ejemplo con alta probabilidad de churn (contrato
# mensual, fibra óptica, pago electrónico, baja antigüedad y sin
# servicios de seguridad/soporte).
HIGH_RISK_CUSTOMER = {
    "gender": "Female",
    "SeniorCitizen": 0,
    "Partner": "Yes",
    "Dependents": "No",
    "tenure": 5,
    "PhoneService": "Yes",
    "MultipleLines": "No",
    "InternetService": "Fiber optic",
    "OnlineSecurity": "No",
    "OnlineBackup": "No",
    "DeviceProtection": "No",
    "TechSupport": "No",
    "StreamingTV": "Yes",
    "StreamingMovies": "Yes",
    "Contract": "Month-to-month",
    "PaperlessBilling": "Yes",
    "PaymentMethod": "Electronic check",
    "MonthlyCharges": 99.9,
    "TotalCharges": 499.5,
}


# Cliente de ejemplo con baja probabilidad de churn (contrato a dos
# años, alta antigüedad, sin fibra óptica, con servicios adicionales).
LOW_RISK_CUSTOMER = {
    "gender": "Male",
    "SeniorCitizen": 0,
    "Partner": "Yes",
    "Dependents": "Yes",
    "tenure": 65,
    "PhoneService": "Yes",
    "MultipleLines": "Yes",
    "InternetService": "DSL",
    "OnlineSecurity": "Yes",
    "OnlineBackup": "Yes",
    "DeviceProtection": "Yes",
    "TechSupport": "Yes",
    "StreamingTV": "No",
    "StreamingMovies": "No",
    "Contract": "Two year",
    "PaperlessBilling": "No",
    "PaymentMethod": "Bank transfer (automatic)",
    "MonthlyCharges": 55.0,
    "TotalCharges": 3500.0,
}


# ---------------------------------------------------------------------------
# Endpoint /
# ---------------------------------------------------------------------------
def test_root_endpoint():
    """El endpoint raíz debe responder con metadatos del servicio."""
    response = client.get("/")
    assert response.status_code == 200
    body = response.json()
    assert body["service"] == "Telco Churn Prediction API"
    assert "endpoints" in body


# ---------------------------------------------------------------------------
# Endpoint /health
# ---------------------------------------------------------------------------
def test_health_endpoint_ok():
    """/health debe retornar 200 y el modelo debe estar cargado."""
    response = client.get("/health")
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "ok"
    assert body["model_loaded"] is True
    assert body["model_name"] == "CatBoost"


# ---------------------------------------------------------------------------
# Endpoint /predict
# ---------------------------------------------------------------------------
def test_predict_endpoint_returns_200():
    """/predict debe responder 200 con un payload válido."""
    response = client.post("/predict", json=HIGH_RISK_CUSTOMER)
    assert response.status_code == 200, response.text


def test_predict_response_schema():
    """La respuesta debe contener los campos esperados."""
    response = client.post("/predict", json=HIGH_RISK_CUSTOMER)
    body = response.json()
    assert "churn_probability" in body
    assert "prediction" in body
    assert "threshold" in body
    assert "model_name" in body


def test_predict_probability_range():
    """churn_probability debe estar entre 0 y 1."""
    response = client.post("/predict", json=HIGH_RISK_CUSTOMER)
    body = response.json()
    prob = body["churn_probability"]
    assert isinstance(prob, (int, float))
    assert 0.0 <= prob <= 1.0


def test_predict_label_is_yes_or_no():
    """prediction debe ser exactamente 'Yes' o 'No'."""
    response = client.post("/predict", json=HIGH_RISK_CUSTOMER)
    body = response.json()
    assert body["prediction"] in ("Yes", "No")


def test_predict_threshold_consistency():
    """
    La predicción debe ser consistente con el umbral aplicado:
    si la probabilidad >= threshold → 'Yes', si no → 'No'.
    """
    response = client.post("/predict", json=HIGH_RISK_CUSTOMER)
    body = response.json()
    threshold = body["threshold"]
    expected = "Yes" if body["churn_probability"] >= threshold else "No"
    assert body["prediction"] == expected


@pytest.mark.parametrize(
    "customer", [HIGH_RISK_CUSTOMER, LOW_RISK_CUSTOMER]
)
def test_predict_works_for_multiple_profiles(customer):
    """La API debe responder correctamente para distintos perfiles."""
    response = client.post("/predict", json=customer)
    assert response.status_code == 200
    body = response.json()
    assert 0.0 <= body["churn_probability"] <= 1.0
    assert body["prediction"] in ("Yes", "No")


# ---------------------------------------------------------------------------
# Casos de error
# ---------------------------------------------------------------------------
def test_predict_rejects_missing_fields():
    """Una solicitud incompleta debe ser rechazada con 422."""
    incomplete = {"gender": "Female", "tenure": 5}
    response = client.post("/predict", json=incomplete)
    assert response.status_code == 422


def test_predict_rejects_invalid_category():
    """Un valor categórico inválido debe ser rechazado con 422."""
    bad_payload = dict(HIGH_RISK_CUSTOMER)
    bad_payload["Contract"] = "Forever"  # valor no permitido
    response = client.post("/predict", json=bad_payload)
    assert response.status_code == 422
