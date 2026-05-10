"""
Pruebas unitarias del modelo serializado.

Verifican que:
1. El archivo app/model.joblib existe.
2. El modelo se puede cargar con joblib.
3. El modelo tiene los métodos predict y predict_proba.
4. El modelo puede generar una predicción válida con un ejemplo
   con todas las variables esperadas.
"""

from pathlib import Path

import joblib
import pandas as pd
import pytest


# Ruta robusta al modelo, calculada respecto a este archivo.
PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODEL_PATH = PROJECT_ROOT / "app" / "model.joblib"


# Ejemplo con las 19 variables predictoras del dataset Telco.
SAMPLE_CUSTOMER = {
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


def test_model_file_exists():
    """El archivo app/model.joblib debe existir."""
    assert MODEL_PATH.exists(), (
        f"No se encontró el modelo en {MODEL_PATH}. "
        "Asegúrese de haber entrenado y guardado el modelo final."
    )


def test_model_can_be_loaded():
    """El modelo debe poder cargarse con joblib sin errores."""
    model = joblib.load(MODEL_PATH)
    assert model is not None


def test_model_has_predict_methods():
    """El modelo debe exponer predict y predict_proba."""
    model = joblib.load(MODEL_PATH)
    assert hasattr(model, "predict"), "El modelo no tiene método predict."
    assert hasattr(model, "predict_proba"), (
        "El modelo no tiene método predict_proba."
    )


def test_model_predicts_on_valid_sample():
    """
    El modelo debe poder generar una predicción y una probabilidad
    válidas para un cliente de ejemplo.
    """
    model = joblib.load(MODEL_PATH)
    df = pd.DataFrame([SAMPLE_CUSTOMER])

    pred = model.predict(df)
    assert len(pred) == 1
    assert pred[0] in (0, 1, "Yes", "No"), (
        f"La predicción debe ser binaria, recibido: {pred[0]}"
    )

    proba = model.predict_proba(df)
    assert proba.shape == (1, 2)
    # Cada fila de probabilidades debe sumar (aprox.) 1.
    assert pytest.approx(proba.sum(), rel=1e-6) == 1.0
    # Cada probabilidad debe estar en [0, 1].
    assert (proba >= 0).all() and (proba <= 1).all()


def test_model_classes_are_binary():
    """El modelo debe tener exactamente dos clases."""
    model = joblib.load(MODEL_PATH)
    assert hasattr(model, "classes_")
    assert len(model.classes_) == 2
