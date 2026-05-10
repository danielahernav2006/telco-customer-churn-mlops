"""
Servicio de inferencia FastAPI para predicción de churn en
telecomunicaciones.

Este módulo expone una API REST que carga el modelo entrenado
(Pipeline de scikit-learn con CatBoost) y responde a solicitudes
JSON con la probabilidad de churn y la predicción final.

Endpoints:
- GET  /         : información básica del servicio.
- GET  /health   : estado del servicio y del modelo cargado.
- POST /predict  : recibe los datos de un cliente y devuelve
                   la probabilidad y la predicción interpretable.
"""

import logging
from pathlib import Path

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException

from app.schemas import CustomerData, HealthResponse, PredictionResponse

# ---------------------------------------------------------------------------
# Configuración de logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger("telco-churn-api")

# ---------------------------------------------------------------------------
# Constantes
# ---------------------------------------------------------------------------
# Ruta robusta basada en la ubicación de este archivo. Esto evita
# problemas si la API se ejecuta desde otro directorio de trabajo.
APP_DIR = Path(__file__).resolve().parent
MODEL_PATH = APP_DIR / "model.joblib"

# Umbral de decisión para clasificar como churn.
THRESHOLD = 0.5

# Nombre del modelo final seleccionado en la fase de modelado.
MODEL_NAME = "CatBoost"

# Orden exacto de columnas que espera el Pipeline (mismo orden con
# que se entrenó en el notebook). El Pipeline en sí no exige un
# orden particular, pero mantenerlo es una buena práctica.
EXPECTED_COLUMNS = [
    "gender",
    "SeniorCitizen",
    "Partner",
    "Dependents",
    "tenure",
    "PhoneService",
    "MultipleLines",
    "InternetService",
    "OnlineSecurity",
    "OnlineBackup",
    "DeviceProtection",
    "TechSupport",
    "StreamingTV",
    "StreamingMovies",
    "Contract",
    "PaperlessBilling",
    "PaymentMethod",
    "MonthlyCharges",
    "TotalCharges",
]


# ---------------------------------------------------------------------------
# Carga del modelo
# ---------------------------------------------------------------------------
def load_model(model_path: Path = MODEL_PATH):
    """
    Carga el modelo serializado con joblib.

    Devuelve el objeto cargado o None si el archivo no existe o
    falla la carga. Cualquier error queda registrado para
    diagnóstico.
    """
    if not model_path.exists():
        logger.error("No se encontró el modelo en %s", model_path)
        return None
    try:
        model = joblib.load(model_path)
        logger.info("Modelo cargado correctamente desde %s", model_path)
        return model
    except Exception as exc:  # pragma: no cover - solo logging
        logger.exception("Error al cargar el modelo: %s", exc)
        return None


# Carga única en el arranque del módulo. Se mantiene en memoria
# durante todo el ciclo de vida del proceso uvicorn.
model = load_model()

# ---------------------------------------------------------------------------
# Aplicación FastAPI
# ---------------------------------------------------------------------------
app = FastAPI(
    title="Telco Churn Prediction API",
    description=(
        "API de inferencia para predecir abandono de clientes "
        "(churn) en una empresa de telecomunicaciones. Usa un "
        "Pipeline de scikit-learn con CatBoost como modelo final."
    ),
    version="1.0.0",
)


# ---------------------------------------------------------------------------
# Utilidades internas
# ---------------------------------------------------------------------------
def _positive_class_index(loaded_model) -> int:
    """
    Devuelve el índice de la clase positiva (churn = 1) en
    predict_proba. Si el modelo expone classes_, lo usa para
    detectarlo de forma robusta; si no, asume que la clase
    positiva está en la última posición.
    """
    classes = getattr(loaded_model, "classes_", None)
    if classes is None:
        return 1
    classes_list = list(classes)
    if 1 in classes_list:
        return classes_list.index(1)
    if "Yes" in classes_list:
        return classes_list.index("Yes")
    return len(classes_list) - 1


def _predict_proba_safe(loaded_model, df: pd.DataFrame) -> float:
    """Ejecuta predict_proba y devuelve la probabilidad de churn."""
    proba = loaded_model.predict_proba(df)
    idx = _positive_class_index(loaded_model)
    return float(proba[0, idx])


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------
@app.get("/", tags=["meta"])
def root() -> dict:
    """Información básica de la API."""
    return {
        "service": "Telco Churn Prediction API",
        "version": "1.0.0",
        "model_name": MODEL_NAME,
        "endpoints": ["/", "/health", "/predict", "/docs"],
    }


@app.get("/health", response_model=HealthResponse, tags=["meta"])
def health() -> HealthResponse:
    """Estado de salud del servicio."""
    return HealthResponse(
        status="ok" if model is not None else "model_not_loaded",
        model_loaded=model is not None,
        model_name=MODEL_NAME,
    )


@app.post("/predict", response_model=PredictionResponse, tags=["inference"])
def predict(customer: CustomerData) -> PredictionResponse:
    """
    Predice si un cliente abandonará la compañía.

    Recibe los datos del cliente (todas las variables predictoras
    del dataset Telco Customer Churn) y devuelve:
    - churn_probability: probabilidad de la clase positiva (churn).
    - prediction: 'Yes' si la probabilidad supera el umbral,
                  'No' en caso contrario.
    - threshold: umbral aplicado.
    - model_name: nombre del modelo utilizado.
    """
    if model is None:
        logger.error("Solicitud /predict recibida pero el modelo no está cargado.")
        raise HTTPException(
            status_code=503,
            detail=(
                "El modelo no está disponible. Verifique que el "
                "archivo app/model.joblib exista y sea válido."
            ),
        )

    try:
        # 1. Convertir entrada validada a DataFrame con el orden
        #    exacto de columnas que espera el Pipeline.
        payload = customer.model_dump()
        df = pd.DataFrame([payload], columns=EXPECTED_COLUMNS)

        # 2. Calcular probabilidad de churn.
        prob = _predict_proba_safe(model, df)

        # 3. Aplicar umbral para obtener la clase final.
        label = "Yes" if prob >= THRESHOLD else "No"

        return PredictionResponse(
            churn_probability=round(prob, 4),
            prediction=label,
            threshold=THRESHOLD,
            model_name=MODEL_NAME,
        )

    except HTTPException:
        raise
    except Exception as exc:
        logger.exception("Error durante la predicción: %s", exc)
        raise HTTPException(
            status_code=500,
            detail=f"Error al generar la predicción: {exc}",
        )


# ---------------------------------------------------------------------------
# Ejecución directa (útil en desarrollo: `python -m app.api`)
# ---------------------------------------------------------------------------
if __name__ == "__main__":  # pragma: no cover
    import uvicorn

    uvicorn.run("app.api:app", host="0.0.0.0", port=8000, reload=True)
