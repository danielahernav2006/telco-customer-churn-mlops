"""
Esquemas Pydantic para validación de entrada y salida de la API.

Define el contrato de datos del endpoint /predict:
- CustomerData: representa todas las variables predictoras de un cliente
  (las 19 columnas del dataset Telco Customer Churn, excluyendo
  customerID y la variable objetivo Churn).
- PredictionResponse: respuesta estándar del endpoint con la
  probabilidad de churn y la predicción interpretable.
"""

from pydantic import BaseModel, Field
from typing import Literal


class CustomerData(BaseModel):
    """
    Datos de un cliente para predicción de churn.

    Las variables siguen los valores originales del dataset
    Telco Customer Churn (categóricas como string, numéricas como
    int/float). El modelo es un Pipeline que ya incluye el
    preprocesamiento (imputación, escalado y OneHotEncoder), por lo
    que no se requiere transformación manual previa.
    """

    # Variables demográficas
    gender: Literal["Female", "Male"] = Field(
        ..., description="Género del cliente."
    )
    SeniorCitizen: int = Field(
        ..., ge=0, le=1,
        description="1 si es adulto mayor, 0 en caso contrario."
    )
    Partner: Literal["Yes", "No"] = Field(
        ..., description="Indica si el cliente tiene pareja."
    )
    Dependents: Literal["Yes", "No"] = Field(
        ..., description="Indica si el cliente tiene dependientes económicos."
    )

    # Antigüedad y servicio telefónico
    tenure: int = Field(
        ..., ge=0, description="Meses que el cliente lleva contratado."
    )
    PhoneService: Literal["Yes", "No"] = Field(
        ..., description="Indica si el cliente tiene servicio telefónico."
    )
    MultipleLines: Literal["Yes", "No", "No phone service"] = Field(
        ..., description="Indica si el cliente tiene múltiples líneas."
    )

    # Servicios de internet
    InternetService: Literal["DSL", "Fiber optic", "No"] = Field(
        ..., description="Tipo de servicio de internet."
    )
    OnlineSecurity: Literal["Yes", "No", "No internet service"] = Field(
        ..., description="Indica si tiene servicio de seguridad en línea."
    )
    OnlineBackup: Literal["Yes", "No", "No internet service"] = Field(
        ..., description="Indica si tiene servicio de respaldo en línea."
    )
    DeviceProtection: Literal["Yes", "No", "No internet service"] = Field(
        ..., description="Indica si tiene protección de dispositivos."
    )
    TechSupport: Literal["Yes", "No", "No internet service"] = Field(
        ..., description="Indica si tiene soporte técnico contratado."
    )
    StreamingTV: Literal["Yes", "No", "No internet service"] = Field(
        ..., description="Indica si tiene servicio de streaming de TV."
    )
    StreamingMovies: Literal["Yes", "No", "No internet service"] = Field(
        ..., description="Indica si tiene servicio de streaming de películas."
    )

    # Información contractual y de facturación
    Contract: Literal["Month-to-month", "One year", "Two year"] = Field(
        ..., description="Tipo de contrato del cliente."
    )
    PaperlessBilling: Literal["Yes", "No"] = Field(
        ..., description="Indica si la facturación es electrónica."
    )
    PaymentMethod: Literal[
        "Electronic check",
        "Mailed check",
        "Bank transfer (automatic)",
        "Credit card (automatic)",
    ] = Field(..., description="Método de pago utilizado.")
    MonthlyCharges: float = Field(
        ..., ge=0, description="Cargo mensual del cliente."
    )
    TotalCharges: float = Field(
        ..., ge=0, description="Total acumulado pagado por el cliente."
    )

    model_config = {
        "json_schema_extra": {
            "example": {
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
        }
    }


class PredictionResponse(BaseModel):
    """Respuesta del endpoint /predict."""

    churn_probability: float = Field(
        ..., ge=0.0, le=1.0,
        description="Probabilidad de que el cliente abandone (clase positiva)."
    )
    prediction: Literal["Yes", "No"] = Field(
        ...,
        description="Predicción interpretable: 'Yes' si abandona, 'No' si no."
    )
    threshold: float = Field(
        0.5, description="Umbral aplicado para clasificar como churn."
    )
    model_name: str = Field(
        "CatBoost",
        description="Nombre del modelo utilizado para la predicción."
    )


class HealthResponse(BaseModel):
    """Respuesta del endpoint /health."""

    status: str
    model_loaded: bool
    model_name: str
