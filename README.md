# Predicción de Churn en Telecomunicaciones con Machine Learning y MLOps

Proyecto académico que aplica el ciclo completo de Machine Learning —desde el análisis exploratorio hasta la puesta en producción— sobre el dataset **Telco Customer Churn**, con el objetivo de predecir la fuga (abandono) de clientes en una empresa de telecomunicaciones.

El proyecto incluye una arquitectura básica de **MLOps**: el modelo final se sirve mediante una API REST con FastAPI, se empaqueta en una imagen Docker y se valida automáticamente con un pipeline de CI/CD en GitHub Actions.

---

## 1. Descripción

El proyecto desarrolla un modelo supervisado de clasificación binaria para predecir si un cliente abandonará la compañía. Se entrenan y comparan cuatro modelos basados en árboles (Random Forest, XGBoost, CatBoost y LightGBM), se selecciona el mejor según ROC AUC en validación cruzada estratificada, y se expone el modelo final como un servicio web reproducible y desplegable.

## 2. Problema de negocio

La retención de clientes es una de las palancas de mayor impacto financiero en el sector de telecomunicaciones: adquirir un cliente nuevo cuesta varias veces más que conservar uno existente. Anticipar qué clientes están en riesgo de abandono permite al equipo comercial diseñar acciones de retención focalizadas (ofertas, descuentos, contacto proactivo) en lugar de aplicar campañas masivas de baja eficiencia.

## 3. Dataset

Se utiliza el dataset **Telco Customer Churn**, que contiene 7 043 registros y 21 columnas con información de clientes individuales:

- **Demográficas**: `gender`, `SeniorCitizen`, `Partner`, `Dependents`.
- **Servicio telefónico e internet**: `PhoneService`, `MultipleLines`, `InternetService`, `OnlineSecurity`, `OnlineBackup`, `DeviceProtection`, `TechSupport`, `StreamingTV`, `StreamingMovies`.
- **Contractuales y de facturación**: `Contract`, `PaperlessBilling`, `PaymentMethod`, `tenure`, `MonthlyCharges`, `TotalCharges`.
- **Variable objetivo**: `Churn` (`Yes`/`No`, codificada a `1`/`0`).

## 4. Objetivo

1. Construir un modelo que prediga la probabilidad de que un cliente abandone la compañía.
2. Diseñar e implementar una arquitectura básica de MLOps que permita servir el modelo de forma reproducible, automatizada y monitoreable.

## 5. Análisis exploratorio (EDA) — hallazgos principales

- **Desbalance moderado de clases**: `Churn = No` ≈ 73.46 %, `Churn = Yes` ≈ 26.54 %.
- Se eliminó `customerID` por ser un identificador sin valor predictivo.
- `TotalCharges` se convirtió a numérica e imputó con `0` para los **11 registros vacíos**, todos correspondientes a clientes con `tenure = 0`.
- **Mayor churn en contratos `Month-to-month`** que en contratos a uno o dos años.
- **Mayor churn en clientes nuevos** (tenure bajo).
- **Mayor churn en clientes con `InternetService = Fiber optic`**.
- **Mayor churn en clientes que pagan con `Electronic check`**.
- Relación clara entre **cargos mensuales altos** (`MonthlyCharges`) y mayor probabilidad de abandono.
- La ausencia de servicios complementarios como `OnlineSecurity` y `TechSupport` también está asociada a un mayor churn.

## 6. Modelos entrenados

Todos los modelos se entrenaron como `Pipeline` de scikit-learn (preprocesamiento + estimador), con partición estratificada 80/20 y validación cruzada estratificada de 5 folds. La búsqueda de hiperparámetros se realizó con `GridSearchCV` y la métrica principal de selección fue **ROC AUC**.

| Modelo         | Accuracy | Precision | Recall | F1     | ROC AUC (Test) | ROC AUC (CV) |
| -------------- | -------- | --------- | ------ | ------ | -------------- | ------------ |
| Random Forest  | 0.7701   | 0.5490    | 0.7487 | 0.6335 | 0.8418         | 0.8468       |
| XGBoost        | 0.7445   | 0.5119    | 0.8021 | 0.6250 | 0.8482         | 0.8500       |
| **CatBoost**   | 0.7445   | 0.5120    | 0.7995 | 0.6242 | 0.8461         | **0.8506**   |
| LightGBM       | 0.7580   | 0.5297    | 0.7861 | 0.6329 | 0.8448         | 0.8467       |

## 7. Modelo seleccionado

Se seleccionó **CatBoost** como modelo final porque obtuvo el mayor **ROC AUC promedio en validación cruzada (0.8506)**, métrica más robusta frente al partido único de test y mejor indicador de la capacidad de generalización del modelo en este problema con clases desbalanceadas.

El modelo final, junto con su preprocesamiento, está serializado como un único `Pipeline` de scikit-learn en `app/model.joblib`.

## 8. Interpretabilidad

Para explicar las decisiones del modelo se aplicaron dos enfoques:

- **Importancia global de variables**: usando la importancia nativa de los modelos de árboles, se identificaron consistentemente las siguientes variables como las más relevantes:
  - `tenure`
  - `Contract_Month-to-month`
  - `Contract_Two year`
  - `InternetService_Fiber optic`
  - `PaymentMethod_Electronic check`
  - `MonthlyCharges`
  - `TotalCharges`
  - `OnlineSecurity`
  - `TechSupport`

- **LIME (Local Interpretable Model-agnostic Explanations)**: se generaron explicaciones locales para tres clientes representativos (alto riesgo, bajo riesgo y zona incierta), confirmando que las variables anteriores son las que más empujan la predicción individual hacia el churn o la retención.

## 9. Arquitectura MLOps

El flujo de extremo a extremo del proyecto es el siguiente:

```
┌──────────────┐    ┌──────────────┐    ┌──────────────┐    ┌──────────────┐
│   Datos      │ -> │ Entrenamiento│ -> │ Modelo       │ -> │ API FastAPI  │
│ (CSV crudo)  │    │  (notebooks) │    │ (model.joblib│    │ (/predict)   │
└──────────────┘    └──────────────┘    └──────────────┘    └──────┬───────┘
                                                                   │
                          ┌────────────────────────────────────────┘
                          v
┌──────────────┐    ┌──────────────┐    ┌──────────────┐    ┌──────────────┐
│  Pruebas     │ -> │  GitHub      │ -> │   Docker     │ -> │  Monitoreo   │
│  (pytest)    │    │  Actions CI  │    │  (imagen)    │    │ (producción) │
└──────────────┘    └──────────────┘    └──────────────┘    └──────────────┘
```

- **Datos** crudos en `data/telco_churn.csv`.
- **Entrenamiento** y selección de hiperparámetros en los notebooks numerados.
- **Modelo final** persistido como `Pipeline` completo en `app/model.joblib`.
- **API FastAPI** (`app/api.py`) que carga el modelo en memoria y expone los endpoints `/`, `/health` y `/predict`.
- **Pruebas unitarias** (`tests/test_api.py`, `tests/test_model.py`) ejecutadas con `pytest`.
- **CI/CD** con GitHub Actions: linting con `flake8`, pruebas con `pytest` y construcción/validación de la imagen Docker.
- **Docker**: la API se empaqueta en una imagen reproducible que escucha en el puerto `8000`.
- **Monitoreo** propuesto para producción (ver sección 16).

## 10. Estructura del repositorio

```
telco-churn-mlops/
├── app/
│   ├── api.py             # Servicio FastAPI
│   ├── schemas.py         # Esquemas Pydantic (entrada/salida)
│   └── model.joblib       # Pipeline (preprocesador + CatBoost)
├── data/
│   ├── telco_churn.csv
│   ├── X_train.joblib / X_test.joblib
│   ├── y_train.joblib / y_test.joblib
│   ├── preprocessor.joblib
│   ├── mejor_rf.joblib / mejor_xgb.joblib
│   ├── mejor_cat.joblib / mejor_lgbm.joblib
│   └── fig_*.png          # Gráficas de importancia y LIME
├── notebooks/
│   ├── 1_eda_preprocessing.ipynb
│   ├── 2_model_training.ipynb
│   └── 3_interpretability.ipynb
├── tests/
│   ├── test_api.py
│   └── test_model.py
├── .github/workflows/
│   └── ci.yml             # Pipeline CI/CD
├── Dockerfile
├── requirements.txt
└── README.md
```

## 11. Instalación local

Se recomienda usar **Python 3.10** y un entorno virtual aislado.

```bash
# Clonar el repositorio
git clone <url-del-repo>
cd telco-churn-mlops

# Crear y activar entorno virtual
python -m venv venv
source venv/bin/activate          # Linux / macOS
# venv\Scripts\activate           # Windows

# Instalar dependencias
pip install -r requirements.txt
```

## 12. Ejecutar la API en local

```bash
uvicorn app.api:app --reload --port 8000
```

Una vez arrancada, la documentación interactiva de Swagger está disponible en:

- <http://localhost:8000/docs>
- <http://localhost:8000/redoc>

## 13. Probar el endpoint `/predict`

### 13.1. Entrada (JSON)

```json
{
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
  "TotalCharges": 499.5
}
```

### 13.2. Solicitud con `curl`

```bash
curl -X POST "http://localhost:8000/predict" \
  -H "Content-Type: application/json" \
  -d '{
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
    "TotalCharges": 499.5
  }'
```

### 13.3. Respuesta esperada

```json
{
  "churn_probability": 0.9231,
  "prediction": "Yes",
  "threshold": 0.5,
  "model_name": "CatBoost"
}
```

## 14. Ejecutar la API con Docker

```bash
# Construir la imagen
docker build -t telco-churn-api .

# Ejecutar el contenedor
docker run -p 8000:8000 telco-churn-api
```

La API queda disponible en `http://localhost:8000`. El `Dockerfile` incluye además un `HEALTHCHECK` que valida periódicamente que `/health` responde correctamente.

## 15. Pruebas

Las pruebas verifican tanto la integridad del modelo serializado como el comportamiento de la API:

```bash
pytest tests/ -v
```

- `tests/test_model.py`: existencia del archivo del modelo, carga con `joblib`, presencia de `predict`/`predict_proba` y predicción válida sobre un cliente de ejemplo.
- `tests/test_api.py`: usa `TestClient` de FastAPI para validar `/`, `/health` y `/predict`, incluyendo rangos de probabilidad, etiquetas válidas, consistencia con el umbral y manejo de errores (campos faltantes y categorías inválidas).

## 16. CI/CD con GitHub Actions

El pipeline (`.github/workflows/ci.yml`) se ejecuta en cada `push` y `pull_request` y consta de dos jobs:

1. **`test`** (Lint y pruebas):
   - Configura Python 3.10.
   - Instala dependencias desde `requirements.txt`.
   - Ejecuta `flake8 app/ tests/` (ignorando `E501` y `W503`).
   - Ejecuta `pytest tests/`.
2. **`docker`** (Construcción de imagen): se ejecuta sólo si `test` pasa.
   - Construye la imagen con `docker/build-push-action`.
   - Levanta el contenedor y verifica que `/health` responde.
   - Deja preparados (deshabilitados con `if: false`) los pasos de **publicación** en Docker Hub mediante los secretos `DOCKERHUB_USERNAME` y `DOCKERHUB_TOKEN`.

## 17. Recomendaciones de monitoreo en producción

Una vez desplegado el modelo, se recomienda implementar las siguientes prácticas de monitoreo:

- **Deriva de datos (data drift)**: comparar periódicamente la distribución de las variables de entrada en producción contra la distribución del set de entrenamiento (test estadístico de Kolmogorov–Smirnov para numéricas, distancia Chi-cuadrado o PSI para categóricas).
- **Distribución de las predicciones**: vigilar que la proporción de predicciones `Yes`/`No` no se aleje significativamente de la tasa base histórica (~26.5 %). Un cambio brusco suele indicar un problema en los datos o un cambio real en el negocio.
- **Distribución de la probabilidad**: monitorear el histograma de `churn_probability` (media, mediana, percentiles 25/75/95) para detectar desplazamientos.
- **Métricas de desempeño con etiquetas reales**: cuando se cuente con la verdad de campo (varios meses después), calcular Accuracy, Precision, Recall, F1 y ROC AUC sobre ventanas móviles.
- **Cambios en variables de entrada**: registrar nuevos valores categóricos no vistos durante el entrenamiento, columnas faltantes y variaciones en los rangos de las numéricas.
- **Reentrenamiento periódico**: definir una cadencia (por ejemplo trimestral) y disparadores automáticos basados en caída de métricas (p. ej. ROC AUC < 0.80) o deriva detectada.
- **Trazabilidad**: registrar cada solicitud con un ID único, el payload, la respuesta y la versión del modelo, para poder auditar y depurar predicciones individuales.

Estas piezas pueden implementarse con herramientas como **Evidently AI**, **WhyLabs**, **MLflow**, **Prometheus + Grafana** o un stack propio basado en logs estructurados.

## 18. Conclusiones

- Los cuatro modelos basados en árboles obtuvieron desempeños muy cercanos entre sí (ROC AUC entre 0.844 y 0.851), lo que sugiere que el problema está bien acotado por las variables disponibles y que la elección del estimador no es el factor diferenciador principal.
- **CatBoost** se eligió como modelo final por su mejor ROC AUC en validación cruzada (0.8506), su robustez con variables categóricas y la consistencia de sus métricas frente a los demás candidatos.
- Las variables más informativas (`tenure`, tipo de contrato, tipo de internet, método de pago y cargos mensuales) son interpretables desde el negocio y permiten formular hipótesis accionables: clientes nuevos, con contrato mensual, fibra óptica y pago electrónico son el segmento de mayor riesgo y deberían concentrar los esfuerzos de retención.
- La arquitectura MLOps implementada cumple con los requisitos académicos de la asignación (servicio de inferencia, contenedor Docker, CI/CD y monitoreo propuesto) y deja una base sólida para evolucionar hacia un sistema productivo real, añadiendo registro de experimentos, versionado de modelos y monitoreo automatizado.
