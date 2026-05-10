# Predicción de Churn en Telecomunicaciones con Machine Learning y MLOps

Proyecto académico de Machine Learning aplicado al dataset **Telco Customer Churn**, cuyo objetivo es predecir la probabilidad de abandono de clientes en una empresa de telecomunicaciones. El proyecto integra el ciclo completo de trabajo: análisis exploratorio, preprocesamiento, entrenamiento de modelos, evaluación, interpretabilidad, despliegue mediante API, contenerización con Docker, pruebas automáticas, CI/CD y monitoreo de deriva de datos.

---

## 1. Descripción general

El proyecto desarrolla un modelo supervisado de clasificación binaria para predecir si un cliente realizará **churn** (`Yes`) o permanecerá en la compañía (`No`). Se entrenaron y compararon modelos basados en árboles: **Random Forest**, **XGBoost**, **CatBoost** y **LightGBM**. El modelo final seleccionado fue **CatBoost**, al obtener el mejor desempeño promedio en validación cruzada según la métrica **ROC AUC**.

Además del desarrollo analítico, el proyecto implementa una arquitectura básica de **MLOps**, compuesta por:

- API de inferencia con **FastAPI**.
- Validación de datos con **Pydantic**.
- Modelo serializado en `app/model.joblib`.
- Contenedor **Docker** para ejecutar la API en un entorno reproducible.
- Pruebas unitarias con **pytest**.
- Flujo **CI/CD con GitHub Actions**.
- Monitoreo propuesto de deriva de datos mediante **Evidently AI**.
- Publicación del notebook como **Jupyter Book**.

---

## 2. Problema de negocio

En el sector de telecomunicaciones, anticipar la pérdida de clientes permite diseñar estrategias de retención más eficientes. Un modelo de churn ayuda a identificar clientes con alta probabilidad de abandono y permite tomar decisiones preventivas, como campañas focalizadas, revisión de contratos, beneficios personalizados o atención prioritaria.

---

## 3. Dataset

Se utiliza el dataset **Telco Customer Churn**, con **7.043 registros** y **21 columnas**. Cada fila representa un cliente y contiene información demográfica, contractual, de servicios y facturación.

Variables principales:

- **Demográficas:** `gender`, `SeniorCitizen`, `Partner`, `Dependents`.
- **Servicios contratados:** `PhoneService`, `MultipleLines`, `InternetService`, `OnlineSecurity`, `OnlineBackup`, `DeviceProtection`, `TechSupport`, `StreamingTV`, `StreamingMovies`.
- **Contrato y facturación:** `Contract`, `PaperlessBilling`, `PaymentMethod`, `tenure`, `MonthlyCharges`, `TotalCharges`.
- **Variable objetivo:** `Churn`, con valores `Yes` y `No`.

---

## 4. Objetivos

1. Desarrollar modelos de clasificación para predecir la probabilidad de abandono de clientes.
2. Comparar distintos modelos basados en árboles usando métricas de clasificación.
3. Seleccionar el mejor modelo con base en ROC AUC en validación cruzada.
4. Explicar las variables más importantes y predicciones individuales mediante interpretabilidad.
5. Implementar una arquitectura MLOps básica para servir el modelo mediante API.
6. Empaquetar la API y el modelo en Docker.
7. Automatizar validaciones con GitHub Actions.
8. Proponer mecanismos de monitoreo de deriva y desempeño en producción.

---

## 5. Análisis exploratorio de datos

Hallazgos principales:

- La variable objetivo presenta un desbalance moderado:
  - `Churn = No`: aproximadamente **73,46 %**.
  - `Churn = Yes`: aproximadamente **26,54 %**.
- Se eliminó `customerID` porque es un identificador sin valor predictivo.
- La variable `TotalCharges` fue convertida a numérica. Se identificaron 11 valores vacíos, asociados a clientes con `tenure = 0`, por lo cual fueron imputados con `0`.
- Los clientes con contrato `Month-to-month` presentan mayor tasa de abandono.
- Los clientes con baja antigüedad (`tenure` bajo) tienen mayor riesgo de churn.
- El servicio `Fiber optic` aparece asociado a una mayor proporción de abandono.
- El método de pago `Electronic check` se relaciona con mayor churn.
- Cargos mensuales altos (`MonthlyCharges`) se asocian con mayor probabilidad de abandono.
- La ausencia de servicios como `OnlineSecurity` y `TechSupport` también incrementa el riesgo de churn.

---

## 6. Preprocesamiento

El preprocesamiento se implementó mediante un `ColumnTransformer` dentro de un `Pipeline` de scikit-learn.

Transformaciones aplicadas:

- Variables numéricas:
  - Imputación con mediana.
  - Escalado con `StandardScaler`.
- Variables categóricas:
  - Imputación con la categoría más frecuente.
  - Codificación con `OneHotEncoder(handle_unknown="ignore")`.

Variables numéricas principales:

- `tenure`
- `MonthlyCharges`
- `TotalCharges`

Las demás variables predictoras fueron tratadas como categóricas, incluyendo `SeniorCitizen`.

---

## 7. Modelos entrenados

Se entrenaron cuatro modelos basados en árboles, todos integrados en `Pipeline` con preprocesamiento y optimizados mediante `GridSearchCV`.

La validación se realizó con **validación cruzada estratificada de 5 particiones**, usando como métrica principal **ROC AUC**.

| Modelo        | Accuracy | Precision | Recall | F1-score | ROC AUC Test | ROC AUC CV |
|--------------|---------:|----------:|-------:|---------:|-------------:|-----------:|
| Random Forest | 0.7701 | 0.5490 | 0.7487 | 0.6335 | 0.8418 | 0.8468 |
| XGBoost       | 0.7445 | 0.5119 | 0.8021 | 0.6250 | 0.8482 | 0.8500 |
| **CatBoost**  | 0.7445 | 0.5120 | 0.7995 | 0.6242 | 0.8461 | **0.8506** |
| LightGBM      | 0.7580 | 0.5297 | 0.7861 | 0.6329 | 0.8448 | 0.8467 |

---

## 8. Modelo seleccionado

El modelo seleccionado fue **CatBoost**, porque obtuvo el mayor **ROC AUC promedio en validación cruzada: 0.8506**.

Aunque XGBoost obtuvo el mejor ROC AUC en test y Random Forest presentó el mejor F1-score, CatBoost fue elegido por su mejor desempeño promedio en validación cruzada, lo cual ofrece una evaluación más estable de su capacidad de generalización.

El modelo final se guardó como un único `Pipeline` en:

```text
app/model.joblib
```

Este archivo incluye el preprocesamiento y el clasificador final, por lo que la API puede recibir los datos originales del cliente en formato tabular sin codificación manual adicional.

---

## 9. Interpretabilidad

Se utilizaron dos enfoques de interpretabilidad:

### 9.1. Importancia global de variables

Las variables más relevantes identificadas de forma recurrente fueron:

- `tenure`
- `Contract_Month-to-month`
- `Contract_Two year`
- `InternetService_Fiber optic`
- `PaymentMethod_Electronic check`
- `MonthlyCharges`
- `TotalCharges`
- `OnlineSecurity`
- `TechSupport`

### 9.2. LIME

Se aplicó **LIME** sobre tres casos representativos:

- Cliente de alto riesgo.
- Cliente de bajo riesgo.
- Cliente en zona incierta.

Las explicaciones locales confirmaron que las predicciones están fuertemente influenciadas por variables como el tipo de contrato, la antigüedad, el método de pago, el tipo de servicio de internet y los cargos mensuales.

---

## 10. Arquitectura MLOps implementada

La arquitectura MLOps convierte el modelo entrenado en un servicio de inferencia reproducible.

Flujo general:

```text
Datos originales
     ↓
EDA y preprocesamiento
     ↓
Entrenamiento y comparación de modelos
     ↓
Selección del modelo CatBoost
     ↓
Serialización del Pipeline en app/model.joblib
     ↓
API FastAPI con endpoint /predict
     ↓
Pruebas unitarias con pytest
     ↓
Contenerización con Docker
     ↓
CI/CD con GitHub Actions
     ↓
Monitoreo de deriva y métricas de producción
```

Componentes implementados:

- `app/api.py`: servicio FastAPI.
- `app/schemas.py`: esquemas Pydantic para validar entrada y salida.
- `app/model.joblib`: modelo final serializado.
- `tests/`: pruebas unitarias de API y modelo.
- `Dockerfile`: configuración del contenedor.
- `.github/workflows/ci.yml`: pipeline de integración continua.
- `monitoring/data_drift_report.py`: generación de reporte de deriva de datos.
- `book/`: Jupyter Book del proyecto.

---

## 11. Estructura actual del repositorio

```text
telco-churn-mlops/
├── _build/                         # Construcción local del Jupyter Book
├── app/
│   ├── __init__.py
│   ├── api.py                      # API FastAPI
│   ├── schemas.py                  # Esquemas Pydantic
│   └── model.joblib                # Modelo final CatBoost en Pipeline
├── book/
│   ├── _config.yml                 # Configuración del Jupyter Book
│   ├── _toc.yml                    # Tabla de contenido del Jupyter Book
│   ├── Proyecto_Final_Telco_Churn_MLOps.ipynb
│   └── images/                     # Recursos gráficos del libro, si aplica
├── data/
│   ├── telco_churn.csv
│   ├── X_train.joblib
│   ├── X_test.joblib
│   ├── y_train.joblib
│   ├── y_test.joblib
│   ├── preprocessor.joblib
│   ├── mejor_rf.joblib
│   ├── mejor_xgb.joblib
│   ├── mejor_cat.joblib
│   ├── mejor_lgbm.joblib
│   ├── drift_report.html           # Reporte de deriva generado con Evidently
│   └── fig_*.png                   # Figuras del análisis e interpretabilidad
├── monitoring/
│   └── data_drift_report.py        # Script para monitoreo de deriva
├── notebooks/
│   ├── 1_eda_preprocessing.ipynb
│   ├── 2_model_training.ipynb
│   ├── 3_interpretability.ipynb
│   └── Proyecto_Final_Telco_Churn_MLOps.ipynb
├── tests/
│   ├── conftest.py
│   ├── test_api.py
│   └── test_model.py
├── .github/
│   └── workflows/
│       └── ci.yml                  # Flujo CI/CD
├── Dockerfile
├── README.md
└── requirements.txt
```

---

## 12. Instalación local

Se recomienda usar **Python 3.10**.

```bash
git clone <url-del-repositorio>
cd telco-churn-mlops

python -m venv venv
source venv/bin/activate      # Linux/macOS
# venv\Scripts\activate       # Windows

pip install -r requirements.txt
```

En caso de usar Conda:

```bash
conda create -n telco_mlops python=3.10 -y
conda activate telco_mlops
pip install -r requirements.txt
```

---

## 13. Ejecutar la API localmente con Uvicorn

Durante el desarrollo, la API puede ejecutarse localmente con Uvicorn:

```bash
uvicorn app.api:app --reload --port 8000
```

Luego se puede abrir la documentación interactiva en:

```text
http://localhost:8000/docs
```

También se puede consultar el estado del servicio en:

```text
http://localhost:8000/health
```

---

## 14. Endpoint de predicción

El endpoint principal es:

```text
POST /predict
```

Recibe un JSON con las 19 variables predictoras del cliente y devuelve la probabilidad de churn.

### Ejemplo de entrada

```json
{
  "gender": "Female",
  "SeniorCitizen": 0,
  "Partner": "No",
  "Dependents": "No",
  "tenure": 2,
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
  "MonthlyCharges": 95.5,
  "TotalCharges": 190.0
}
```

### Ejemplo de respuesta

```json
{
  "churn_probability": 0.86,
  "prediction": "Yes",
  "threshold": 0.5,
  "model_name": "CatBoost"
}
```

Interpretación:

- `churn_probability`: probabilidad estimada de que el cliente abandone la compañía.
- `prediction`: clasificación final según el umbral.
- `threshold`: punto de corte usado para clasificar.
- `model_name`: modelo utilizado por la API.

---

## 15. Diferencia entre Uvicorn y Docker

**Uvicorn** ejecuta la API directamente en el entorno local de desarrollo. Es útil para probar rápidamente la API desde el computador.

**Docker** empaqueta la API, el modelo y las dependencias en un contenedor. Esto permite que el proyecto se ejecute de forma reproducible en otros equipos o servidores, sin depender de la configuración local.

Aunque en ambos casos se visualiza la misma documentación de FastAPI en `http://localhost:8000/docs`, la diferencia está en el entorno donde se ejecuta el servicio.

---

## 16. Ejecutar la API con Docker

Construir la imagen:

```bash
docker build -t telco-churn-api .
```

Ejecutar el contenedor:

```bash
docker run -p 8000:8000 telco-churn-api
```

Luego abrir:

```text
http://localhost:8000/docs
```

Nota: dentro del contenedor, Uvicorn escucha en `0.0.0.0:8000`, pero desde el navegador se debe acceder mediante `localhost:8000`.

---

## 17. Pruebas unitarias

Ejecutar pruebas:

```bash
python -m pytest tests/ -v
```

Las pruebas verifican:

- Que el modelo `app/model.joblib` existe.
- Que el modelo puede cargarse con `joblib`.
- Que el modelo tiene métodos de predicción.
- Que la API responde en `/`, `/health` y `/predict`.
- Que la probabilidad retornada está entre 0 y 1.
- Que la predicción corresponde a `Yes` o `No`.
- Que los errores de validación se manejan correctamente.

---

## 18. CI/CD con GitHub Actions

El archivo `.github/workflows/ci.yml` define un flujo automático que se ejecuta en cada `push` o `pull_request`.

El pipeline incluye dos jobs:

### 18.1. Lint y pruebas

- Configura Python 3.10.
- Instala dependencias desde `requirements.txt`.
- Ejecuta `flake8` sobre `app/` y `tests/`.
- Ejecuta `pytest tests/ -v`.

### 18.2. Construcción y validación Docker

- Construye la imagen Docker.
- Levanta el contenedor.
- Verifica que el endpoint `/health` responda correctamente.
- Detiene y elimina el contenedor.

También se dejaron preparados pasos opcionales para publicar la imagen en Docker Hub, deshabilitados por defecto mediante `if: false`.

---

## 19. Monitoreo de deriva de datos

El proyecto incluye una propuesta de monitoreo mediante **Evidently AI**. El script se encuentra en:

```text
monitoring/data_drift_report.py
```

Para generar el reporte:

```bash
python monitoring/data_drift_report.py
```

El reporte se guarda en:

```text
data/drift_report.html
```

Este reporte compara `X_train` como datos de referencia frente a `X_test` como datos actuales simulados. En un entorno real, `X_test` sería reemplazado por datos recientes recolectados en producción.

Variables relevantes para monitoreo:

- `tenure`
- `MonthlyCharges`
- `TotalCharges`
- `Contract`
- `InternetService`
- `PaymentMethod`
- `OnlineSecurity`
- `TechSupport`

Si se detecta deriva significativa, se recomienda revisar el desempeño del modelo y considerar un proceso de reentrenamiento.

---

## 20. Recomendaciones de monitoreo en producción

En una implementación productiva se recomienda monitorear:

- Distribución de variables numéricas.
- Frecuencias de variables categóricas.
- Porcentaje de predicciones `Yes` y `No`.
- Distribución de `churn_probability`.
- Número de solicitudes a la API.
- Valores faltantes o categorías no vistas.
- Métricas reales cuando se disponga de etiquetas posteriores: Accuracy, Precision, Recall, F1-score y ROC AUC.

El reentrenamiento debe considerarse si:

- La deriva de datos es significativa.
- El ROC AUC baja frente al valor esperado.
- El recall disminuye y el modelo deja de detectar clientes que abandonan.
- La distribución de predicciones cambia abruptamente.

---

## 21. Jupyter Book

El notebook final del proyecto se publicó como Jupyter Book. Link directo: https://danielahernav2006.github.io/telco-customer-churn-mlops/

---

## 22. Conclusiones

- El proyecto permitió desarrollar un flujo completo de Machine Learning para predicción de churn, desde el análisis exploratorio hasta una arquitectura básica de despliegue.
- Los modelos entrenados presentaron desempeños cercanos, con ROC AUC entre aproximadamente 0.84 y 0.85.
- **CatBoost** fue seleccionado como modelo final por obtener el mejor ROC AUC en validación cruzada.
- Los factores más asociados al churn fueron contratos mensuales, baja antigüedad, fibra óptica, pago mediante electronic check, cargos mensuales altos y ausencia de servicios de soporte o seguridad.
- La implementación de FastAPI permite servir el modelo como API de inferencia, recibiendo datos nuevos y retornando predicciones en formato JSON.
- Docker permite ejecutar la API en un entorno reproducible e independiente del computador local.
- GitHub Actions automatiza pruebas, revisión de calidad del código y construcción de la imagen Docker.
- El monitoreo de deriva de datos con Evidently fortalece la propuesta MLOps, al permitir evaluar si los datos nuevos se alejan de los datos usados para entrenar el modelo.
- Como trabajo futuro, se recomienda incorporar registro de experimentos con MLflow, almacenamiento de logs en producción, versionado de modelos y reentrenamiento automático.
