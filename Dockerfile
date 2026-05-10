# =====================================================================
# Dockerfile para el servicio de predicción de churn
# =====================================================================
# Construye una imagen ligera basada en Python 3.10-slim que ejecuta
# la API FastAPI con uvicorn en el puerto 8000.
#
# Uso:
#   docker build -t telco-churn-api .
#   docker run -p 8000:8000 telco-churn-api
# =====================================================================

FROM python:3.10-slim

# Variables de entorno recomendadas para Python en contenedores
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# Dependencias del sistema necesarias para algunos paquetes
# (libgomp1 lo requieren CatBoost, LightGBM y XGBoost en runtime)
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        build-essential \
        libgomp1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Instalar dependencias primero para aprovechar la caché de capas
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

# Copiar código de la aplicación y artefactos necesarios
COPY app/ ./app/
COPY data/ ./data/

# Puerto en el que escucha uvicorn
EXPOSE 8000

# Healthcheck que valida que la API responde correctamente
HEALTHCHECK --interval=30s --timeout=10s --start-period=20s --retries=3 \
    CMD python -c "import urllib.request,sys; \
sys.exit(0) if urllib.request.urlopen('http://localhost:8000/health').status==200 else sys.exit(1)" \
    || exit 1

# Comando de arranque
CMD ["uvicorn", "app.api:app", "--host", "0.0.0.0", "--port", "8000"]
