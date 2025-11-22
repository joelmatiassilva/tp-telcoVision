# Guía de Despliegue - TelcoVision Churn Prediction

Este documento describe el proceso de despliegue del modelo en AWS Lambda, correspondiente a la **Etapa 7** del proyecto.

## Flujo de Despliegue (Branching Strategy)

El proyecto utiliza una estrategia de ramas para separar el entrenamiento del despliegue:

1.  **Rama `main` (Integración Continua - Etapa 5)**:
    *   Cada cambio aquí dispara el re-entrenamiento del modelo.
    *   El objetivo es generar y registrar una nueva versión del modelo en **MLflow**.
    *   **No despliega a producción**.

2.  **Rama `deploy` (Despliegue Continuo - Etapa 7)**:
    *   Esta rama es exclusiva para producción.
    *   Al recibir cambios (merge desde `main`), dispara el proceso de despliegue a AWS.
    *   **Verificación Automática**: Antes de desplegar, el sistema verifica que exista un modelo válido en MLflow.

## Requisitos Previos

-   Repositorio en **Amazon ECR** creado.
-   Función **AWS Lambda** creada (tipo Container Image).
-   Secretos configurados en GitHub (ver README).

## Verificación Automática

El pipeline incluye un paso de seguridad (`src/check_model.py`) que se ejecuta antes de construir la imagen Docker. Este script:
1.  Se conecta a MLflow.
2.  Busca el modelo `telco-churn-prediction` en stage `Production`.
3.  Si no lo encuentra, **aborta el despliegue**.

> [!TIP]
> Si el despliegue falla en el paso "Verify Model", asegúrate de que el pipeline de la rama `main` haya terminado exitosamente primero.

## API REST con FastAPI

La aplicación se expone mediante FastAPI en `src/api/app.py`.

### Prueba Local del Modelo

Antes de ejecutar la API, verifica que el modelo se puede cargar desde DagsHub:

```bash
python test_model_loading.py
```

Este script te confirmará si:
- El modelo existe en MLflow
- Está en stage `Production`
- Se puede descargar correctamente
- Puede hacer predicciones

### Ejecución Local de la API

**Opción 1: Script automatizado (Recomendado)**

```bash
./run_api.sh
```

**Opción 2: Manual**

```bash
conda activate pycaret-env
python -c "
import sys
sys.path.insert(0, '.')
from src.api.app import app
import uvicorn
uvicorn.run(app, host='0.0.0.0', port=8000)
"
```

Accede a la documentación interactiva en: `http://localhost:8000/docs`

### Endpoints Disponibles

- `GET /` - Health check básico con información del modelo
- `GET /health` - Health check detallado (estado del modelo)
- `POST /predict` - Predicción de churn
- `GET /docs` - Documentación interactiva Swagger UI

### Configuración del Modelo

La API busca el modelo en MLflow Model Registry bajo:
- **Nombre**: `telco-churn-prediction` (configurable con `MLFLOW_MODEL_NAME`)
- **Stage**: `Production` (configurable con `MLFLOW_MODEL_STAGE`)

## Ejecución con Docker

Para probar la imagen exactamente como correrá en Lambda (usando el emulador RIE si fuera necesario, o simplemente verificando el build):

```bash
docker build -t telco-api .
# Nota: La imagen de Lambda no se ejecuta directamente como un servidor web normal localmente sin el Runtime Interface Emulator.
```

## Troubleshooting

### Error: "Missing cache files" en GitHub Actions

Si el CI falla con `ERROR: failed to pull data from the cloud - Checkout failed`, verifica:

1.  **Secretos de GitHub**: Asegúrate de que `DAGSHUB_USER` y `DAGSHUB_TOKEN` estén configurados correctamente en **Settings > Secrets and variables > Actions**.
2.  **Permisos en DagsHub**: El token debe tener permisos de lectura en el repositorio de datos.
3.  **Debug**: El workflow incluye un paso "Debug Secrets" que mostrará si los secretos están vacíos (sin revelar su valor).

### Error: "Verify Model" falla en deploy

Si el job de `deploy` falla en la verificación del modelo:

1.  Asegúrate de que el job `build-and-evaluate` en la rama `main` haya corrido exitosamente.
2.  Verifica en [DagsHub/MLflow](https://dagshub.com/joelmatiassilva/tp-labMineriaDeDatos-telco/experiments) que exista un modelo registrado.
3.  Si el modelo existe pero no está en stage `Production`, promuévelo manualmente en la UI de MLflow.
