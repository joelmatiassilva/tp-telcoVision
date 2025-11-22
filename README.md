# TelcoVision - Churn Prediction (Producción)

Este proyecto implementa un pipeline de MLOps completo para la predicción de churn en una empresa de telecomunicaciones, cumpliendo con los estándares de producción.

## 🎯 Cumplimiento de Etapas del Proyecto

### Etapa 5: CI/CD con GitHub Actions
**Objetivo**: Automatizar la validación y el entrenamiento del modelo.

Implementación en `.github/workflows/ci.yaml` (Rama `main`):
1.  **Automatización**: Se dispara automáticamente en cada `push` a la rama `main`.
2.  **Reproducibilidad**: Descarga los datos versionados con **DVC** desde DagsHub.
3.  **Entrenamiento**: Ejecuta el pipeline de DVC (`dvc repro`) que corre `src/train.py`.
4.  **Persistencia**: El modelo entrenado se registra automáticamente en **MLflow (DagsHub)**. Esto es crítico para conectar con la siguiente etapa.
5.  **Reporte**: Publica métricas de evaluación en el resumen del Pull Request.

### Etapa 7: Producción (Despliegue)
**Objetivo**: Disponibilizar el modelo para consumo externo.

Implementación en `.github/workflows/ci.yaml` (Rama `deploy`):
1.  **Separación de Entornos**: El despliegue solo ocurre cuando se hace merge/push a la rama `deploy`.
2.  **Verificación de Seguridad**: Antes de desplegar, el script `src/check_model.py` verifica que exista un modelo válido en MLflow (`Production`). Si no existe, el despliegue se cancela para evitar errores.
3.  **Arquitectura Serverless con Docker**:
    *   Se utiliza **AWS Lambda** para escalar automáticamente y reducir costos.
    *   Se empaqueta la aplicación en una imagen **Docker** (basada en `public.ecr.aws/lambda/python:3.9`) para soportar el tamaño de las dependencias de **PyCaret** (>250MB).
4.  **API**: Se expone el modelo mediante **FastAPI** (`src/app.py`) con un endpoint `/predict` documentado.

## 🏗 Arquitectura Técnica

### Stack Tecnológico
-   **Entrenamiento**: PyCaret (AutoML).
-   **Tracking & Registry**: MLflow + DagsHub.
-   **Versionado de Datos**: DVC (Data Version Control).
-   **Métricas**: Estrategia híbrida DVC + MLflow.
-   **API**: FastAPI + Mangum (adaptador serverless).
-   **Infraestructura**: AWS Lambda (Docker Image) + Amazon ECR.
-   **CI/CD**: GitHub Actions.

### Estrategia de Métricas (Híbrida)

El proyecto utiliza **ambos** sistemas de métricas para obtener lo mejor de cada uno:

1.  **DVC Metrics** (`outputs/metrics/metrics.json`):
    *   Métricas básicas en formato JSON versionado.
    *   Visible en terminal con `dvc metrics show`.
    *   Mostrado automáticamente en el resumen de GitHub Actions.
    *   Ideal para comparaciones rápidas entre ramas (`dvc metrics diff`).

2.  **MLflow**:
    *   Métricas detalladas + gráficos (ROC, Confusion Matrix).
    *   Histórico completo de experimentos.
    *   Registro de modelos con versionado.
    *   Accesible vía web en [DagsHub](https://dagshub.com/joelmatiassilva/tp-labMineriaDeDatos-telco/experiments).

### Flujo de Trabajo Recomendado
1.  **Desarrollo (`main`)**:
    *   Hacer cambios en código o datos.
    *   `git push origin main` -> GitHub Actions entrena y valida el modelo.
    *   Verificar métricas y que el modelo aparezca en MLflow.
2.  **Despliegue (`deploy`)**:
    *   Una vez validado `main`, hacer merge a `deploy`.
    *   `git push origin deploy` -> GitHub Actions verifica el modelo, construye la imagen Docker y actualiza AWS Lambda.

## 📂 Estructura del Proyecto

```
telco_prod/
├── .github/workflows/  # CI/CD: Separado en jobs build (main) y deploy (deploy)
├── data/               # Datos gestionados por DVC
├── src/
│   ├── app.py          # API FastAPI (Entrypoint Lambda)
│   ├── check_model.py  # Script de verificación pre-deploy
│   ├── train.py        # Script de entrenamiento
│   ├── evaluate.py     # Evaluación y generación de métricas
│   └── data_prep.py    # Preparación de datos
├── Dockerfile          # Definición de la imagen para Lambda
├── dvc.yaml            # Pipeline reproducible (Data Prep -> Train -> Eval)
├── params.yaml         # Hiperparámetros globales
└── requirements.txt    # Dependencias del proyecto
```

## ☁️ Configuración de Secretos

Para que el despliegue funcione, se requieren los siguientes secretos en GitHub:
*   `DAGSHUB_USER`, `DAGSHUB_TOKEN`: Acceso a datos y MLflow.
*   `AWS_ACCESS_KEY_ID`, `AWS_SECRET_ACCESS_KEY`, `AWS_REGION`: Credenciales AWS.
*   `ECR_REPOSITORY`: Nombre del repo ECR.
*   `LAMBDA_FUNCTION_NAME`: Nombre de la función Lambda.
