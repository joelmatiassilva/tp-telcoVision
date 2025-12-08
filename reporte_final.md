# Reporte Final: Proyecto de MLOps - Telco Churn

## 1. Comparación de Experimentos

Durante el desarrollo del proyecto, se estableció un entorno de experimentación robusto utilizando **DVC** para la gestión de datos y **MLflow** para el rastreo de métricas y parámetros. La experimentación se centró en la comparación de múltiples algoritmos supervisados para clasificación.

### Configuración Experimental
Según lo definido en `params.yaml`, se exploraron los siguientes modelos candidatos:
- **Regresión Logística (lr):** Como línea base (baseline) por su interpretabilidad.
- **Random Forest (rf):** Para capturar no linealidades complejas.
- **XGBoost (xgboost):** Buscando maximizar el rendimiento mediante boosting.
- **Naive Bayes (nb) y SVM (svm):** Como alternativas para diferentes distribuciones de datos.

### Resultados Observados
Los experimentos fueron registrados en DagsHub a través de MLflow. Se monitorearon métricas clave como **Accuracy**, **AUC**, **Recall** y **Precision**.
- En las ejecuciones analizadas (ej. run `1da008...` en `telco_prod`), se observaron métricas de exactitud (Accuracy) rondando el **65%** en etapas iniciales o de validación cruzada.
- Las variaciones entre modelos permitieron identificar que modelos basados en árboles (Random Forest/XGBoost) tienden a manejar mejor las características categóricas del dataset de telecomunicaciones comparado con la regresión logística base.

El uso de **PyCaret** facilitó esta comparación masiva, generando artefactos como matrices de confusión y curvas ROC automáticamente, lo que permitió una selección basada en evidencia visual y cuantitativa.

## 2. Justificación del Modelo Final

La elección del modelo final no se basó únicamente en la métrica de exactitud (Accuracy), sino en el impacto al negocio:

1.  **Criterio de Negocio (Churn):** En la predicción de abandono, los **Falsos Negativos** (predecir que un cliente se queda cuando en realidad se va) son costosos, ya que se pierde la oportunidad de retenerlo.
2.  **Selección:** Se priorizó el modelo que ofreció el mejor balance en la curva ROC y un **Recall** competitivo, asegurando la detección de la mayor cantidad de clientes en riesgo.
3.  **Promoción:** El modelo ganador fue registrado en el **MLflow Model Registry** y promovido al stage de **"Production"**. Esta etiqueta es la "llave" que valida su calidad ante el sistema de despliegue automatizado.

## 3. Reflexión sobre el Despliegue en un Entorno Productivo

Basándonos en la arquitectura implementada en el directorio `telco_prod` y el documento `DEPLOYMENT.md`, la estrategia de puesta en producción diseñada es moderna, escalable y segura.

### Estrategia de Ramas (Branching Strategy)
Se ha desacoplado el entrenamiento del despliegue para mayor seguridad:
- **Rama `main` (CI):** Ejecuta el re-entrenamiento y registro de modelos. No toca la infraestructura de producción.
- **Rama `deploy` (CD):** Exclusiva para despliegue. Solo se activa cuando se aprueban cambios estables.

### Arquitectura de Servicio (Serving)
El despliegue no es monolítico, sino basado en microservicios contenerizados:
1.  **API REST:** Se utiliza **FastAPI** (`src/api/app.py`) para exponer el modelo. Esto permite una integración sencilla con cualquier frontend o CRM mediante peticiones HTTP estándar (POST `/predict`).
2.  **Containerización (Docker):** La aplicación se empaqueta en una imagen Docker. Esto garantiza que las dependencias (`requirements.txt`, versiones de librerías) sean idénticas en desarrollo y producción.
3.  **Infraestructura Serverless:** El destino final configurado es **AWS Lambda** (usando imágenes de contenedor almacenadas en **Amazon ECR**). Esto ofrece:
    *   **Escalado automático:** Si el tráfico sube, Lambda escala sin intervención manual.
    *   **Costos eficientes:** Solo se paga por el tiempo de cómputo de cada predicción.

### Gatekeepers de Seguridad
Un punto destacado de la implementación es el script `check_model.py`. Antes de cualquier despliegue, este script verifica automáticamente:
- ¿Existe el modelo en MLflow?
- ¿Está marcado como `Production`?

Si estas condiciones no se cumplen, el pipeline de despliegue se aborta, previniendo que una versión defectuosa o no aprobada llegue a los usuarios finales.

---
**Conclusión:** La solución transita de un notebook experimental a una API robusta en la nube, gobernada por prácticas de MLOps que aseguran que solo los mejores modelos, validados y versionados, sirvan predicciones en tiempo real.
