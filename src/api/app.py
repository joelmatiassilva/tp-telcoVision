# src/app.py
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import mlflow
import pandas as pd
import os
from dotenv import load_dotenv
import uvicorn
import logging

# Configurar logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Cargar variables de entorno
load_dotenv()

app = FastAPI(
    title="TelcoVision Churn Prediction API",
    description="API para predecir la retención de clientes (Churn) usando el modelo productivo de MLflow.",
    version="1.0.0"
)

# Configuración de MLflow
MODEL_NAME = os.getenv("MLFLOW_MODEL_NAME", "telco-churn-prediction")
MODEL_STAGE = os.getenv("MLFLOW_MODEL_STAGE", "Production")

# Variable global para el modelo
model = None
model_info = {}

@app.on_event("startup")
def load_model():
    """Carga el modelo desde MLflow con manejo robusto de errores"""
    global model, model_info
    
    try:
        # Configurar MLflow
        tracking_uri = os.getenv("MLFLOW_TRACKING_URI")
        if tracking_uri:
            mlflow.set_tracking_uri(tracking_uri)
            logger.info(f"MLflow URI configurado: {tracking_uri}")
        else:
            logger.warning("MLFLOW_TRACKING_URI no configurado, usando configuración local")
        
        # Construir URI del modelo
        model_uri = f"models:/{MODEL_NAME}/{MODEL_STAGE}"
        logger.info(f"Intentando cargar modelo desde: {model_uri}")
        
        # Cargar modelo
        model = mlflow.pyfunc.load_model(model_uri)
        
        # Guardar información del modelo
        model_info = {
            "name": MODEL_NAME,
            "stage": MODEL_STAGE,
            "uri": model_uri,
            "status": "loaded"
        }
        
        logger.info(f"✅ Modelo cargado exitosamente: {MODEL_NAME} ({MODEL_STAGE})")
        
    except Exception as e:
        logger.error(f"❌ Error al cargar modelo '{MODEL_NAME}' en stage '{MODEL_STAGE}': {e}")
        logger.error("La API iniciará pero las predicciones fallarán hasta que se cargue un modelo válido")
        
        model_info = {
            "name": MODEL_NAME,
            "stage": MODEL_STAGE,
            "status": "error",
            "error": str(e)
        }

class CustomerData(BaseModel):
    customer_id: str
    age: int
    gender: str
    region: str
    contract_type: str
    tenure_months: int
    monthly_charges: float
    total_charges: float
    internet_service: str
    phone_service: str
    multiple_lines: str
    payment_method: str

    class Config:
        schema_extra = {
            "example": {
                "customer_id": "CUST-12345",
                "age": 30,
                "gender": "Female",
                "region": "West",
                "contract_type": "Month-to-Month",
                "tenure_months": 12,
                "monthly_charges": 50.5,
                "total_charges": 600.0,
                "internet_service": "DSL",
                "phone_service": "Yes",
                "multiple_lines": "No",
                "payment_method": "Electronic check"
            }
        }

@app.get("/")
def read_root():
    """Endpoint de health check"""
    return {
        "status": "online",
        "model": model_info,
        "message": "TelcoVision Churn Prediction API"
    }

@app.get("/health")
def health_check():
    """Endpoint detallado de health check"""
    if model is None:
        return {
            "status": "unhealthy",
            "model": model_info,
            "message": "Modelo no cargado"
        }
    
    return {
        "status": "healthy",
        "model": model_info,
        "message": "API lista para predicciones"
    }

@app.post("/predict")
def predict(data: CustomerData):
    """
    Realiza una predicción de churn para un cliente.
    
    Returns:
        - churn_prediction: 0 (no churn) o 1 (churn)
        - churn_risk: "LOW" o "HIGH"
        - customer_id: ID del cliente
    """
    # Verificar que el modelo esté cargado
    if model is None:
        logger.error("Intento de predicción sin modelo cargado")
        raise HTTPException(
            status_code=503,
            detail={
                "error": "Modelo no disponible",
                "message": "El modelo no pudo ser cargado desde MLflow",
                "model_info": model_info
            }
        )
    
    try:
        # Convertir input a DataFrame
        input_data = data.dict()
        customer_id = input_data.get("customer_id")
        
        logger.info(f"Procesando predicción para cliente: {customer_id}")
        
        df = pd.DataFrame([input_data])
        
        # Realizar predicción
        prediction = model.predict(df)
        result = int(prediction[0])
        
        # Interpretar resultado
        churn_risk = "HIGH" if result == 1 else "LOW"
        
        logger.info(f"Predicción completada para {customer_id}: {result} ({churn_risk})")
        
        return {
            "customer_id": customer_id,
            "churn_prediction": result,
            "churn_risk": churn_risk,
            "interpretation": "Cliente con riesgo de abandono" if result == 1 else "Cliente sin riesgo de abandono"
        }
        
    except Exception as e:
        logger.error(f"Error en predicción: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail={
                "error": "Error en predicción",
                "message": str(e),
                "customer_id": data.customer_id
            }
        )

# Handler para AWS Lambda
from mangum import Mangum
handler = Mangum(app)

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
