# src/app.py
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import mlflow
import pandas as pd
import os
from dotenv import load_dotenv
import uvicorn

# Cargar variables de entorno
load_dotenv()

app = FastAPI(
    title="TelcoVision Churn Prediction API",
    description="API para predecir la retención de clientes (Churn) usando el modelo productivo de MLflow.",
    version="1.0.0"
)

# Configuración de MLflow
# Se intenta leer la URI de tracking de las variables de entorno, o se usa la default del proyecto
tracking_uri = os.getenv("MLFLOW_TRACKING_URI")
if tracking_uri:
    mlflow.set_tracking_uri(tracking_uri)

# Nombre del modelo y stage por defecto
MODEL_NAME = os.getenv("MLFLOW_MODEL_NAME", "telco-churn-prediction") # Ajustar al nombre registrado en MLflow
MODEL_STAGE = os.getenv("MLFLOW_MODEL_STAGE", "Production")

model = None

@app.on_event("startup")
def load_model():
    global model
    try:
        # Construir la URI del modelo
        # Si estamos local y no hay registro de modelos, podríamos cargar desde un run específico si se define
        # Pero para "Productivo" asumimos Model Registry.
        # Fallback: Si no hay registry, intentar cargar el último run exitoso (lógica simplificada)
        
        model_uri = f"models:/{MODEL_NAME}/{MODEL_STAGE}"
        print(f"Intentando cargar modelo desde: {model_uri}")
        
        # Nota: Si no existe el modelo en registry, esto fallará.
        # Para desarrollo local sin registry, se podría usar una ruta local o un Run ID fijo.
        model = mlflow.pyfunc.load_model(model_uri)
        print("Modelo cargado exitosamente.")
    except Exception as e:
        print(f"Error cargando el modelo '{MODEL_NAME}' en stage '{MODEL_STAGE}': {e}")
        print("ADVERTENCIA: La API iniciará pero las predicciones fallarán hasta que se cargue un modelo válido.")

class CustomerData(BaseModel):
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
    return {"status": "online", "model": MODEL_NAME, "stage": MODEL_STAGE}

@app.post("/predict")
def predict(data: CustomerData):
    if not model:
        raise HTTPException(status_code=503, detail="Modelo no cargado.")
    
    try:
        # Convertir input a DataFrame
        input_data = data.dict()
        df = pd.DataFrame([input_data])
        
        # Realizar predicción
        prediction = model.predict(df)
        
        # Asumimos que la predicción devuelve un array/series. PyCaret suele devolver label (0/1) o etiqueta.
        # Si devuelve dataframe (label, score), hay que procesarlo.
        # MLflow pyfunc devuelve numpy array o pandas series/df dependiendo del flavor.
        
        result = prediction[0]
        
        # Intentar obtener probabilidad si es posible (depende del modelo cargado)
        # Esto es complejo con pyfunc genérico sin saber la estructura exacta de salida.
        # Devolveremos el resultado directo.
        
        return {
            "churn_prediction": int(result) if isinstance(result, (int, float)) or str(result).isdigit() else str(result),
            "input_received": input_data
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error en predicción: {str(e)}")

from mangum import Mangum

# ... (código existente)

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)

# Handler para AWS Lambda
handler = Mangum(app)
