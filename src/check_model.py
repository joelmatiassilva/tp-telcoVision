import mlflow
import os
import sys
from dotenv import load_dotenv

def check_model_existence():
    # Cargar variables de entorno
    load_dotenv()
    
    tracking_uri = os.getenv("MLFLOW_TRACKING_URI")
    if not tracking_uri:
        print("ERROR: MLFLOW_TRACKING_URI no definida.")
        sys.exit(1)
        
    mlflow.set_tracking_uri(tracking_uri)
    
    model_name = os.getenv("MLFLOW_MODEL_NAME", "telco-churn-prediction")
    stage = os.getenv("MLFLOW_MODEL_STAGE", "Production")
    
    print(f"Verificando existencia de modelo '{model_name}' en stage '{stage}'...")
    
    client = mlflow.tracking.MlflowClient()
    
    try:
        # Obtener versiones del modelo en ese stage
        # get_latest_versions devuelve una lista, si está vacía es que no hay modelo
        versions = client.get_latest_versions(model_name, stages=[stage])
        
        if not versions:
            print(f"ERROR: No se encontró ninguna versión del modelo '{model_name}' en stage '{stage}'.")
            print("Asegúrate de que el pipeline de entrenamiento (rama main) haya corrido exitosamente.")
            sys.exit(1)
            
        latest = versions[0]
        print(f"✅ Modelo encontrado: Versión {latest.version} (Run ID: {latest.run_id})")
        sys.exit(0)
        
    except Exception as e:
        print(f"ERROR consultando MLflow: {e}")
        sys.exit(1)

if __name__ == "__main__":
    check_model_existence()
