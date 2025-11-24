"""
Script para promover el mejor modelo a Production en MLflow Model Registry.

Este script:
1. Busca el mejor run en el experimento basándose en una métrica (ej: final_accuracy).
2. Registra el modelo en el Model Registry si no está registrado.
3. Promueve la versión a stage 'Production'.
"""

import mlflow
from mlflow.tracking import MlflowClient
import os
import yaml
from dotenv import load_dotenv

def promote_best_model():
    # Cargar configuración
    load_dotenv()
    
    with open('params.yaml') as f:
        params = yaml.safe_load(f)
    
    # Configurar MLflow
    track_remote = params.get('track_to_dagshub', False) or os.getenv('TRACK_TO_DAGSHUB') == 'true'
    
    if track_remote:
        load_dotenv()
        tracking_uri = os.getenv('MLFLOW_TRACKING_URI') or params.get('dagshub_tracking_uri')
        if tracking_uri:
            mlflow.set_tracking_uri(tracking_uri)
    
    experiment_name = "telco-churn-prediction"
    model_name = "telco-churn-prediction"
    
    # Métrica para seleccionar el mejor modelo
    metric_name = "final_accuracy"  # Puedes cambiar a final_f1, final_auc, etc.
    
    print(f"Buscando el mejor modelo en experimento '{experiment_name}' por métrica '{metric_name}'...")
    
    # Buscar el mejor run
    best_run = mlflow.search_runs(
        experiment_names=[experiment_name],
        filter_string=f"metrics.{metric_name} > 0",  # Solo runs con esta métrica
        order_by=[f"metrics.{metric_name} DESC"],
        max_results=1
    )
    
    if best_run.empty:
        print(f"ERROR: No se encontraron runs con la métrica '{metric_name}'.")
        return
    
    best_run_id = best_run.iloc[0]["run_id"]
    best_metric_value = best_run.iloc[0][f"metrics.{metric_name}"]
    
    print(f"✅ Mejor run encontrado: {best_run_id}")
    print(f"   {metric_name}: {best_metric_value:.4f}")
    
    # Inicializar cliente de MLflow
    client = MlflowClient()
    
    # Verificar si el modelo ya está registrado
    try:
        registered_model = client.get_registered_model(model_name)
        print(f"Modelo '{model_name}' ya existe en el registry.")
    except:
        print(f"Modelo '{model_name}' no existe. Creándolo...")
        client.create_registered_model(model_name)
    
    # Registrar la versión del modelo desde el run
    model_uri = f"runs:/{best_run_id}/model"
    
    print(f"Registrando modelo desde {model_uri}...")
    model_version = mlflow.register_model(model_uri, model_name)
    
    print(f"✅ Modelo registrado como versión {model_version.version}")
    
    # Promover a Production
    print(f"Promoviendo versión {model_version.version} a stage 'Production'...")
    
    # Primero, archivar cualquier versión anterior en Production
    for mv in client.get_latest_versions(model_name, stages=["Production"]):
        print(f"   Archivando versión anterior {mv.version} de Production...")
        client.transition_model_version_stage(
            name=model_name,
            version=mv.version,
            stage="Archived"
        )
    
    # Promover la nueva versión
    client.transition_model_version_stage(
        name=model_name,
        version=model_version.version,
        stage="Production"
    )
    
    print(f"✅ Versión {model_version.version} promovida a Production exitosamente.")
    print(f"\nEl modelo está listo para ser usado por la API en AWS Lambda.")

if __name__ == "__main__":
    promote_best_model()
