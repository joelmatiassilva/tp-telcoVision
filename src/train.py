# src/train.py (usando PyCaret y MLflow)
from pycaret.classification import *
import yaml
import pandas as pd
import os
import mlflow
from dotenv import load_dotenv

def train_model():
    # Cargar parámetros PRIMERO
    with open('params.yaml') as f:
        params = yaml.safe_load(f)

    # --- MLflow setup ---
    # --- MLflow setup ---
    track_remote = params.get('track_to_dagshub', False) or os.getenv('TRACK_TO_DAGSHUB') == 'true'
    
    if track_remote:
        # Si vamos a DagsHub, cargar credenciales y configurar URI
        load_dotenv()
        # Si la URI viene por variable de entorno (CI), usarla. Si no, usar params.
        tracking_uri = os.getenv('MLFLOW_TRACKING_URI') or params.get('dagshub_tracking_uri')
        if tracking_uri:
            mlflow.set_tracking_uri(tracking_uri)
    else:
        # Si es local, asegurarse de que no haya URI de tracking de DagsHub
        if 'MLFLOW_TRACKING_URI' in os.environ:
            del os.environ['MLFLOW_TRACKING_URI']
        if 'MLFLOW_TRACKING_USERNAME' in os.environ:
            del os.environ['MLFLOW_TRACKING_USERNAME']
        if 'MLFLOW_TRACKING_PASSWORD' in os.environ:
            del os.environ['MLFLOW_TRACKING_PASSWORD']
        
    # Cargar datos procesados
    df = pd.read_csv(params['data_read_csv'])
    
    # Setup PyCaret - esto leerá las variables de entorno de MLflow y lo configurará todo
    exp = ClassificationExperiment()
    exp.setup(
        data=df, 
        target='churn',
        train_size=params['train_size'],
        session_id=params['seed'],
        log_experiment=True, # Activar logging a MLflow
        experiment_name="telco-churn-prediction"
    )
    
    # Comparar modelos automáticamente (PyCaret logueará todo)
    best_models = exp.compare_models(
        include=params['models_to_compare'],
        sort=params['metric'],
        n_select=3
    )
    
    # Seleccionar y tunear el mejor
    best_model = exp.tune_model(best_models[0])
    
    # Finalizar modelo
    final_model = exp.finalize_model(best_model)
    
    # No es necesario guardar el modelo localmente, MLflow se encarga
    # print("Modelo guardado en MLflow")

if __name__ == "__main__":
    train_model()
