# src/evaluate.py (con MLflow)
import mlflow
import pandas as pd
import yaml
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix, roc_curve
import matplotlib.pyplot as plt
import seaborn as sns
import os
from dotenv import load_dotenv

def evaluate_model():
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

    experiment_name = "telco-churn-prediction"

    # Cargar datos de prueba (el mismo hold-out set que usó PyCaret)
    df = pd.read_csv(params['data_read_csv'])
    
    data_unseen = df.sample(frac=1-params['train_size'], random_state=params['seed'])
    X_unseen = data_unseen.drop('churn', axis=1)
    y_unseen = data_unseen['churn']

    # --- Búsqueda del mejor modelo en MLflow ---
    # Buscar el mejor run dentro del experimento, ordenado por Accuracy
    best_run = mlflow.search_runs(
        experiment_names=[experiment_name],
        order_by=["metrics.Accuracy DESC"],
        max_results=1
    ).iloc[0]

    best_run_id = best_run["run_id"]
    print(f"Mejor Run ID encontrado: {best_run_id}")
    print(f"Métricas del mejor run: Accuracy = {best_run['metrics.Accuracy']:.4f}")

    # Cargar el modelo del mejor run
    model_uri = f"runs:/{best_run_id}/model"
    loaded_model = mlflow.pyfunc.load_model(model_uri)

    # --- Evaluación del modelo cargado ---
    # Predecir clases y probabilidades
    predictions = loaded_model.predict(X_unseen)
    # Algunos modelos devuelven probabilidades de forma diferente, intentamos predict_proba si es posible
    try:
        # PyCaret/Sklearn models usually have predict_proba
        # loaded_model es un PyFuncModel, necesitamos el underlying model o usar predict con params si es soportado
        # Para simplificar con mlflow pyfunc, a veces devuelve solo clases.
        # Si el modelo fue logueado con sklearn flavor, podríamos cargarlo como sklearn.
        # Pero pyfunc es genérico. Vamos a asumir que predict devuelve clases.
        # Para ROC necesitamos probabilidades.
        # Intentaremos obtener probabilidades si el flavor lo permite o si predict devuelve df con probas (raro).
        # NOTA: MLflow pyfunc predict output depends on the model.
        # Si no podemos obtener probas fácilmente de pyfunc genérico sin más info,
        # cargaremos como sklearn si es posible, o omitiremos ROC si falla.
        
        # Intento de cargar como sklearn para tener predict_proba
        sklearn_model = mlflow.sklearn.load_model(model_uri)
        probs = sklearn_model.predict_proba(X_unseen)[:, 1]
        has_probs = True
    except Exception as e:
        print(f"No se pudo cargar como modelo sklearn para probabilidades: {e}")
        print("Se intentará usar pyfunc, pero ROC/AUC podría no estar disponible.")
        probs = None
        has_probs = False

    # Calcular métricas de evaluación final
    final_accuracy = accuracy_score(y_unseen, predictions)
    final_precision = precision_score(y_unseen, predictions)
    final_recall = recall_score(y_unseen, predictions)
    final_f1 = f1_score(y_unseen, predictions)
    
    metrics = {
        "final_accuracy": final_accuracy,
        "final_precision": final_precision,
        "final_recall": final_recall,
        "final_f1": final_f1
    }

    if has_probs:
        final_auc = roc_auc_score(y_unseen, probs)
        metrics["final_auc"] = final_auc
        print(f"  AUC-ROC: {final_auc:.4f}")

    print("--- Métricas de Evaluación Final en Hold-Out Set ---")
    print(f"  Accuracy: {final_accuracy:.4f}")
    print(f"  Precision: {final_precision:.4f}")
    print(f"  Recall: {final_recall:.4f}")
    print(f"  F1-Score: {final_f1:.4f}")

    # --- Generación de Gráficos ---
    os.makedirs("outputs/plots", exist_ok=True)

    # 1. Matriz de Confusión
    cm = confusion_matrix(y_unseen, predictions)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title('Matriz de Confusión - Hold-Out Set')
    plt.ylabel('Verdadero')
    plt.xlabel('Predicho')
    plt.tight_layout()
    cm_path = "outputs/plots/confusion_matrix.png"
    plt.savefig(cm_path)
    plt.close()

    # 2. Curva ROC (si hay probabilidades)
    roc_path = None
    if has_probs:
        fpr, tpr, _ = roc_curve(y_unseen, probs)
        plt.figure(figsize=(8, 6))
        plt.plot(fpr, tpr, label=f'AUC = {final_auc:.2f}')
        plt.plot([0, 1], [0, 1], 'k--')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Curva ROC - Hold-Out Set')
        plt.legend(loc='lower right')
        plt.tight_layout()
        roc_path = "outputs/plots/roc_curve.png"
        plt.savefig(roc_path)
        plt.close()

    # --- Loguear las métricas y artefactos al run original de MLflow ---
    with mlflow.start_run(run_id=best_run_id):
        mlflow.log_metrics(metrics)
        mlflow.log_artifact(cm_path)
        if roc_path:
            mlflow.log_artifact(roc_path)

    print("Métricas y gráficos de evaluación final logueados en el run de MLflow existente.")
    
    # --- Guardar métricas también en formato JSON para DVC ---
    import json
    os.makedirs("outputs/metrics", exist_ok=True)
    with open("outputs/metrics/metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)
    
    print("Métricas guardadas también en outputs/metrics/metrics.json para DVC")

if __name__ == "__main__":
    evaluate_model()