"""
Script de prueba local para verificar que el modelo en Production
puede ser cargado desde MLflow (DagsHub) y usado para predicciones.

Este script simula exactamente lo que hace src/app.py en producción.
"""

import mlflow
from mlflow.tracking import MlflowClient
import pandas as pd
import os
from dotenv import load_dotenv
from datetime import datetime

# Cargar variables de entorno
load_dotenv()

def show_model_registry_info():
    """Muestra información detallada del Model Registry"""
    
    print("\n" + "=" * 60)
    print("INFORMACIÓN DEL MODEL REGISTRY")
    print("=" * 60)
    
    # Configurar MLflow
    tracking_uri = os.getenv("MLFLOW_TRACKING_URI")
    if tracking_uri:
        mlflow.set_tracking_uri(tracking_uri)
    
    client = MlflowClient()
    model_name = os.getenv("MLFLOW_MODEL_NAME", "telco-churn-prediction")
    
    try:
        # Obtener información del modelo registrado
        registered_model = client.get_registered_model(model_name)
        
        print(f"\n📦 Modelo: {registered_model.name}")
        print(f"   Descripción: {registered_model.description or 'Sin descripción'}")
        
        # Obtener todas las versiones
        all_versions = client.search_model_versions(f"name='{model_name}'")
        
        print(f"\n📊 Total de versiones registradas: {len(all_versions)}")
        
        # Agrupar por stage
        versions_by_stage = {}
        for version in all_versions:
            stage = version.current_stage
            if stage not in versions_by_stage:
                versions_by_stage[stage] = []
            versions_by_stage[stage].append(version)
        
        # Mostrar versión en Production
        if "Production" in versions_by_stage:
            print("\n" + "=" * 60)
            print("🟢 VERSIÓN EN PRODUCTION (ACTIVA)")
            print("=" * 60)
            
            for version in versions_by_stage["Production"]:
                print(f"\n   Versión: {version.version}")
                print(f"   Run ID: {version.run_id}")
                print(f"   Fecha de registro: {datetime.fromtimestamp(version.creation_timestamp/1000).strftime('%Y-%m-%d %H:%M:%S')}")
                print(f"   Última actualización: {datetime.fromtimestamp(version.last_updated_timestamp/1000).strftime('%Y-%m-%d %H:%M:%S')}")
                
                # Obtener métricas del run
                try:
                    run = client.get_run(version.run_id)
                    print(f"\n   📈 Métricas del Modelo:")
                    
                    metrics_to_show = ['final_accuracy', 'final_f1', 'final_auc', 'final_precision', 'final_recall']
                    for metric_name in metrics_to_show:
                        if metric_name in run.data.metrics:
                            value = run.data.metrics[metric_name]
                            print(f"      • {metric_name.replace('final_', '').title()}: {value:.4f}")
                    
                    # Intentar obtener el nombre del modelo/algoritmo
                    if run.data.params:
                        print(f"\n   ⚙️ Parámetros del Modelo:")
                        # Mostrar algunos parámetros clave
                        key_params = ['model_name', 'algorithm', 'n_estimators', 'max_depth']
                        for param_name in key_params:
                            if param_name in run.data.params:
                                print(f"      • {param_name}: {run.data.params[param_name]}")
                    
                except Exception as e:
                    print(f"      (No se pudieron cargar métricas: {e})")
        
        # Mostrar otras versiones
        other_stages = [stage for stage in versions_by_stage.keys() if stage != "Production"]
        if other_stages:
            print("\n" + "=" * 60)
            print("📚 OTRAS VERSIONES")
            print("=" * 60)
            
            for stage in sorted(other_stages):
                print(f"\n   Stage: {stage}")
                for version in sorted(versions_by_stage[stage], key=lambda v: v.version, reverse=True):
                    print(f"      • Versión {version.version} - Registrada: {datetime.fromtimestamp(version.creation_timestamp/1000).strftime('%Y-%m-%d %H:%M:%S')}")
        
        return True
        
    except Exception as e:
        print(f"\n❌ ERROR al obtener información del registro: {e}")
        return False

def test_model_loading():
    """Prueba la carga del modelo desde MLflow"""
    
    print("\n" + "=" * 60)
    print("TEST: Carga de Modelo desde MLflow (DagsHub)")
    print("=" * 60)
    
    # Configurar MLflow
    tracking_uri = os.getenv("MLFLOW_TRACKING_URI")
    if tracking_uri:
        mlflow.set_tracking_uri(tracking_uri)
        print(f"\n✓ MLflow URI configurado: {tracking_uri}")
    else:
        print("\n⚠ MLFLOW_TRACKING_URI no configurado en .env")
        print("  Usando configuración por defecto (local)")
    
    # Configuración del modelo
    model_name = os.getenv("MLFLOW_MODEL_NAME", "telco-churn-prediction")
    model_stage = os.getenv("MLFLOW_MODEL_STAGE", "Production")
    
    print(f"\nIntentando cargar modelo:")
    print(f"  Nombre: {model_name}")
    print(f"  Stage: {model_stage}")
    
    try:
        # Cargar modelo
        model_uri = f"models:/{model_name}/{model_stage}"
        print(f"\n  URI: {model_uri}")
        print("\n  Descargando modelo desde DagsHub...")
        
        model = mlflow.pyfunc.load_model(model_uri)
        
        print("\n✅ ÉXITO: Modelo cargado correctamente desde DagsHub")
        
        return model
    
    except Exception as e:
        print(f"\n❌ ERROR al cargar modelo: {e}")
        print("\nPosibles causas:")
        print("  1. El modelo no existe en MLflow")
        print("  2. El modelo no está en stage 'Production'")
        print("  3. Las credenciales en .env son incorrectas")
        print("  4. No hay conexión a DagsHub")
        return None

def test_prediction(model):
    """Prueba una predicción con el modelo"""
    
    print("\n" + "=" * 60)
    print("TEST: Predicción con Datos de Ejemplo")
    print("=" * 60)
    
    # Datos de ejemplo (mismo formato que CustomerData en app.py)
    sample_data = {
        "customer_id": "TEST-001",  # ID de prueba
        "age": 35,
        "gender": "Male",
        "region": "North",
        "contract_type": "Month-to-Month",
        "tenure_months": 6,
        "monthly_charges": 65.5,
        "total_charges": 393.0,
        "internet_service": "Fiber optic",
        "phone_service": "Yes",
        "multiple_lines": "No",
        "payment_method": "Electronic check"
    }
    
    print("\nDatos de entrada:")
    for key, value in sample_data.items():
        print(f"  {key}: {value}")
    
    try:
        # Convertir a DataFrame
        df = pd.DataFrame([sample_data])
        
        print("\n  Ejecutando predicción...")
        prediction = model.predict(df)
        
        result = prediction[0]
        
        print(f"\n✅ ÉXITO: Predicción completada")
        print(f"\n  Resultado: {result}")
        
        if isinstance(result, (int, float)):
            if int(result) == 1:
                print("  Interpretación: Cliente con RIESGO DE CHURN")
            else:
                print("  Interpretación: Cliente SIN RIESGO DE CHURN")
        
        return True
    
    except Exception as e:
        print(f"\n❌ ERROR en predicción: {e}")
        return False

def main():
    """Función principal de prueba"""
    
    print("\n" + "=" * 60)
    print("PRUEBA LOCAL: Consumo de Modelo desde DagsHub")
    print("Proyecto: resolucion/telco_prod")
    print("=" * 60)
    
    # Verificar archivo .env
    if not os.path.exists(".env"):
        print("\n⚠ ADVERTENCIA: No se encontró archivo .env")
        print("\nPara que este test funcione, necesitas crear un archivo .env con:")
        print("""
MLFLOW_TRACKING_URI=https://dagshub.com/joelmatiassilva/tp-labMineriaDeDatos-telco.mlflow
MLFLOW_TRACKING_USERNAME=joelmatiassilva
MLFLOW_TRACKING_PASSWORD=tu_token_de_dagshub
""")
        print("\nPuedes copiar el .env desde resolucion/telco/.env")
        return
    
    # Test 0: Mostrar información del Model Registry
    registry_ok = show_model_registry_info()
    
    if not registry_ok:
        print("\n⚠ Continuando sin información del registry...")
    
    # Test 1: Cargar modelo
    model = test_model_loading()
    
    if model is None:
        print("\n" + "=" * 60)
        print("❌ TEST FALLIDO: No se pudo cargar el modelo")
        print("=" * 60)
        return
    
    # Test 2: Hacer predicción
    success = test_prediction(model)
    
    # Resumen final
    print("\n" + "=" * 60)
    if success:
        print("✅ TODOS LOS TESTS PASARON")
        print("\nEl modelo en Production está listo para ser usado en Lambda")
    else:
        print("❌ ALGUNOS TESTS FALLARON")
    print("=" * 60 + "\n")

if __name__ == "__main__":
    main()
