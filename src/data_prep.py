# src/data_prep.py (versión simplificada)
import pandas as pd
import os

def prepare_data():
    # Carga de datos
    df = pd.read_csv('data/raw/telco_churn.csv')
    
    # Solo limpieza básica manual si es necesaria
    df_clean = df.copy()
    
    # Crear el directorio de salida si no existe
    output_dir = 'data/processed'
    os.makedirs(output_dir, exist_ok=True)
    
    # Guardar los datos procesados
    df_clean.to_csv(os.path.join(output_dir, 'telco_churn_processed.csv'), index=False)

if __name__ == "__main__":
    prepare_data()
