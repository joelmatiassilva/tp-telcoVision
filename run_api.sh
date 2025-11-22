#!/bin/bash

# Script para ejecutar la API localmente
# Proyecto: resolucion/telco_prod

echo "🚀 Iniciando TelcoVision API..."
echo ""

# Verificar que el archivo .env existe
if [ ! -f ".env" ]; then
    echo "⚠️  ADVERTENCIA: No se encontró archivo .env"
    echo "   Copia el archivo .env desde resolucion/telco/.env"
    echo "   cp ../telco/.env ."
    echo ""
    read -p "¿Deseas continuar sin .env? (s/N): " continue
    if [[ ! $continue =~ ^[Ss]$ ]]; then
        exit 1
    fi
fi

echo "✓ Activando entorno conda..."
echo "✓ Iniciando servidor en http://localhost:8000"
echo ""
echo "Endpoints disponibles:"
echo "  - GET  /          : Health check básico"
echo "  - GET  /health    : Health check detallado"
echo "  - POST /predict   : Predicción de churn"
echo "  - GET  /docs      : Documentación interactiva"
echo ""
echo "Presiona CTRL+C para detener el servidor"
echo ""

# Ejecutar con conda
cd "$(dirname "$0")"
conda run -n pycaret-env python -c "
import sys
sys.path.insert(0, '.')
from src.api.app import app
import uvicorn
uvicorn.run(app, host='0.0.0.0', port=8000)
"
