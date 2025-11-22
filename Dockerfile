FROM public.ecr.aws/lambda/python:3.9

# Copiar requirements.txt
COPY requirements.txt ${LAMBDA_TASK_ROOT}

# Instalar dependencias
# --no-cache-dir para reducir tamaño
RUN pip install --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copiar código fuente
COPY src/ ${LAMBDA_TASK_ROOT}/src/
COPY params.yaml ${LAMBDA_TASK_ROOT}

# Configurar el CMD para el handler de Mangum
# src.api.app.handler apunta al objeto Mangum(app) en src/api/app.py
CMD [ "src.api.app.handler" ]
