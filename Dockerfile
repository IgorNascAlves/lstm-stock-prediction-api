# Usa uma imagem leve do Python
FROM python:3.9-slim

# Define a pasta de trabalho dentro do container
WORKDIR /code

# 1. Copia e instala as dependências (faz isso antes para aproveitar o cache)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 2. Copia as pastas do projeto
# Copia a pasta app para dentro de /code/app
COPY ./app ./app
# Copia a pasta models para dentro de /code/models
COPY ./models ./models

# Expõe a porta 8000
EXPOSE 8000

# Comando para rodar a API
# "--host 0.0.0.0" é OBRIGATÓRIO no Docker para aceitar conexões externas
CMD ["uvicorn", "app.app:app", "--host", "0.0.0.0", "--port", "8000"]