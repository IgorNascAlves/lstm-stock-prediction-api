import requests
import random
import json

# URL da sua API local
url = "http://127.0.0.1:8000/predict-data"

# Gerar 60 dias de dados fictícios perfeitos
dados_ficticios = []
for _ in range(60):
    dados_ficticios.append({
        "Open": 30.0 + random.random(),
        "High": 31.0 + random.random(),
        "Low": 29.0 + random.random(),
        "Close": 30.5 + random.random(),
        "Volume": 1000000.0
    })

payload = {"data": dados_ficticios}

print("Enviando requisição...")
try:
    response = requests.post(url, json=payload)
    
    if response.status_code == 200:
        print("✅ SUCESSO!")
        print(json.dumps(response.json(), indent=2))
    else:
        print(f"❌ ERRO {response.status_code}:")
        print(response.text)
except Exception as e:
    print(f"Erro de conexão: {e}")