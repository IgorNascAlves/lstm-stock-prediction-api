# Arquivo: app/app.py
from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel
from prometheus_client import make_asgi_app, Counter, Histogram, Gauge
import numpy as np
import joblib
import tensorflow as tf
import yfinance as yf
import pandas as pd
import os
import time
import psutil  # Biblioteca para medir CPU e RAM

# --- INICIALIZAÇÃO DA API ---
app = FastAPI(title="API Predictor WEGE3 - Tech Challenge 4")

# --- CONFIGURAÇÃO DE MONITORAMENTO (PROMETHEUS) ---
metrics_app = make_asgi_app()
app.mount("/metrics", metrics_app)

# 1. Métrica de Desempenho (Tempo de Resposta)
REQUEST_LATENCY = Histogram(
    "app_request_latency_seconds", 
    "Tempo de resposta da API em segundos", 
    ["endpoint"]
)

# 2. Métrica de Volume (Contador de Requisições)
REQUEST_COUNT = Counter(
    "app_request_count_total", 
    "Total de requisições à API", 
    ["method", "endpoint", "status"]
)

# 3. Métrica de Recursos (CPU e RAM) - Gauge (Valor que sobe e desce)
SYSTEM_CPU_USAGE = Gauge("app_system_cpu_usage_percent", "Uso de CPU do sistema (%)")
APP_RAM_USAGE = Gauge("app_memory_usage_bytes", "Uso de Memória RAM da aplicação (bytes)")

# Middleware: Executado em TODA requisição para atualizar as métricas
@app.middleware("http")
async def monitor_requests(request: Request, call_next):
    start_time = time.time()
    
    # Processa a requisição
    response = await call_next(request)
    
    process_time = time.time() - start_time
    
    # Ignora a rota de métricas para não gerar ruído
    if request.url.path != "/metrics":
        # Atualiza latência e contagem
        REQUEST_LATENCY.labels(endpoint=request.url.path).observe(process_time)
        REQUEST_COUNT.labels(
            method=request.method, 
            endpoint=request.url.path, 
            status=response.status_code
        ).inc()
        
        # Atualiza métricas de recursos (Utilização de Recursos)
        # Pega o consumo de memória do processo atual
        process = psutil.Process(os.getpid())
        APP_RAM_USAGE.set(process.memory_info().rss)
        # Pega o uso de CPU (intervalo=None para não bloquear a chamada)
        SYSTEM_CPU_USAGE.set(psutil.cpu_percent(interval=None))
    
    return response

# --- CONFIGURAÇÃO DO MODELO ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, '../models/modelo_lstm.keras')
SCALER_PATH = os.path.join(BASE_DIR, '../models/scaler.joblib')
TICKER_ALVO = "WEGE3.SA"

# Carregar artefatos
try:
    model = tf.keras.models.load_model(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)
    print(f"✅ Modelo carregado para o ativo: {TICKER_ALVO}")
except Exception as e:
    print(f"❌ Erro ao carregar modelo/scaler: {e}")

# --- MODELOS DE DADOS ---
class PredictionRequest(BaseModel):
    ticker: str

def to_float(valor):
    if hasattr(valor, 'item'):
        return float(valor.item())
    return float(valor)

# --- ROTAS DA API ---
@app.post("/predict")
def predict_stock(request: PredictionRequest):
    # Trava de Segurança
    if request.ticker.upper() != TICKER_ALVO:
        raise HTTPException(
            status_code=400, 
            detail=f"Este modelo é exclusivo para {TICKER_ALVO}. Você enviou: {request.ticker}"
        )

    # 2. Baixar dados
    try:
        df = yf.download(TICKER_ALVO, period="6mo", progress=False).reset_index()
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erro no Yahoo Finance: {str(e)}")

    if len(df) < 60:
        raise HTTPException(status_code=400, detail="Histórico insuficiente.")

    # 3. Tratamento
    if 'Date' not in df.columns and 'index' in df.columns:
        df.rename(columns={'index': 'Date'}, inplace=True)
    
    df['Date'] = pd.to_datetime(df['Date'])
    df['Volume'] = np.log1p(df['Volume'])
    last_60_days = df.iloc[-60:]
    
    try:
        input_data = last_60_days[['Open', 'High', 'Low', 'Close', 'Volume']].values
    except KeyError:
        raise HTTPException(status_code=500, detail="Erro nas colunas do Yahoo Finance.")

    # 4. Previsão
    try:
        input_scaled = scaler.transform(input_data)
        input_reshaped = input_scaled.reshape(1, 60, 5)
        prediction_scaled = model.predict(input_reshaped)
        
        dummy = np.zeros((1, 5))
        dummy[:, 3] = prediction_scaled[0, 0]
        prediction_final = scaler.inverse_transform(dummy)[0, 3]

        fechamento_anterior_bruto = last_60_days['Close'].iloc[-1]
        fechamento_real = to_float(fechamento_anterior_bruto)
        previsao_real = to_float(prediction_final)
        data_ref = last_60_days['Date'].iloc[-1].date()

    except Exception as e:
        print(f"Erro detalhado: {e}")
        raise HTTPException(status_code=500, detail=f"Erro no cálculo: {str(e)}")

    return {
        "ticker": TICKER_ALVO,
        "data_referencia": str(data_ref),
        "fechamento_anterior": round(fechamento_real, 2),
        "previsao_para_amanha": round(previsao_real, 2)
    }

@app.get("/")
def home():
    return {"message": "API Online. Acesse /docs para testar ou /metrics para monitoramento."}
