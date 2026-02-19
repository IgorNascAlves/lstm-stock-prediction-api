# Arquivo: app/app.py
from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel, Field
from typing import List
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

# --- MODELOS DE DADOS (PYDANTIC) ---
class TickerRequest(BaseModel):
    ticker: str = Field(..., example="WEGE3.SA", description="Ticker da ação no Yahoo Finance")

class DailyData(BaseModel):
    Open: float
    High: float
    Low: float
    Close: float
    Volume: float

class HistoryRequest(BaseModel):
    data: List[DailyData] = Field(
        ..., 
        min_items=60, 
        max_items=60, 
        description="Lista exata de 60 dias de dados históricos (OHLCV)."
    )

def to_float(valor):
    if hasattr(valor, 'item'):
        return float(valor.item())
    return float(valor)

# --- FUNÇÕES DE LÓGICA DE NEGÓCIO ---

def download_stock_data(ticker: str) -> pd.DataFrame:
    """Baixa os últimos 60 dias de dados do Yahoo Finance e formata o DataFrame."""
    try:
        df = yf.download(ticker, period="6mo", progress=False).reset_index()
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erro no Yahoo Finance: {str(e)}")

    if len(df) < 60:
        raise HTTPException(status_code=400, detail="Histórico insuficiente no Yahoo Finance.")

    # Tratamento de colunas de Data
    if 'Date' not in df.columns and 'index' in df.columns:
        df.rename(columns={'index': 'Date'}, inplace=True)
    
    df['Date'] = pd.to_datetime(df['Date'])
    
    # Retorna apenas os últimos 60 dias
    return df.iloc[-60:].copy()


def make_prediction(input_data: np.ndarray) -> float:
    """Recebe um array (60, 5), aplica o scaler, prevê e desfaz o scaler."""
    try:
        # Scale
        input_scaled = scaler.transform(input_data)
        # Reshape para (1, 60, 5)
        input_reshaped = input_scaled.reshape(1, 60, 5)
        
        # Previsão
        prediction_scaled = model.predict(input_reshaped)
        
        # Inverse Transform (colocando a previsão na coluna 3 que é o Close)
        dummy = np.zeros((1, 5))
        dummy[:, 3] = prediction_scaled[0, 0]
        prediction_final = scaler.inverse_transform(dummy)[0, 3]
        
        return to_float(prediction_final)
    except Exception as e:
        print(f"Erro detalhado na previsão: {e}")
        raise HTTPException(status_code=500, detail=f"Erro no cálculo do modelo: {str(e)}")

# --- ROTAS DA API ---

@app.post("/predict", tags=["Yahoo Finance"])
def predict_stock(request: TickerRequest):
    """Consulta o Yahoo Finance e realiza a previsão do próximo dia."""
    # Trava de Segurança
    if request.ticker.upper() != TICKER_ALVO:
        raise HTTPException(
            status_code=400, 
            detail=f"Este modelo é exclusivo para {TICKER_ALVO}. Você enviou: {request.ticker}"
        )

    # 1. Download de dados (Função isolada)
    last_60_days = download_stock_data(TICKER_ALVO)
    
    # 2. Preparação específica (aplicar log no volume)
    last_60_days['Volume'] = np.log1p(last_60_days['Volume'])
    
    try:
        input_data = last_60_days[['Open', 'High', 'Low', 'Close', 'Volume']].values
    except KeyError:
        raise HTTPException(status_code=500, detail="Erro nas colunas retornadas pelo Yahoo Finance.")

    # 3. Realiza a Previsão (Função isolada)
    previsao_real = make_prediction(input_data)

    # 4. Formata a resposta
    fechamento_real = to_float(last_60_days['Close'].iloc[-1])
    data_ref = last_60_days['Date'].iloc[-1].date()

    return {
        "ticker": TICKER_ALVO,
        "data_referencia": str(data_ref),
        "fechamento_anterior": round(fechamento_real, 2),
        "previsao_para_amanha": round(previsao_real, 2)
    }


@app.post("/predict-data", tags=["Dados Fornecidos"])
def predict_by_data(request: HistoryRequest):
    """Recebe dados históricos enviados pelo utilizador (Requisito da Pos Tech) e faz a previsão."""
    
    # 1. Preparação dos dados fornecidos
    data_list = []
    for item in request.data:
        # Aplica o log1p no volume exatamente como foi treinado e usado no outro endpoint
        vol = np.log1p(item.Volume) if item.Volume > 0 else 0
        data_list.append([item.Open, item.High, item.Low, item.Close, vol])
    
    input_data = np.array(data_list)
    
    if input_data.shape != (60, 5):
        raise HTTPException(status_code=400, detail="Formato de dados incorreto. Necessário matriz (60, 5).")

    # 2. Realiza a Previsão (Função isolada reaproveitada)
    previsao_real = make_prediction(input_data)

    # 3. Formata a resposta
    last_close = request.data[-1].Close

    return {
        "msg": "Previsão baseada nos dados históricos fornecidos.",
        "fechamento_anterior_fornecido": round(last_close, 2),
        "previsao_para_amanha": round(previsao_real, 2)
    }