# Arquivo: app/app.py
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import numpy as np
import joblib
import tensorflow as tf
import yfinance as yf
import pandas as pd
import os

app = FastAPI(title="API Predictor WEGE3 - Tech Challenge 4")

# --- CONFIGURAÇÃO ---
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

class PredictionRequest(BaseModel):
    ticker: str

# Função auxiliar para limpar números que vêm do Pandas/Numpy
def to_float(valor):
    if hasattr(valor, 'item'):
        return float(valor.item())
    return float(valor)

@app.post("/predict")
def predict_stock(request: PredictionRequest):
    # 1. Trava de Segurança
    if request.ticker.upper() != TICKER_ALVO:
        raise HTTPException(
            status_code=400, 
            detail=f"Este modelo é exclusivo para {TICKER_ALVO}. Você enviou: {request.ticker}"
        )

    print(f"📥 Baixando dados recentes para {TICKER_ALVO}...")

    # 2. Baixar dados
    try:
        # reset_index traz a data para uma coluna normal
        df = yf.download(TICKER_ALVO, period="6mo", progress=False).reset_index()
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erro no Yahoo Finance: {str(e)}")

    if len(df) < 60:
        raise HTTPException(status_code=400, detail="Histórico insuficiente.")

    # 3. Tratamento de Data e Colunas
    # Se o yfinance trouxe a data como 'index', renomeia para 'Date'
    if 'Date' not in df.columns and 'index' in df.columns:
        df.rename(columns={'index': 'Date'}, inplace=True)
    
    # Garante que é datetime
    df['Date'] = pd.to_datetime(df['Date'])
    
    # Transformação do volume
    df['Volume'] = np.log1p(df['Volume'])

    # Pega os últimos 60 dias
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
        
        # Inverter normalização
        dummy = np.zeros((1, 5))
        dummy[:, 3] = prediction_scaled[0, 0]
        prediction_final = scaler.inverse_transform(dummy)[0, 3]

        # --- EXTRAÇÃO SEGURA DOS VALORES ---
        # Pegamos o último fechamento real
        fechamento_anterior_bruto = last_60_days['Close'].iloc[-1]
        
        # Usamos a função auxiliar para garantir que virou número Python
        fechamento_real = to_float(fechamento_anterior_bruto)
        previsao_real = to_float(prediction_final)
        
        # Pega a data de referência
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