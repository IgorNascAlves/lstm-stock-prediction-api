import streamlit as st
import requests
import pandas as pd
import plotly.graph_objects as go
import yfinance as yf
from datetime import datetime, timedelta

# --- CONFIGURAÇÃO DA PÁGINA ---
st.set_page_config(
    page_title="Preditor WEGE3",
    page_icon="gw",
    layout="wide"
)

# --- CONSTANTES ---
# Se estiver rodando localmente sem Docker, use localhost
# Se estiver usando Docker Compose, usaria o nome do container (ex: "http://backend:8000")
API_URL = "http://127.0.0.1:8000"
TICKER = "WEGE3.SA"

# --- CABEÇALHO ---
st.title("📈 Dashboard de Previsão - WEG S.A.")
st.markdown(f"""
Este dashboard conecta-se à **API de Deep Learning (LSTM)** para prever o preço de fechamento da **{TICKER}**.
""")

# --- SIDEBAR (CONTROLES) ---
st.sidebar.header("Painel de Controle")
st.sidebar.info("O modelo foi treinado exclusivamente para WEGE3 com dados de 60 dias.")

if st.sidebar.button("Realizar Previsão Agora", type="primary"):
    with st.spinner('Consultando a API...'):
        try:
            # 1. Chama a API para pegar a previsão
            response = requests.post(f"{API_URL}/predict", json={"ticker": TICKER})
            
            if response.status_code == 200:
                data = response.json()
                fechamento = data['fechamento_anterior']
                previsao = data['previsao_para_amanha']
                data_ref = data['data_referencia']
                
                # --- ÁREA DE DESTAQUE (METRICS) ---
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.metric("Data de Referência", datetime.strptime(data_ref, "%Y-%m-%d").strftime("%d/%m/%Y"))
                
                with col2:
                    st.metric("Último Fechamento", f"R$ {fechamento:.2f}")
                    
                with col3:
                    delta = previsao - fechamento
                    st.metric(
                        "Previsão para Amanhã", 
                        f"R$ {previsao:.2f}", 
                        delta=f"{delta:.2f} R$",
                        delta_color="normal"
                    )

                # --- GRÁFICO ---
                st.subheader("Histórico Recente + Previsão")
                
                # Baixa dados apenas para visualização (últimos 3 meses)
                # Nota: O Frontend baixa dados para desenhar o gráfico, mas quem calcula é a API
                df_hist = yf.download(TICKER, period="3mo", progress=False).reset_index()
                
                # Criar gráfico com Plotly
                fig = go.Figure()

                # Linha histórica
                fig.add_trace(go.Scatter(
                    x=df_hist['Date'], 
                    y=df_hist['Close'], 
                    mode='lines', 
                    name='Histórico Real',
                    line=dict(color='#1f77b4')
                ))

                # Ponto de Previsão
                # Adiciona 1 dia útil à última data (aproximação simples para visualização)
                last_date = df_hist['Date'].iloc[-1]
                next_date = last_date + timedelta(days=1)
                
                fig.add_trace(go.Scatter(
                    x=[last_date, next_date],
                    y=[fechamento, previsao],
                    mode='lines+markers',
                    name='Tendência Prevista',
                    line=dict(color='green', dash='dot')
                ))

                st.plotly_chart(fig, use_container_width=True)
                
            else:
                st.error(f"Erro na API: {response.text}")
                
        except Exception as e:
            st.error(f"Não foi possível conectar à API. Verifique se ela está rodando.\nErro: {e}")

# --- MONITORIZAÇÃO (BÔNUS) ---
st.divider()
st.subheader("🔍 Monitorização da API (Health Check)")

if st.button("Atualizar Métricas de Sistema"):
    try:
        metrics_res = requests.get(f"{API_URL}/metrics")
        if metrics_res.status_code == 200:
            lines = metrics_res.text.split('\n')
            
            # Parser simples para extrair valores do Prometheus
            cpu = 0
            ram = 0
            reqs = 0
            
            for line in lines:
                if "app_system_cpu_usage_percent" in line and "#" not in line:
                    cpu = float(line.split(' ')[1])
                if "app_memory_usage_bytes" in line and "#" not in line:
                    ram = float(line.split(' ')[1]) / 1024 / 1024 # Converter para MB
                if "app_request_count_total" in line and "#" not in line and 'status="200"' in line:
                    # Soma as requisições 200 OK
                    reqs += float(line.split(' ')[1])

            m1, m2, m3 = st.columns(3)
            m1.metric("Uso de CPU", f"{cpu:.1f}%")
            m2.metric("Uso de RAM", f"{ram:.1f} MB")
            m3.metric("Total de Requisições (200 OK)", int(reqs))
            
            with st.expander("Ver Log Bruto do Prometheus"):
                st.text(metrics_res.text)
                
    except:
        st.warning("Não foi possível ler as métricas.")