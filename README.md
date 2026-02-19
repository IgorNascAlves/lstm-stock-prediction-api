# 📈 LSTM Stock Prediction API & Dashboard - Tech Challenge Fase 4

Este projeto compõe a entrega da **Fase 4 do Tech Challenge da Pós Tech**. O objetivo é desenvolver um sistema preditivo utilizando Deep Learning (**LSTM**) para prever o preço de fechamento de ações da **WEGE3 (Weg S.A.)**, disponibilizando o modelo através de uma API RESTful (FastAPI), com monitorização (Prometheus) e um Dashboard interativo (Streamlit).

## 📋 Sobre o Projeto

O sistema foi treinado com dados históricos do Yahoo Finance, utilizando uma rede neural **Long Short-Term Memory (LSTM)** para identificar padrões temporais e prever o preço de fechamento do próximo dia (`Close`). 

Para atender a todos os requisitos arquiteturais e práticos, o projeto é composto por:
1. **API Backend (FastAPI):** Disponibiliza a inferência do modelo e métricas de sistema.
2. **Dashboard Frontend (Streamlit):** Interface gráfica para facilitar o uso da API e visualizar a saúde do sistema.
3. **Monitorização Integrada:** Captura de latência, contagem de requisições, uso de CPU e RAM.

## 📂 Estrutura de Pastas
```text
lstm-stock-prediction-api/
│
├── app/
│   └── app.py                 # Código fonte da API (FastAPI) com Prometheus
│
├── models/
│   ├── modelo_lstm.keras      # Modelo LSTM treinado
│   └── scaler.joblib          # Normalizador (MinMaxScaler) ajustado
│
├── notebooks/
│   └── Modelo_preditivo_LSTM_Tech_Challenge_Fase_04.ipynb # Treinamento e Análise
│
├── frontend.py                # Dashboard Interativo (Streamlit)
├── script.py                  # Script automatizado para testar o endpoint /predict-data
├── Dockerfile                 # Receita para containerização da API
├── requirements.txt           # Dependências do projeto
└── README.md                  # Documentação

```

---

## 🚀 Guia de Instalação e Execução (Local)

Siga os passos abaixo para rodar a aplicação diretamente na sua máquina.

### 1. Preparação do Ambiente
Recomenda-se o uso de um ambiente virtual para evitar conflitos de bibliotecas.
Abra o terminal na raiz do projeto e execute:

**No Windows:**
```bash
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
```

**No Linux/Mac:**
```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### 2. ⚙️ Rodando a API (Backend)

Com as dependências instaladas, inicie o servidor FastAPI usando o Uvicorn. O comando abaixo usa a flag `--reload`, que atualiza a API automaticamente se você alterar o código.

```bash
uvicorn app.app:app --reload

```

* A API estará acessível em: `http://127.0.0.1:8000`
* A documentação automática (Swagger) estará em: `http://127.0.0.1:8000/docs`


### 3. 🖥️ Rodando o Dashboard (Streamlit)

Com a API rodando (seja via Docker ou local), abra um novo terminal e execute o frontend:

```bash
pip install streamlit requests pandas plotly yfinance
streamlit run frontend.py

```

*O Dashboard abrirá automaticamente no seu navegador.*

---

## 🧪 Endpoints e Como Testar

A API possui documentação automática acessível em: **`http://localhost:8000/docs`**

O sistema possui dois endpoints principais para previsão, desenhados para diferentes cenários:

### 1. Endpoint Oficial (Requisito da Pós Tech): `POST /predict-data`

Recebe o histórico bruto de 60 dias diretamente no corpo da requisição e devolve a previsão. Não depende de internet para baixar dados.

* **Como testar rapidamente:** Use o arquivo `script.py` fornecido no projeto. Ele gera automaticamente uma carga de 60 dias de dados simulados e faz a requisição correta para a API.

```bash
python script.py

```

### 2. Endpoint Prático: `POST /predict`

Recebe apenas o ticker da ação (`WEGE3.SA`), vai ao Yahoo Finance, baixa os últimos 60 dias reais, processa e devolve a previsão.

* **Como testar:** Através do Swagger UI (`/docs`) ou diretamente no **Dashboard Streamlit**.

---

## 📊 Monitorização e Escalabilidade

A API expõe métricas nativas no formato **Prometheus** através do endpoint `GET /metrics`.
O nosso *middleware* customizado captura:

* `app_request_latency_seconds`: Tempo de resposta das requisições.
* `app_request_count_total`: Volume de chamadas por endpoint.
* `app_system_cpu_usage_percent`: Consumo de CPU do servidor.
* `app_memory_usage_bytes`: Consumo de RAM da aplicação.

*(Nota: Estes dados podem ser visualizados de forma amigável no botão "Atualizar Métricas de Sistema" dentro do Dashboard Streamlit).*