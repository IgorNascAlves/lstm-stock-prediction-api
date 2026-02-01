# 📈 LSTM Stock Prediction API - Tech Challenge Fase 4

Este projeto compõe a entrega da **Fase 4 do Tech Challenge**. O objetivo é desenvolver um sistema preditivo utilizando Deep Learning (**LSTM**) para prever o preço de fechamento de ações da **WEGE3 (Weg S.A.)**, disponibilizando o modelo através de uma API RESTful containerizada com Docker.

## 📋 Sobre o Projeto

O sistema coleta dados históricos do Yahoo Finance, processa as informações utilizando uma rede neural **Long Short-Term Memory (LSTM)** treinada para identificar padrões temporais e prevê o preço de fechamento do próximo dia (`Close`).

A aplicação garante que apenas o ticker para o qual o modelo foi treinado (`WEGE3.SA`) seja processado, garantindo a integridade da inferência.

## 📂 Estrutura de Pastas

```text
lstm-stock-prediction-api/
│
├── app/
│   └── app.py                # Código fonte da API (FastAPI)
│
├── models/
│   ├── modelo_lstm.keras     # Modelo LSTM treinado
│   └── scaler.joblib         # Normalizador (MinMaxScaler) ajustado
│
├── notebooks/
│   └── Modelo_preditivo.ipynb # Notebook usado para análise e treinamento
│
├── Dockerfile                # Receita para containerização
├── requirements.txt          # Dependências do projeto
└── README.md                 # Documentação

```

---

## 🚀 Guia de Instalação e Execução

Você pode rodar este projeto de três formas: **Notebook (Treino)**, **Python Local (Dev)** ou **Docker (Produção)**.

### 1. 🐳 Rodando com Docker (Recomendado)

Esta é a forma mais simples de testar a entrega final. Certifique-se de ter o Docker Desktop instalado e rodando.

1. **Construa a imagem:**
```bash
docker build -t tech-challenge-api .

```


2. **Execute o container:**
```bash
docker run -p 8000:8000 tech-challenge-api

```


3. **Acesse a documentação:** Abra `http://localhost:8000/docs` no navegador.

---

### 2. 🐍 Rodando Localmente com Python

Para desenvolvimento ou teste sem Docker.

1. **Crie um ambiente virtual (opcional mas recomendado):**
```bash
python -m venv venv
# Windows:
venv\Scripts\activate
# Linux/Mac:
source venv/bin/activate

```


2. **Instale as dependências:**
```bash
pip install -r requirements.txt

```


3. **Inicie o servidor:**
```bash
uvicorn app.app:app --reload

```



---

### 3. 📓 Rodando o Notebook de Treinamento

Caso queira ver como o modelo foi criado ou retreiná-lo.

1. Navegue até a pasta `notebooks`.
2. Abra o arquivo `.ipynb` no Jupyter Notebook, VS Code ou Google Colab.
3. Instale as dependências necessárias dentro do notebook (`%pip install ...`).
4. Execute as células sequencialmente.
* **Nota:** Ao final da execução, o notebook salvará novos arquivos `.keras` e `.joblib` na pasta `models`.



---

## 🧪 Como Testar a API

A API possui documentação automática (Swagger UI).

1. Com a API rodando (Docker ou Local), acesse: **`http://localhost:8000/docs`**
2. Vá até o endpoint **`POST /predict`**.
3. Clique em **Try it out**.
4. Insira o JSON abaixo:

**✅ Teste de Sucesso:**

```json
{
  "ticker": "WEGE3.SA"
}

```

*Retorno esperado: Código 200 com a previsão do preço.*

**❌ Teste de Validação (Erro esperado):**

```json
{
  "ticker": "PETR4.SA"
}

```

*Retorno esperado: Código 400 informando que o modelo é exclusivo para WEGE3.*

---

## 🛠️ Tecnologias Utilizadas

* **Linguagem:** Python 3.9
* **Machine Learning:** TensorFlow (Keras), Scikit-learn
* **API:** FastAPI, Uvicorn, Pydantic
* **Dados:** Yahoo Finance (`yfinance`), Pandas, NumPy
* **DevOps:** Docker

## 📝 Notas de Arquitetura

* **Versionamento de Modelos:** Os arquivos binários (`modelo_lstm.keras` e `scaler.joblib`) foram incluídos no repositório para facilitar a clonagem e avaliação imediata do projeto, dispensando a necessidade de retreino por parte do avaliador.
* **Robustez:** A API implementa tratamento de erros para falhas de conexão com o Yahoo Finance e validação estrita do Ticker de entrada.