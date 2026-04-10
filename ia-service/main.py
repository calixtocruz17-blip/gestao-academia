from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import pandas as pd
import joblib
import os

app = FastAPI()

# Permite que o site acesse esta IA
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Caminho da IA produzida no Colab
MODEL_PATH = "modelo_churn_academia.pkl"

@app.get("/")
def home():
    return {"status": "IA Online e Pronta"}

@app.post("/predict")
def predict_churn(data: dict):
    try:
        # 1. Extraímos os valores do dicionário enviado no teste
        freq = data.get('frequencia_semanal', 0)
        atraso = data.get('atrasos_pagamento', 0)

        # 2. Verifica se o arquivo .pkl existe dentro do Docker
        if os.path.exists(MODEL_PATH):
            # Carregamos o modelo usando joblib
            model = joblib.load(MODEL_PATH)
            
            # 3. Cria um (DataFrame) com os nomes exatos das colunas
            df = pd.DataFrame([[freq, atraso]], columns=['frequencia_semanal', 'atrasos_pagamento'])
            
            # 4. A IA faz a previsão real baseada no treinamento
            prediction = model.predict_proba(df)[0][1]
        else:
            # Caso o arquivo não seja encontrado, usar a lógica de reserva
            prediction = 0.8 if freq < 3 and atraso > 5 else 0.2

        # Isso resolve o erro "TypeError: 'numpy.bool' object is not iterable"
        risk_score = float(prediction)
        is_alert = bool(risk_score > 0.7)

        return {
            "status": "success",
            "churn_risk": round(risk_score, 2),
            "alert": is_alert
        }
    except Exception as e:
        # Retorna o erro real para facilitar o seu diagnóstico se algo falhar
        return {"status": "error", "message": str(e)}