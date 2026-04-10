from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import pandas as pd
import joblib
import os

app = FastAPI()

# Permite que o Next.js (frontend) acesse esta IA
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Caminho do modelo treinado
MODEL_PATH = "modelo_churn_academia.pkl"

@app.get("/")
def home():
    return {"status": "IA Online"}

@app.post("/predict")
def predict_churn(data: dict):
    try:
        # Se você já tiver o arquivo .pkl do Colab:
        if os.path.exists(MODEL_PATH):
            model = joblib.load(MODEL_PATH)
            df = pd.DataFrame([data])
            prediction = model.predict_proba(df)[0][1]
        else:
            # Lógica de teste caso o arquivo .pkl ainda não esteja na pasta
            freq = data.get('frequencia_mes', 10)
            atraso = data.get('atrasos_pagamento', 0)
            prediction = 0.8 if freq < 8 and atraso > 5 else 0.2

        return {
            "status": "success",
            "churn_risk": round(float(prediction), 2),
            "alert": prediction > 0.7
        }
    except Exception as e:
        return {"status": "error", "message": str(e)}