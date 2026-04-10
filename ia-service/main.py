from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import pandas as pd
import joblib
import os

app = FastAPI()

# Permite que o seu site (Vercel) acesse esta IA
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Caminho do cérebro da IA que você baixou do Colab
MODEL_PATH = "modelo_churn_academia.pkl"

@app.get("/")
def home():
    return {"status": "IA Online e Pronta"}

@app.post("/predict")
def predict_churn(data: dict):
    try:
        # 1. Extraímos os valores do dicionário enviado no teste
        # Padronizamos para 'frequencia_semanal', como no seu treino do Colab
        freq = data.get('frequencia_semanal', 0)
        atraso = data.get('atrasos_pagamento', 0)

        # 2. Verificamos se o arquivo .pkl real existe dentro do Docker
        if os.path.exists(MODEL_PATH):
            # Carregamos o modelo usando joblib
            model = joblib.load(MODEL_PATH)
            
            # 3. Criamos uma "mini tabela" (DataFrame) com os nomes exatos das colunas
            # Isso resolve o erro de nomes que o modelo exige
            df = pd.DataFrame([[freq, atraso]], columns=['frequencia_semanal', 'atrasos_pagamento'])
            
            # 4. A IA faz a previsão real baseada no treinamento
            prediction = model.predict_proba(df)[0][1]
        else:
            # Caso o arquivo não seja encontrado, usamos a lógica de reserva
            prediction = 0.8 if freq < 3 and atraso > 5 else 0.2

        # 5. CORREÇÃO TÉCNICA: Convertemos tipos Numpy para tipos padrão do Python
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