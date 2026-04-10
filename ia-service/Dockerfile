FROM python:3.9-slim
WORKDIR /app

# Copia a lista de bibliotecas
COPY requirements.txt .

# Copia o código da API
COPY main.py .

# NOVIDADE: Copia o banco de dados e o modelo treinado
COPY dados_academia.csv .
COPY modelo_churn_academia.pkl . 

# Instala tudo
RUN pip install --no-cache-dir -r requirements.txt

EXPOSE 8000
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]