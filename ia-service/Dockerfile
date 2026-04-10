# Imagem do Python já pronta
FROM python:3.9-slim

# Local do código no Docker
WORKDIR /app

# Copiar para o Docker
COPY requirements.txt .
COPY main.py .
COPY dados_academia.csv .

# Instala as bibliotecas 
RUN pip install --no-cache-dir -r requirements.txt

# Avisa que a IA vai rodar na porta 8000
EXPOSE 8000

# Comando para ligar a IA
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]