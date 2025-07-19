# Dockerfile.app

FROM python:3.9-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY app.py .
COPY train.py .
# COPY scaler.joblib .
# COPY imputer.joblib .

EXPOSE 7860

CMD ["python", "app.py"]