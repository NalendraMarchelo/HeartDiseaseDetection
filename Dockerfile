# 225150207111001_1 MUHAMMAD NADHIF_1
# 225150201111002_2 NALENDRA MARCHELO_2
# 225150200111005_3 NARENDRA ATHA ABHINAYA_3
# 225150200111003_4 YOSUA SAMUEL EDLYN SINAGA_4

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