# 225150207111001_1 MUHAMMAD NADHIF_1
# 225150201111002_2 NALENDRA MARCHELO_2
# 225150200111005_3 NARENDRA ATHA ABHINAYA_3
 # 225150200111003_4 YOSUA SAMUEL EDLYN SINAGA_4

# Dockerfile

FROM python:3.9-slim

# Tetapkan direktori kerja di dalam container
WORKDIR /app

# Salin file requirements.txt terlebih dahulu untuk caching layer
COPY requirements.txt .

# Install semua library yang dibutuhkan
RUN pip install --no-cache-dir -r requirements.txt

# Salin semua file dari proyek lokal ke dalam direktori kerja di container
COPY . .

RUN python app.py --train

# Ekspos port yang digunakan oleh Gradio
EXPOSE 7860

# menjalankan aplikasi Gradio
CMD ["python", "app.py"]