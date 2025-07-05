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

# [PENTING] Jalankan proses training model saat image Docker dibuat.
# memastikan file model.joblib tersedia di dalam container.
RUN python app.py --train

# Ekspos port yang digunakan oleh Gradio
EXPOSE 7860

# menjalankan aplikasi Gradio
CMD ["python", "app.py"]