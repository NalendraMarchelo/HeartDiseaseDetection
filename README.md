---
title: Prediksi Penyakit Jantung
emoji: 🩺
colorFrom: indigo
colorTo: purple
sdk: gradio
app_file: app.py
pinned: false
---

# 🩺 Prediksi Penyakit Jantung MLOps

Repositori ini adalah implementasi end-to-end dari siklus hidup Machine Learning (MLOps) untuk kasus prediksi penyakit jantung. Proyek ini mencakup semua tahapan, mulai dari pembuatan data sintetis, versioning, pelatihan model, monitoring, hingga retraining otomatis saat terdeteksi adanya data drift.

![Made with Python](https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white)
![Made with Scikit-learn](https://img.shields.io/badge/scikit--learn-%23F7931E.svg?style=for-the-badge&logo=scikit-learn&logoColor=white)
![Gradio](https://img.shields.io/badge/Gradio-FF7C00.svg?style=for-the-badge&logo=Gradio&logoColor=white)
![Made with Docker](https://img.shields.io/badge/Docker-2496ED?style=for-the-badge&logo=docker&logoColor=white)
![Made with Github Actions](https://img.shields.io/badge/GitHub_Actions-2088FF?style=for-the-badge&logo=github-actions&logoColor=white)

---

#### 📖 Ringkasan Proyek

Proyek ini bertujuan untuk membangun sebuah sistem yang tidak hanya dapat melatih model untuk mendeteksi penyakit jantung, tetapi juga mampu memantau performa model di lingkungan produksi. Jika terjadi pergeseran distribusi data (data drift), sistem dapat memberikan notifikasi dan memicu proses pelatihan ulang (retraining) untuk memastikan model tetap relevan dan akurat.

---

#### ⚙️ Alur Kerja MLOps

Berikut adalah tahapan alur kerja MLOps yang diimplementasikan dalam proyek ini:

1. Simulasi Data Baru: Data sintetis dibuat menggunakan CTGAN untuk mensimulasikan data baru yang akan masuk di lingkungan produksi. Data ini digunakan untuk menguji ketahanan model terhadap data drift.
2. Data Versioning: DVC (Data Version Control) digunakan untuk melacak versi dataset (data asli, sintetis, dan gabungan). Hal ini memastikan setiap eksperimen menggunakan data yang tepat dan dapat direproduksi.
3. Experiment Logging & Model Versioning: MLflow digunakan untuk mencatat semua eksperimen, termasuk parameter, metrik, dan artefak model. Model yang telah dilatih kemudian di-versi-kan dan didaftarkan di MLflow Model Registry.
4. Containerization: Seluruh komponen aplikasi, termasuk API, sistem monitoring, dan visualisasi, dibungkus dalam kontainer menggunakan Docker dan diorkestrasi dengan Docker Compose. Ini mencakup layanan untuk:

- - app: Aplikasi utama dengan API (FastAPI) dan UI (Gradio).
- - prometheus: Untuk scraping dan penyimpanan metrik.
- - grafana: Untuk visualisasi dasbor dan alerting.

5. Monitoring & Logging: Prometheus memantau endpoint metrik dari aplikasi untuk melacak fitur-fitur input secara real-time.
6. Alerting: Grafana digunakan untuk membuat dasbor visualisasi dari data Prometheus. Sistem alert dikonfigurasi untuk mengirim notifikasi ke Slack melalui webhook ketika nilai metrik melewati ambang batas yang ditentukan (data drift terdeteksi).
7. Retraining Model: Jika drift terdeteksi, proses pelatihan ulang dapat dijalankan menggunakan data gabungan (data lama + data baru) untuk menghasilkan versi model yang lebih baik. Model baru ini kemudian didaftarkan kembali di MLflow Registry dan siap untuk di-deploy.

---

#### 🛠️ Teknologi yang Digunakan

- Data Synthesis: ctgan
- Data & Model Versioning: DVC, MLflow
- ML Framework: scikit-learn
- API & Web App: FastAPI, Gradio
- Containerization: Docker, Docker Compose
- Monitoring: Prometheus
- Visualisasi & Alerting: Grafana
- Notifikasi: Slack
- CI/CD & Hosting: Dagshub, GitHub

---

#### 🚀 Instalasi dan Menjalankan Proyek

Untuk menjalankan tumpukan penuh (full stack) dari proyek ini, ikuti langkah-langkah berikut:

1. **Clone Repositori**

- - git clone https://github.com/NalendraMarchelo/HeartDiseaseDetection.git
- - cd HeartDiseaseDetection

2. **Setup Environment Variables**
   Buat file .env dan isi dengan konfigurasi yang diperlukan, seperti DAGSHUB_TOKEN jika diperlukan.
3. **Tarik Data dengan DVC**
   Pastikan DVC terinstal, lalu jalankan:

- - dvc pull

4. **jalankan Semua Layanan dengan Docker Compose**
   Pastikan Docker Desktop sudah berjalan. Kemudian, bangun dan jalankan semua kontainer:

- - docker-compose up --build

5. **Akses Layanan**
   Setelah semua kontainer berjalan, Anda dapat mengakses layanan berikut:

- - - Aplikasi Prediksi (Gradio UI): http://localhost:7860
- - - API Docs (FastAPI): http://localhost:8000/docs
- - - Prometheus UI: http://localhost:9090
- - - Grafana UI: http://localhost:3000
- - - MLflow UI: http://localhost:5000

---

#### 🔬 Menjalankan Pipeline Secara Manual

Anda juga bisa menjalankan setiap langkah secara individual.

1. Pelatihan Model Awal

- - python src/app.py dan train.py

2. Memulai UI Pelacakan Eksperimen

- - mlflow ui

3. Simulasi Prediksi untuk Menghasilkan Metrik
   Interaksi dengan UI Gradio di

- - http://localhost:7860 untuk membuat prediksi. Aksi ini akan menghasilkan metrik yang akan di-scrape oleh Prometheus.

4. Pelatihan Ulang (Retraining) saat Drift Terdeteksi
   Setelah mendapatkan notifikasi drift di Slack, jalankan skrip pelatihan ulang dengan data baru.

- - python train.py --data-path data/Heart_Disease_Prediction_Combined.csv --model-name "HeartDiseaseClassifier-Retrained"

---

#### 📈 Hasil Eksperimen

Model dievaluasi menggunakan beberapa metrik. Berikut adalah hasil dari model

RandomForest yang dicatat oleh MLflow:

- Accuracy: 0.759
- Precision: 0.764
- Recall: 0.989
- F1-score: 0.862

Setelah retraining dengan data gabungan, performa model meningkat:

- Accuracy: 0.843
- Precision: 0.829
- Recall: 0.965
- F1-score: 0.892

---

#### Link

- Link Github:
  https://github.com/NalendraMarchelo/HeartDiseaseDetection
- Link Hugging Face:
  https://huggingface.co/spaces/NalendraMarchelo/Heart-Disease-Prediction
- Link Dagshub:
  https://dagshub.com/NalendraMarchelo/HeartDiseaseDetection
- Link Slack:
  https://join.slack.com/t/mlops-semesterantara/shared_invite/zt-39nqr1yfn-5I8Ttn_TjZ_SwbbYoAfC5A

---

#### 👥 Kontributor

Proyek ini dikerjakan oleh kelompok mahasiswa dari Fakultas Ilmu Komputer, Universitas Brawijaya sebagai bagian dari mata kuliah Machine Learning Operations.

Muhammad Nadhif (225150207111001)

Nalendra Machelo (225150201111002)

Narendra Atha Abhinaya (225150200111005)

Yosua Samuel Edlyn Sinaga (225150200111003)

---

#### Dosen Pengampu

Putra Pandu Adikara, S.Kom., M.Kom.
