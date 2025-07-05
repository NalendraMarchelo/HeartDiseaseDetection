---
title: Prediksi Penyakit Jantung
emoji: 🩺
colorFrom: indigo
colorTo: purple
sdk: gradio
app_file: app.py
pinned: false
---

# 🩺 Prediksi Penyakit Jantung - Proyek MLOps

Proyek ini bertujuan untuk membangun sistem prediksi penyakit jantung menggunakan model machine learning dan menerapkan praktik MLOps untuk otomatisasi, mulai dari pelatihan model hingga deployment aplikasi web interaktif.

#### Teknologi Utama:

- Model: Scikit-learn (RandomForestClassifier)
- Aplikasi Web: Gradio
- Kontainerisasi: Docker
- CI/CD: GitHub Actions
- Deployment: Hugging Face Space

#### 📁 Struktur Folder

Berikut adalah struktur folder utama dari proyek ini dan penjelasan singkatnya:
├── .github/workflows/
│ └── main.yml # Konfigurasi GitHub Actions untuk CI/CD ke Hugging Face
├── data/
│ └── Heart_Disease_Prediction.csv # Dataset yang digunakan
├── model/
│ └── # Folder ini akan berisi model.joblib setelah dilatih
├── app.py # Kode utama aplikasi (fungsi training & antarmuka Gradio)
├── Dockerfile # Resep untuk membangun container Docker
├── notebook.ipynb # Notebook untuk eksplorasi data (EDA) dan pengujian awal
└── requirements.txt # Daftar library Python yang dibutuhkan

- app.py adalah file final yang akan dijalankan di produksi (di Docker dan Hugging Face).
- notebook.ipynb digunakan sebagai tempat eksplorasi, analisis, dan pengujian logika model. Alur di notebook ini telah disesuaikan agar cocok dengan logika final di app.py.

#### 🚀 Memulai (Panduan)

Ikuti langkah-langkah ini untuk menjalankan proyek di komputer lokal.

1. Requirements

- Git
- Python 3.9+
- Docker Desktop

2. Instalasi Proyek

- Salin repositori ini ke komputer dan install semua library yang dibutuhkan.

### 1. Clone repositori dari GitHub

- **git clone https://github.com/QAmatAS/HeartDisesaseDetection.git**

### 2. Masuk ke direktori proyek

- **cd HeartDisesaseDetection**

### 3. Buat dan aktifkan virtual environment

Di Mac/Linux:

- **python3 -m venv venv**
- **source venv/bin/activate**

Di Windows:

- **python -m venv venv**
- **venv\Scripts\activate**

### 4. Install semua library yang diperlukan

- **pip install -r requirements.txt**
  📊 Eksplorasi & Pelatihan Model (Menggunakan Notebook)
  Untuk memahami data dan logika model, bisa membuka dan menjalankan file notebook.ipynb.

#### 🐳 Menjalankan Aplikasi dengan Docker (Lokal)

Cara menjalankan aplikasi seperti saat di-deployment.

1. Hentikan dan Hapus Container Lama (Jika Ada)
   Jika pernah menjalankan container ini sebelumnya, bersihkan dulu untuk menghindari error.

- **docker stop heart-disease-container**
- **docker rm heart-disease-container**

2. Bangun (Build) Image Docker
   Perintah ini akan membaca Dockerfile untuk membangun image yang berisi semua yang dibutuhkan aplikasi. Proses ini juga akan melatih model secara otomatis (--train).

- **docker build -t heart-disease-app .**

3. Jalankan (Run) Container Docker
   Setelah image berhasil dibuat, jalankan sebagai container.
   Opsi --rm akan otomatis menghapus container saat dihentikan

- **docker run --rm -p 7860:7860 --name heart-disease-container heart-disease-app**

4. Akses Aplikasi
   Buka browser web dan kunjungi alamat http://localhost:7860.

#### ⚙️ Alur Kerja CI/CD (Otomatisasi)

Proyek ini menggunakan GitHub Actions untuk proses Continuous Integration/Continuous Deployment (CI/CD).

- Pemicu (Trigger): Setiap kali ada push ke branch main.
- Aksi (Action): GitHub akan secara otomatis mengambil seluruh kode dari repositori dan menyalinnya (sync) ke Hugging Face Space yang telah dikonfigurasi.
- Kebutuhan: Proses ini memerlukan secret bernama HF_TOKEN yang disimpan di Settings > Secrets and variables > Actions pada repositori ini.

#### 👥 Tim Pengembang

- 225150207111001 - Muhammad Nadhif
- 225150201111002 - Nalendra Machelo
- 225150200111005 - Narendra Atha Abhinaya
- 225150200111003 - Yosua Samuel Edlyn Sinaga
