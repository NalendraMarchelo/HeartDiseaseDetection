# check_drift.py (for evidently==0.4.0)
import pandas as pd
from evidently.report import Report
from evidently.metric_preset import DataDriftPreset
import mlflow
import os

# Muat data referensi
try:
    reference_data = pd.read_csv('data/old_data.csv')
    reference_data.dropna(subset=['Heart Disease'], inplace=True)
    print("Data referensi (old_data.csv) berhasil dimuat.")
except FileNotFoundError:
    print("Error: Pastikan file 'data/old_data.csv' ada.")
    exit()

# Muat data saat ini
try:
    current_data = pd.read_csv('data/synthetic_data.csv')
    current_data.dropna(subset=['Heart Disease'], inplace=True)
    print("Data saat ini (synthetic_data.csv) berhasil dimuat.")
except FileNotFoundError:
    print("Error: Pastikan file 'data/synthetic_data.csv' ada.")
    exit()

# Buat laporan drift data
print("Membuat laporan data drift...")
data_drift_report = Report(metrics=[
    DataDriftPreset(),
])

# Jalankan perbandingan
data_drift_report.run(reference_data=reference_data, current_data=current_data)

# Simpan laporan sebagai file HTML
report_path = 'data_drift_report.html'
data_drift_report.save_html(report_path)  # Changed from save() to save_html()

print(f"\n✅ Laporan data drift berhasil dibuat!")
print(f"Buka file '{report_path}' di browser Anda untuk melihat hasilnya.")

# --- Integrasi dengan MLflow (Opsional) ---
try:
    if os.getenv("DAGSHUB_TOKEN"):
        DAGSHUB_USER = "NalendraMarchelo"
        DAGSHUB_REPO = "HeartDiseaseDetection"
        os.environ['MLFLOW_TRACKING_USERNAME'] = DAGSHUB_USER
        os.environ['MLFLOW_TRACKING_PASSWORD'] = os.getenv("DAGSHUB_TOKEN")
        mlflow.set_tracking_uri(f"https://dagshub.com/{DAGSHUB_USER}/{DAGSHUB_REPO}.mlflow")
    else:
        mlflow.set_tracking_uri("http://localhost:5000")

    mlflow.set_experiment("Analisis Drift")

    with mlflow.start_run(run_name="Pengecekan Drift Data Sintetis"):
        mlflow.log_artifact(report_path)
        print("Laporan drift berhasil dicatat sebagai artefak di MLflow.")

except Exception as e:
    print(f"\nTidak dapat terhubung ke MLflow, melewati pencatatan artefak. Error: {e}")