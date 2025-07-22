# 225150207111001_1 MUHAMMAD NADHIF_1
# 225150201111002_2 NALENDRA MARCHELO_2
# 225150200111005_3 NARENDRA ATHA ABHINAYA_3
# 225150200111003_4 YOSUA SAMUEL EDLYN SINAGA_4

import pandas as pd
import gradio as gr
import mlflow.pyfunc
import joblib
import os
import sys
import time
import threading
import logging
from flask import Flask, Response
from prometheus_client import Gauge, generate_latest, REGISTRY
from scipy.stats import wasserstein_distance, ks_2samp # Import untuk perhitungan drift

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# --- 1. SETUP PROMETHEUS METRICS ---
PREDICTION_GAUGE = Gauge('prediction_feature_value', 'Last value of a feature for prediction', ['feature_name'])
# Metrik baru untuk data drift
DATA_DRIFT_GAUGE = Gauge('data_drift_score', 'Data drift score for a feature', ['feature_name', 'metric_type'])
print("Metrik Prometheus didefinisikan.")

flask_app = Flask(__name__)
@flask_app.route("/metrics")
def get_metrics():
    return Response(generate_latest(REGISTRY), mimetype="text/plain")

# Variabel global untuk model, preprocessor, dan data referensi
model = None
scaler = None
imputer = None
REFERENCE_DATA = None # Akan diisi dengan data training awal

def load_model_and_preprocessors():
    """
    Fungsi ini memuat model "Production" terbaru dan preprocessor-nya dari MLflow/DagsHub.
    """
    global model, scaler, imputer   
    try:
        MODEL_NAME = "HeartDiseaseClassifier"
        MODEL_STAGE = "Production"
        model_uri = f"models:/{MODEL_NAME}/{MODEL_STAGE}"
        
        logger.info(f"Memuat model dari: {model_uri}")
        new_model = mlflow.pyfunc.load_model(model_uri) # Muat ke variabel lokal dulu

        client = mlflow.tracking.MlflowClient()
        # Menggunakan get_latest_versions yang akan deprecated, tapi masih berfungsi
        latest_version = client.get_latest_versions(name=MODEL_NAME, stages=[MODEL_STAGE])[0]
        run_id = latest_version.run_id
        
        logger.info(f"Mengunduh preprocessor dari run ID: {run_id}")
        # Mengunduh ke lokasi sementara atau dalam memori jika memungkinkan
        # Untuk kesederhanaan, kita tetap unduh ke root dir
        client.download_artifacts(run_id=run_id, path="scaler.joblib", dst_path=".")
        client.download_artifacts(run_id=run_id, path="imputer.joblib", dst_path=".")

        new_scaler = joblib.load("scaler.joblib")
        new_imputer = joblib.load("imputer.joblib")
        
        # Hanya ganti model dan preprocessor global jika berhasil semua
        model = new_model
        scaler = new_scaler
        imputer = new_imputer

        logger.info("Model dan preprocessor berhasil dimuat.")
        return True, f"Model version {latest_version.version} loaded successfully."

    except Exception as e:
        error_message = f"Error saat memuat model atau preprocessor: {e}"
        logger.error(error_message)
        return False, error_message

# --- FUNGSI BARU: Pengecekan Model Berkala ---
def check_for_model_updates(interval_seconds=600): # Cek setiap 10 menit
    global model, scaler, imputer
    MODEL_NAME = "HeartDiseaseClassifier"
    MODEL_STAGE = "Production"

    while True:
        try:
            client = mlflow.tracking.MlflowClient()
            latest_prod_version_in_registry = client.get_latest_versions(name=MODEL_NAME, stages=[MODEL_STAGE])[0]

            current_run_id = getattr(model, '_run_id', None) 
            
            if current_run_id != latest_prod_version_in_registry.run_id:
                logger.info(f"Model baru terdeteksi di Dagshub! Versi saat ini: {current_run_id}, Versi terbaru: {latest_prod_version_in_registry.run_id}")
                success, message = load_model_and_preprocessors()
                if success:
                    # Perbarui atribut run_id pada model yang baru dimuat
                    model._run_id = latest_prod_version_in_registry.run_id
                    logger.info(f"Berhasil memuat model versi terbaru: {latest_prod_version_in_registry.version}")
                else:
                    logger.error(f"Gagal memuat model terbaru: {message}")
            else:
                logger.info(f"Model saat ini sudah yang terbaru. Versi: {latest_prod_version_in_registry.version}")

        except Exception as e:
            logger.error(f"Error saat mengecek update model: {e}")
        
        time.sleep(interval_seconds)

# --- FUNGSI BARU: Perhitungan Data Drift ---
def calculate_data_drift(reference_df, current_df, feature_names_to_monitor):
    """
    Menghitung metrik data drift (Wasserstein Distance dan KS Statistic p-value)
    untuk fitur-fitur tertentu dan mengirimkannya ke Prometheus.
    """
    logger.info("Memulai perhitungan data drift...")
    for feature in feature_names_to_monitor:
        if feature in reference_df.columns and feature in current_df.columns:
            # Pastikan data numerik dan handle non-numerik jika ada
            ref_data = pd.to_numeric(reference_df[feature], errors='coerce').dropna()
            curr_data = pd.to_numeric(current_df[feature], errors='coerce').dropna()

            if not ref_data.empty and not curr_data.empty:
                try:
                    # Wasserstein Distance
                    wd = wasserstein_distance(ref_data, curr_data)
                    DATA_DRIFT_GAUGE.labels(feature_name=feature, metric_type='wasserstein_distance').set(wd)
                    logger.info(f"Drift untuk {feature} (Wasserstein): {wd:.4f}")

                    # Kolmogorov-Smirnov Test (p-value)
                    # ks_2samp membutuhkan setidaknya 2 sampel di setiap array
                    if len(ref_data) >= 2 and len(curr_data) >= 2:
                        ks_stat, p_value = ks_2samp(ref_data, curr_data)
                        DATA_DRIFT_GAUGE.labels(feature_name=feature, metric_type='ks_p_value').set(p_value)
                        # Catatan: p-value rendah (<0.05) menunjukkan perbedaan distribusi yang signifikan (drift)
                        logger.info(f"Drift untuk {feature} (KS p-value): {p_value:.4f}")
                    else:
                        logger.warning(f"Tidak cukup sampel untuk KS test pada fitur {feature}. Min 2 sampel diperlukan.")

                except Exception as e:
                    logger.warning(f"Gagal menghitung drift untuk fitur {feature}: {e}")
            else:
                logger.warning(f"Data referensi atau saat ini kosong untuk fitur {feature}. Tidak dapat menghitung drift.")
        else:
            logger.warning(f"Fitur '{feature}' tidak ditemukan di data referensi atau data saat ini.")
    logger.info("Perhitungan data drift selesai.")

# --- FUNGSI BARU: Pengecekan Data Drift Berkala ---
def check_for_data_drift(interval_seconds=180): # Cek setiap 3 menit
    global REFERENCE_DATA
    data_dir = 'data'
    logs_file = os.path.join(data_dir, 'new_logs.csv')
    
    # Fitur-fitur yang ingin dimonitor drift-nya (sesuaikan dengan dataset Anda)
    # Ini adalah fitur-fitur numerik yang paling mungkin mengalami drift
    # Pastikan nama-nama ini sesuai dengan kolom di old_data.csv / combined_data.csv
    features_to_monitor = ['Age', 'BP', 'Cholesterol', 'Max HR', 'ST depression'] 

    while True:
        try:
            if REFERENCE_DATA is None or REFERENCE_DATA.empty:
                logger.warning("REFERENCE_DATA belum dimuat atau kosong. Melewatkan pengecekan drift.")
                time.sleep(interval_seconds)
                continue

            if os.path.exists(logs_file) and os.path.getsize(logs_file) > 0:
                logger.info(f"Mendeteksi data baru di {logs_file} untuk perhitungan drift.")
                new_logs_df = pd.read_csv(logs_file)
                
                # Pastikan kolom-kolom yang relevan ada di new_logs_df sebelum memilih
                # Jika ada kolom yang tidak ada, ini akan menyebabkan KeyError
                missing_cols = [col for col in features_to_monitor if col not in new_logs_df.columns]
                if missing_cols:
                    logger.error(f"Kolom yang dimonitor tidak ditemukan di new_logs.csv: {missing_cols}. Melewatkan perhitungan drift.")
                    time.sleep(interval_seconds)
                    continue

                current_data_for_drift = new_logs_df[features_to_monitor]
                
                calculate_data_drift(REFERENCE_DATA, current_data_for_drift, features_to_monitor)
                
                # Setelah diproses, hapus file new_logs.csv
                os.remove(logs_file) 
                logger.info(f"File {logs_file} telah diproses dan dihapus.")
            else:
                logger.info(f"Tidak ada data baru di {logs_file}. Melewatkan perhitungan drift.")
        except pd.errors.EmptyDataError:
            logger.warning(f"File {logs_file} kosong. Menghapus file kosong.")
            # Hapus file kosong agar tidak terus-menerus dicek dan menyebabkan EmptyDataError
            if os.path.exists(logs_file):
                os.remove(logs_file)
        except Exception as e:
            logger.error(f"Error saat mengecek atau menghitung data drift: {e}")
        
        time.sleep(interval_seconds)


# --- 2. KONFIGURASI DAN PEMUATAN MODEL ---
def setup_mlflow_tracking():
    MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000")
    logger.info(f"Menggunakan MLflow Tracking URI: {MLFLOW_TRACKING_URI}")
    
    """Mengatur koneksi ke MLflow Tracking Server (DagsHub atau lokal)."""
    if os.getenv("DAGSHUB_TOKEN"):
        logger.info("Menggunakan DagsHub MLflow Tracking Server...")
        DAGSHUB_USER = "NalendraMarchelo"
        DAGSHUB_REPO = "HeartDiseaseDetection"
        os.environ['MLFLOW_TRACKING_USERNAME'] = DAGSHUB_USER
        os.environ['MLFLOW_TRACKING_PASSWORD'] = os.getenv("DAGSHUB_TOKEN")
        mlflow.set_tracking_uri(f"https://dagshub.com/{DAGSHUB_USER}/{DAGSHUB_REPO}.mlflow")
    else:
        logger.info("Menggunakan MLflow Tracking Server lokal...")
        mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)

setup_mlflow_tracking()

def log_prediction_data(feature_dict):
    """Mencatat data prediksi baru ke file log."""
    log_file = 'data/new_logs.csv'
    df_new = pd.DataFrame([feature_dict])
    
    # Jika file sudah ada, tambahkan tanpa header. Jika tidak, buat baru dengan header.
    if os.path.exists(log_file):
        df_new.to_csv(log_file, mode='a', header=False, index=False)
    else:
        df_new.to_csv(log_file, index=False)

# --- 3. FUNGSI PREDIKSI (DENGAN LOGGING METRIK) ---
def predict_heart_disease(Age, Sex, Chest_pain_type, BP, Cholesterol, FBS_over_120, EKG_results, Max_HR, Exercise_angina, ST_depression, Slope_of_ST, Number_of_vessels_fluro, Thallium):
    global model, scaler, imputer # Pastikan variabel global digunakan

    feature_names = ['Age', 'Sex', 'Chest pain type', 'BP', 'Cholesterol', 'FBS over 120', 
                     'EKG results', 'Max HR', 'Exercise angina', 'ST depression', 'Slope of ST', 
                     'Number of vessels fluro', 'Thallium']
    
    input_values = [Age, Sex, Chest_pain_type, BP, Cholesterol, FBS_over_120, EKG_results, 
                    Max_HR, Exercise_angina, ST_depression, Slope_of_ST, 
                    Number_of_vessels_fluro, Thallium]
    
    # Kirim nilai fitur mentah ke Prometheus sebelum diproses
    for feature, value in zip(feature_names, input_values):
        PREDICTION_GAUGE.labels(feature_name=feature).set(value)
    
    input_data = pd.DataFrame([input_values], columns=feature_names)
    
    # Preprocessing
    # Pastikan imputer dan scaler sudah dimuat
    if imputer is None or scaler is None or model is None:
        logger.error("Model atau preprocessor belum dimuat. Tidak dapat melakukan prediksi.")
        return "Error: Model tidak siap. Coba lagi nanti."

    # Periksa dan konversi kolom kategorikal jika perlu, sebelum imputasi/scaling
    # Asumsi: input_data sudah dalam format yang benar untuk imputer/scaler
    # Jika ada kolom non-numerik, mereka harus di-handle sebelum imputasi/scaling
    # Contoh: mengonversi string ke numerik sesuai mapping
    
    # Contoh sederhana untuk memastikan semua kolom numerik sebelum imputasi/scaling
    # Anda mungkin perlu logika yang lebih kompleks jika ada banyak kolom kategorikal
    # yang perlu di-encode sebelum scaling.
    # Untuk contoh ini, kita asumsikan input_data sudah siap untuk imputer/scaler
    
    try:
        input_imputed = imputer.transform(input_data)
        input_scaled = scaler.transform(input_imputed)
        input_processed = pd.DataFrame(input_scaled, columns=feature_names)
    except Exception as e:
        logger.error(f"Error during preprocessing: {e}")
        return "Error: Preprocessing gagal."
    
    # Prediksi
    prediction = model.predict(input_processed)[0]
    
    # --- BAGIAN INI UNTUK MENCATAT LOG PREDIKSI ---
    prediction_result = "Presence" if prediction == 1 else "Absence"
    log_entry = dict(zip(feature_names, input_values))
    log_entry['Prediction'] = prediction_result 
    log_prediction_data(log_entry)
    # --------------------------------------------------------    
    
    return "Berisiko Tinggi (Presence)" if prediction == 1 else "Berisiko Rendah (Absence)"

# --- 4. ANTARMUKA GRADIO ---
def create_gradio_interface():
    examples_list = [
        [35, "Wanita", "Typical Angina", 120, 190, "Tidak", "Normal", 170, "Tidak", 0.5, "Upsloping", 0, "Normal"],
        [42, "Pria", "Non-anginal Pain", 130, 210, "Tidak", "Normal", 165, "Tidak", 0.0, "Upsloping", 0, "Normal"],
        [65, "Pria", "Asymptomatic", 155, 280, "Ya", "Hipertrofi Ventrikel Kiri", 120, "Ya", 2.5, "Flat", 2, "Reversible Defect"],
        [58, "Wanita", "Atypical Angina", 160, 320, "Tidak", "Abnormalitas ST-T", 115, "Ya", 3.0, "Downsloping", 3, "Fixed Defect"]
    ]

    with gr.Blocks(theme=gr.themes.Default(), title="Prediksi Penyakit Jantung") as demo:
        gr.Markdown("# 🩺 Aplikasi Prediksi Penyakit Jantung")
        gr.Markdown("Masukkan data medis pasien untuk memprediksi risiko penyakit jantung.")

        with gr.Row():
            with gr.Column():
                age_input = gr.Slider(label="Usia", minimum=29, maximum=77, step=1, value=54)
                sex_input = gr.Radio(label="Jenis Kelamin", choices=["Wanita", "Pria"], value="Pria")
                cp_input = gr.Dropdown(label="Jenis Nyeri Dada", choices=["Typical Angina", "Atypical Angina", "Non-anginal Pain", "Asymptomatic"], value="Non-anginal Pain")
            with gr.Column():
                bp_input = gr.Slider(label="Tekanan Darah", minimum=94, maximum=200, step=1, value=131)
                chol_input = gr.Slider(label="Kolesterol", minimum=126, maximum=564, step=1, value=249)
                max_hr_input = gr.Slider(label="Detak Jantung Maks", minimum=71, maximum=202, step=1, value=149)
        with gr.Row():
            with gr.Column():
                fbs_input = gr.Radio(label="Gula Darah > 120", choices=["Tidak", "Ya"], value="Tidak")
                ekg_input = gr.Dropdown(label="Hasil EKG", choices=["Normal", "Abnormalitas ST-T", "Hipertrofi Ventrikel Kiri"], value="Abnormalitas ST-T")
                exang_input = gr.Radio(label="Angina Saat Olahraga", choices=["Tidak", "Ya"], value="Tidak")
                st_depression_input = gr.Slider(label="ST Depression", minimum=0.0, maximum=6.2, step=0.1, value=1.0)
                slope_input = gr.Dropdown(label="Slope ST", choices=["Upsloping", "Flat", "Downsloping"], value="Flat")
                vessels_input = gr.Dropdown(label="Jumlah Pembuluh Terlihat", choices=[0,1,2,3], value=0)
                thallium_input = gr.Dropdown(label="Thallium", choices=["Normal", "Fixed Defect", "Reversible Defect"], value="Normal")
            with gr.Column():
                predict_btn = gr.Button("🔮 Lakukan Prediksi", variant="primary")
                output_label = gr.Label(label="Status Risiko")

        # Fungsi wrapper untuk mapping input teks dari UI/Examples ke angka
        def wrapped_predict(Age, Sex_str, cp_str, BP, Chol, FBS_str, ekg_str, Max_HR, exang_str, ST_dep, slope_str, vessel, thallium_str):
            mapping = {
                "cp": {"Typical Angina": 1, "Atypical Angina": 2, "Non-anginal Pain": 3, "Asymptomatic": 4},
                "ekg": {"Normal": 0, "Abnormalitas ST-T": 1, "Hipertrofi Ventrikel Kiri": 2},
                "slope": {"Upsloping": 1, "Flat": 2, "Downsloping": 3},
                "thallium": {"Normal": 3, "Fixed Defect": 6, "Reversible Defect": 7}
            }
            Sex = 0 if Sex_str == "Wanita" else 1
            FBS = 0 if FBS_str == "Tidak" else 1
            exang = 0 if exang_str == "Tidak" else 1
            return predict_heart_disease(
                Age, Sex, mapping["cp"][cp_str], BP, Chol, FBS, mapping["ekg"][ekg_str], Max_HR, 
                exang, ST_dep, mapping["slope"][slope_str], vessel, mapping["thallium"][thallium_str])
        
        inputs_list = [
            age_input, sex_input, cp_input, bp_input, chol_input, fbs_input, 
            ekg_input, max_hr_input, exang_input, st_depression_input, slope_input, 
            vessels_input, thallium_input]
        
        predict_btn.click(fn=wrapped_predict, inputs=inputs_list, outputs=output_label)

        gr.Examples(
            examples=examples_list,
            inputs=inputs_list,
            outputs=output_label,
            fn=wrapped_predict,
            cache_examples=False) # Set cache_examples to False to avoid potential caching issues during development
    return demo

# --- BLOK EKSEKUSI UTAMA (RESTRUKTURISASI) ---
if __name__ == "__main__":
    logger.info("Memulai aplikasi...")

    # 1. Pemuatan Model Awal
    logger.info("Melakukan pemuatan model awal...")
    initial_load_success, message = load_model_and_preprocessors()
    if not initial_load_success:
        logger.error(f"Gagal memuat model saat startup: {message}. Aplikasi akan berhenti.")
        sys.exit(1)
    else:
        # PENTING: Setelah pemuatan model awal, simpan run_id model yang dimuat
        client = mlflow.tracking.MlflowClient()
        latest_version = client.get_latest_versions(name="HeartDiseaseClassifier", stages=["Production"])[0]
        # Pastikan 'model' tidak None di sini sebelum menambahkan atribut
        if model:
            model._run_id = latest_version.run_id
            logger.info(f"Model awal berhasil dimuat: versi {latest_version.version}, run_id {latest_version.run_id}")

    # 2. Pemuatan Data Referensi untuk Drift Detection
    data_dir = 'data'
    old_file = os.path.join(data_dir, 'old_data.csv')
    combined_file = os.path.join(data_dir, 'combined_data.csv')

    if os.path.exists(combined_file):
        REFERENCE_DATA = pd.read_csv(combined_file)
        logger.info(f"Memuat data referensi dari {combined_file}. Jumlah baris: {len(REFERENCE_DATA)}")
    elif os.path.exists(old_file):
        REFERENCE_DATA = pd.read_csv(old_file)
        logger.info(f"Memuat data referensi dari {old_file}. Jumlah baris: {len(REFERENCE_DATA)}")
    else:
        logger.error("Tidak dapat menemukan data referensi (combined_data.csv atau old_data.csv). Drift detection mungkin tidak berfungsi.")
        REFERENCE_DATA = pd.DataFrame() # Inisialisasi kosong agar tidak error lebih lanjut

    # 3. Inisialisasi dan Jalankan Threads
    # Flask server untuk Prometheus
    flask_thread = threading.Thread(target=lambda: flask_app.run(host='0.0.0.0', port=8000), daemon=True)
    flask_thread.start()
    logger.info("Flask server untuk Prometheus berjalan di port 8000.")

    # Thread untuk pengecekan update model
    model_update_thread = threading.Thread(target=check_for_model_updates, args=(600,), daemon=True) # Cek setiap 10 menit
    model_update_thread.start()
    logger.info("Thread pengecekan update model berjalan.")

    # Thread untuk pengecekan data drift
    data_drift_thread = threading.Thread(target=check_for_data_drift, args=(180,), daemon=True) # Cek setiap 3 menit
    data_drift_thread.start()
    logger.info("Thread pengecekan data drift berjalan.")

    # 4. Menjalankan aplikasi Gradio
    logger.info("Menjalankan aplikasi Gradio...")
    demo = create_gradio_interface()
    demo.launch(server_name="0.0.0.0", server_port=7860)

