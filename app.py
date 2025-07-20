# app.py (Final)
import pandas as pd
import gradio as gr
import mlflow.pyfunc
import joblib
import os
import sys

# --- 1. KONFIGURASI DAN PEMUATAN MODEL ---
def setup_mlflow_tracking():
    """Mengatur koneksi ke MLflow Tracking Server (DagsHub atau lokal)."""
    DAGSHUB_USER = "NalendraMarchelo"
    DAGSHUB_REPO = "HeartDiseaseDetection"
    
    # Prioritaskan koneksi ke DagsHub jika kredensial ada (untuk Hugging Face)
    if os.getenv("DAGSHUB_TOKEN"):
        print("Menggunakan DagsHub MLflow Tracking Server...")
        os.environ['MLFLOW_TRACKING_USERNAME'] = DAGSHUB_USER
        os.environ['MLFLOW_TRACKING_PASSWORD'] = os.getenv("DAGSHUB_TOKEN")
        mlflow.set_tracking_uri(f"https://dagshub.com/{DAGSHUB_USER}/{DAGSHUB_REPO}.mlflow")
    else:
        # Fallback ke server lokal jika tidak ada konfigurasi lain (untuk development)
        print("Menggunakan MLflow Tracking Server lokal...")
        mlflow.set_tracking_uri("http://localhost:5000")

# Panggil fungsi setup saat aplikasi dimulai
setup_mlflow_tracking()

try:
    MODEL_NAME = "HeartDiseaseClassifier"
    MODEL_STAGE = "Production"
    model_uri = f"models:/{MODEL_NAME}/{MODEL_STAGE}"
    
    print(f"Memuat model dari: {model_uri}")
    model = mlflow.pyfunc.load_model(model_uri)
    
    client = mlflow.tracking.MlflowClient()
    latest_version = client.get_latest_versions(name=MODEL_NAME, stages=[MODEL_STAGE])[0]
    run_id = latest_version.run_id
    
    print(f"Mengunduh preprocessor dari run ID: {run_id}")
    client.download_artifacts(run_id=run_id, path="scaler.joblib", dst_path=".")
    client.download_artifacts(run_id=run_id, path="imputer.joblib", dst_path=".")

    scaler = joblib.load("scaler.joblib")
    imputer = joblib.load("imputer.joblib")
    
    print("Model dan preprocessor berhasil dimuat.")

except Exception as e:
    print(f"Error saat memuat model atau preprocessor: {e}")
    sys.exit(1)


# --- 2. FUNGSI PREDIKSI ---
def predict_heart_disease(Age, Sex, Chest_pain_type, BP, Cholesterol, FBS_over_120, EKG_results, Max_HR, Exercise_angina, ST_depression, Slope_of_ST, Number_of_vessels_fluro, Thallium):
    feature_names = ['Age', 'Sex', 'Chest pain type', 'BP', 'Cholesterol', 'FBS over 120', 
                     'EKG results', 'Max HR', 'Exercise angina', 'ST depression', 'Slope of ST', 
                     'Number of vessels fluro', 'Thallium']
    
    input_data = pd.DataFrame([[Age, Sex, Chest_pain_type, BP, Cholesterol, FBS_over_120, EKG_results, 
                                Max_HR, Exercise_angina, ST_depression, Slope_of_ST, 
                                Number_of_vessels_fluro, Thallium]], columns=feature_names)
    
    input_imputed = imputer.transform(input_data)
    input_scaled = scaler.transform(input_imputed)
    input_processed = pd.DataFrame(input_scaled, columns=feature_names)
    
    prediction = model.predict(input_processed)[0]
    return "Berisiko Tinggi (Presence)" if prediction == 1 else "Berisiko Rendah (Absence)"


# --- 3. ANTARMUKA GRADIo (Sesuai Kode Lama Anda) ---
def create_gradio_interface():
    # Contoh input dari kode lama Anda (dalam format teks)
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
                sex_input = gr.Radio(label="Jenis Kelamin", choices=["Wanita", "Pria"], value="Pria", type="index")
                cp_input = gr.Dropdown(label="Jenis Nyeri Dada", choices=["Typical Angina", "Atypical Angina", "Non-anginal Pain", "Asymptomatic"], value="Non-anginal Pain")
            with gr.Column():
                bp_input = gr.Slider(label="Tekanan Darah", minimum=94, maximum=200, step=1, value=131)
                chol_input = gr.Slider(label="Kolesterol", minimum=126, maximum=564, step=1, value=249)
                max_hr_input = gr.Slider(label="Detak Jantung Maks", minimum=71, maximum=202, step=1, value=149)
        with gr.Row():
            with gr.Column():
                fbs_input = gr.Radio(label="Gula Darah > 120", choices=["Tidak", "Ya"], value="Tidak", type="index")
                ekg_input = gr.Dropdown(label="Hasil EKG", choices=["Normal", "Abnormalitas ST-T", "Hipertrofi Ventrikel Kiri"], value="Abnormalitas ST-T")
                exang_input = gr.Radio(label="Angina Saat Olahraga", choices=["Tidak", "Ya"], value="Tidak", type="index")
                st_depression_input = gr.Slider(label="ST Depression", minimum=0.0, maximum=6.2, step=0.1, value=1.0)
                slope_input = gr.Dropdown(label="Slope ST", choices=["Upsloping", "Flat", "Downsloping"], value="Flat")
                vessels_input = gr.Dropdown(label="Jumlah Pembuluh Terlihat", choices=[0,1,2,3], value=0)
                thallium_input = gr.Dropdown(label="Thallium", choices=["Normal", "Fixed Defect", "Reversible Defect"], value="Normal")
            with gr.Column():
                predict_btn = gr.Button("🔮 Lakukan Prediksi", variant="primary")
                output_label = gr.Label(label="Status Risiko")

        # Fungsi wrapper untuk mapping string ke angka
        def wrapped_predict(Age, Sex_str, cp_str, BP, Chol, FBS_str, ekg_str, Max_HR, exang_str, ST_dep, slope_str, vessel, thallium_str):
            mapping_cp = {"Typical Angina": 1, "Atypical Angina": 2, "Non-anginal Pain": 3, "Asymptomatic": 4}
            mapping_ekg = {"Normal": 0, "Abnormalitas ST-T": 1, "Hipertrofi Ventrikel Kiri": 2}
            mapping_slope = {"Upsloping": 1, "Flat": 2, "Downsloping": 3}
            mapping_thallium = {"Normal": 3, "Fixed Defect": 6, "Reversible Defect": 7}
            
            # Konversi string dari UI ke angka yang dibutuhkan model
            Sex = 0 if Sex_str == "Wanita" else 1
            FBS = 0 if FBS_str == "Tidak" else 1
            exang = 0 if exang_str == "Tidak" else 1

            return predict_heart_disease(
                Age, Sex, mapping_cp[cp_str], BP, Chol, FBS, mapping_ekg[ekg_str], Max_HR, 
                exang, ST_dep, mapping_slope[slope_str], vessel, mapping_thallium[thallium_str])
        
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
            cache_examples=True
        )
    return demo

# --- 4. BLOK EKSEKUSI UTAMA ---
if __name__ == "__main__":
    print("Menjalankan aplikasi Gradio...")
    demo = create_gradio_interface()
    demo.launch(server_name="0.0.0.0", server_port=7860)