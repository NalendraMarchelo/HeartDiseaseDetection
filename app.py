# app.py (Versi Baru untuk Serving)

import pandas as pd
import gradio as gr
import mlflow.pyfunc
import joblib
import os
import sys

# --- 1. MUAT MODEL DAN PREPROCESSOR ---
# Pastikan file scaler.joblib dan imputer.joblib ada di direktori yang sama
SCALER_PATH = "scaler.joblib"
IMPUTER_PATH = "imputer.joblib"
MODEL_NAME = "HeartDiseaseClassifier"
MODEL_STAGE = "Production"
model_uri = f"models:/{MODEL_NAME}/{MODEL_STAGE}"

try:
    print(f"Memuat model dari: {model_uri}")
    model = mlflow.pyfunc.load_model(model_uri)
    
    print(f"Memuat scaler dari: {SCALER_PATH}")
    scaler = joblib.load(SCALER_PATH)
    
    print(f"Memuat imputer dari: {IMPUTER_PATH}")
    imputer = joblib.load(IMPUTER_PATH)
    
    print("Model dan preprocessor berhasil dimuat.")

except Exception as e:
    print(f"Error saat memuat model atau preprocessor: {e}")
    sys.exit(1)


# --- 2. FUNGSI PREDIKSI BARU ---
def predict_heart_disease(Age, Sex, Chest_pain_type, BP, Cholesterol, FBS_over_120, EKG_results, Max_HR, Exercise_angina, ST_depression, Slope_of_ST, Number_of_vessels_fluro, Thallium):
    
    feature_names = [
        'Age', 'Sex', 'Chest pain type', 'BP', 'Cholesterol', 'FBS over 120', 
        'EKG results', 'Max HR', 'Exercise angina', 'ST depression', 'Slope of ST', 
        'Number of vessels fluro', 'Thallium'
    ]
    
    input_data = pd.DataFrame({
        'Age': [Age], 'Sex': [Sex], 'Chest pain type': [Chest_pain_type], 'BP': [BP], 'Cholesterol': [Cholesterol],
        'FBS over 120': [FBS_over_120], 'EKG results': [EKG_results], 'Max HR': [Max_HR], 'Exercise angina': [Exercise_angina],
        'ST depression': [ST_depression], 'Slope of ST': [Slope_of_ST], 'Number of vessels fluro': [Number_of_vessels_fluro],
        'Thallium': [Thallium]
    }, columns=feature_names)
    
    input_imputed = imputer.transform(input_data)
    input_scaled = scaler.transform(input_imputed)
    
    input_processed = pd.DataFrame(input_scaled, columns=feature_names)
    
    prediction = model.predict(input_processed)[0]
    
    return "Berisiko Tinggi (Presence)" if prediction == 1 else "Berisiko Rendah (Absence)"


# --- 3. ANTARMUKA GRADIo ---
def create_gradio_interface():
    with gr.Blocks(theme=gr.themes.Default(), title="Prediksi Penyakit Jantung") as demo:
        gr.Markdown("# 🩺 Aplikasi Prediksi Penyakit Jantung")
        gr.Markdown("Masukkan data medis pasien untuk memprediksi risiko penyakit jantung.")

        with gr.Row():
            with gr.Column():
                age_input = gr.Slider(label="Usia", minimum=29, maximum=77, step=1, value=54)
                sex_input = gr.Radio(label="Jenis Kelamin", choices=["Wanita", "Pria"], value="Pria", type="index")
                cp_input = gr.Dropdown(label="Jenis Nyeri Dada", choices=["Typical Angina", "Atypical Angina", "Non-anginal Pain", "Asymptomatic"], value="Non-anginal Pain")
                bp_input = gr.Slider(label="Tekanan Darah", minimum=94, maximum=200, step=1, value=131)
                chol_input = gr.Slider(label="Kolesterol", minimum=126, maximum=564, step=1, value=249)
                max_hr_input = gr.Slider(label="Detak Jantung Maks", minimum=71, maximum=202, step=1, value=149)
            
            with gr.Column():
                fbs_input = gr.Radio(label="Gula Darah > 120", choices=["Tidak", "Ya"], value="Tidak", type="index")
                ekg_input = gr.Dropdown(label="Hasil EKG", choices=["Normal", "Abnormalitas ST-T", "Hipertrofi Ventrikel Kiri"], value="Abnormalitas ST-T")
                exang_input = gr.Radio(label="Angina Saat Olahraga", choices=["Tidak", "Ya"], value="Tidak", type="index")
                st_depression_input = gr.Slider(label="ST Depression", minimum=0.0, maximum=6.2, step=0.1, value=1.0)
                slope_input = gr.Dropdown(label="Slope ST", choices=["Upsloping", "Flat", "Downsloping"], value="Flat")
                vessels_input = gr.Dropdown(label="Jumlah Pembuluh Terlihat", choices=[0, 1, 2, 3], value=0)
                thallium_input = gr.Dropdown(label="Thallium", choices=["Normal", "Fixed Defect", "Reversible Defect"], value="Normal")
                
                predict_btn = gr.Button("🔮 Lakukan Prediksi", variant="primary")
                output_label = gr.Label(label="Status Risiko")

        def wrapped_predict(*args):
            mapping_cp = {"Typical Angina": 1, "Atypical Angina": 2, "Non-anginal Pain": 3, "Asymptomatic": 4}
            mapping_ekg = {"Normal": 0, "Abnormalitas ST-T": 1, "Hipertrofi Ventrikel Kiri": 2}
            mapping_slope = {"Upsloping": 1, "Flat": 2, "Downsloping": 3}
            mapping_thallium = {"Normal": 3, "Fixed Defect": 6, "Reversible Defect": 7}
            
            # Unpack all 13 arguments
            (Age, Sex, cp_str, BP, Chol, FBS, ekg_str, Max_HR, exang, 
             ST_dep, slope_str, vessel, thallium_str) = args
            
            # Map string values to numbers
            cp_num = mapping_cp[cp_str]
            ekg_num = mapping_ekg[ekg_str]
            slope_num = mapping_slope[slope_str]
            thallium_num = mapping_thallium[thallium_str]
            
            # Call the main prediction function
            return predict_heart_disease(
                Age, Sex, cp_num, BP, Chol, FBS, ekg_num, Max_HR, 
                exang, ST_dep, slope_num, vessel, thallium_num
            )
        
        inputs_list = [
            age_input, sex_input, cp_input, bp_input, chol_input, fbs_input, 
            ekg_input, max_hr_input, exang_input, st_depression_input, slope_input, 
            vessels_input, thallium_input
        ]
        
        predict_btn.click(fn=wrapped_predict, inputs=inputs_list, outputs=output_label)

    return demo

# --- 4. BLOK EKSEKUSI UTAMA ---
if __name__ == "__main__":
    print("Menjalankan aplikasi Gradio...")
    app_interface = create_gradio_interface()
    app_interface.launch(server_name="0.0.0.0", server_port=7860)