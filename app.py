# 225150207111001_1 MUHAMMAD NADHIF_1
# 225150201111002_2 NALENDRA MARCHELO_2
# 225150200111005_3 NARENDRA ATHA ABHINAYA_3
 # 225150200111003_4 YOSUA SAMUEL EDLYN SINAGA_4

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import joblib
import gradio as gr
import os
import sys

# --- Fungsi pelatihan model ---
def train_model():
    print("Memulai proses training model...")
    data_path = 'data/Heart_Disease_Prediction.csv'
    model_dir = 'model'
    model_path = os.path.join(model_dir, 'random_forest_heart_disease.joblib')

    if not os.path.exists(model_dir):
        os.makedirs(model_dir)

    try:
        data = pd.read_csv(data_path)
    except FileNotFoundError:
        print(f"Error: File dataset tidak ditemukan di {data_path}")
        return

    data['Heart Disease'] = data['Heart Disease'].map({'Presence': 1, 'Absence': 0})
    X = data.drop('Heart Disease', axis=1)
    y = data['Heart Disease']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Akurasi model pada data uji: {accuracy:.4f}")

    joblib.dump(model, model_path)
    print(f"Model berhasil dilatih dan disimpan di {model_path}")

# --- Fungsi prediksi ---
def predict_heart_disease(Age, Sex, Chest_pain_type, BP, Cholesterol, FBS_over_120, EKG_results, Max_HR, Exercise_angina, ST_depression, Slope_of_ST, Number_of_vessels_fluro, Thallium):
    model_path = 'model/random_forest_heart_disease.joblib'
    try:
        model = joblib.load(model_path)
    except FileNotFoundError:
        return "Error: File model tidak ditemukan."

    input_data = pd.DataFrame({
        'Age': [Age], 'Sex': [Sex], 'Chest pain type': [Chest_pain_type], 'BP': [BP], 'Cholesterol': [Cholesterol],
        'FBS over 120': [FBS_over_120], 'EKG results': [EKG_results], 'Max HR': [Max_HR], 'Exercise angina': [Exercise_angina],
        'ST depression': [ST_depression], 'Slope of ST': [Slope_of_ST], 'Number of vessels fluro': [Number_of_vessels_fluro],
        'Thallium': [Thallium]
    })

    prediction = model.predict(input_data)[0]
    return "Berisiko Tinggi (Presence)" if prediction == 1 else "Berisiko Rendah (Absence)"

# --- Contoh input ---
examples_list = [
    [35, "Wanita", "Typical Angina", 120, 190, "Tidak", "Normal", 170, "Tidak", 0.5, "Upsloping", 0, "Normal"],
    [42, "Pria", "Non-anginal Pain", 130, 210, "Tidak", "Normal", 165, "Tidak", 0.0, "Upsloping", 0, "Normal"],
    [65, "Pria", "Asymptomatic", 155, 280, "Ya", "Hipertrofi Ventrikel Kiri", 120, "Ya", 2.5, "Flat", 2, "Reversible Defect"],
    [58, "Wanita", "Atypical Angina", 160, 320, "Tidak", "Abnormalitas ST-T", 115, "Ya", 3.0, "Downsloping", 3, "Fixed Defect"]
]

# --- Antarmuka Gradio ---
def create_gradio_interface():
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

        # Mapping string ke angka sesuai model
        def wrapped_predict(*args):
            mapping_cp = {"Typical Angina":1, "Atypical Angina":2, "Non-anginal Pain":3, "Asymptomatic":4}
            mapping_ekg = {"Normal":0, "Abnormalitas ST-T":1, "Hipertrofi Ventrikel Kiri":2}
            mapping_slope = {"Upsloping":1, "Flat":2, "Downsloping":3}
            mapping_thallium = {"Normal":3, "Fixed Defect":6, "Reversible Defect":7}

            Age, Sex, cp, BP, Chol, FBS, ekg, Max_HR, exang, ST_dep, slope, vessel, thallium = args

            return predict_heart_disease(
                Age,
                Sex,
                mapping_cp[cp],
                BP,
                Chol,
                FBS,
                mapping_ekg[ekg],
                Max_HR,
                exang,
                ST_dep,
                mapping_slope[slope],
                vessel,
                mapping_thallium[thallium]
            )

        inputs_list = [
            age_input, sex_input, cp_input, bp_input, chol_input,
            fbs_input, ekg_input, max_hr_input, exang_input, st_depression_input,
            slope_input, vessels_input, thallium_input
        ]

        predict_btn.click(fn=wrapped_predict, inputs=inputs_list, outputs=output_label)

        gr.Examples(
            examples=examples_list,
            inputs=inputs_list,
            outputs=output_label,
            fn=wrapped_predict,
            cache_examples=True
        )

    return demo

# --- Main execution ---
if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == '--train':
        train_model()
    else:
        if not os.path.exists('model/random_forest_heart_disease.joblib'):
            print("Model belum ada. Melakukan training model terlebih dahulu...")
            train_model()
        print("Menjalankan aplikasi Gradio...")
        demo = create_gradio_interface()
        demo.launch(server_name="0.0.0.0", server_port=7860)
