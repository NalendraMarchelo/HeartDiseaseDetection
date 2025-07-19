# train.py

import pandas as pd
import numpy as np
import argparse
import mlflow
import mlflow.sklearn
import joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from mlflow.models import infer_signature
import os

# --- 1. FUNGSI UNTUK MEMUAT DATA ---
def load_data(file_path):
    """Memuat data dari path file CSV."""
    print(f"Memuat data dari: {file_path}")
    df = pd.read_csv(file_path)
    # Mapping label target jika ada
    if 'Heart Disease' in df.columns and df['Heart Disease'].dtype == 'object':
        df['Heart Disease'] = df['Heart Disease'].map({'Presence': 1, 'Absence': 0})
    return df

# --- 2. FUNGSI UNTUK PREPROCESSING ---
def preprocess_data(df):
    """Melakukan preprocessing lengkap pada data dan mengembalikan preprocessor."""
    TARGET_COLUMN = 'Heart Disease'
    
    df.dropna(subset=[TARGET_COLUMN], inplace=True)
    print(f"1. Baris dengan target kosong dihapus. Baris tersisa: {len(df)}")
    
    X = df.drop(TARGET_COLUMN, axis=1)
    y = df[TARGET_COLUMN]
    
    imputer = SimpleImputer(strategy='median')
    scaler = StandardScaler()
    
    X_imputed = imputer.fit_transform(X)
    X_scaled = scaler.fit_transform(X_imputed)
    
    X_preprocessed = pd.DataFrame(X_scaled, columns=X.columns)
    print("2. Preprocessing (imputasi & standarisasi) selesai.")
    
    X_train, X_test, y_train, y_test = train_test_split(
        X_preprocessed, y, test_size=0.2, random_state=42, stratify=y
    )
    print(f"3. Data siap. Bentuk X_train: {X_train.shape}")
    
    return X_train, X_test, y_train, y_test, scaler, imputer

# --- 3. FUNGSI UNTUK TRAINING DAN LOGGING ---
def train_and_log_model(X_train, y_train, X_test, y_test, scaler, imputer, experiment_name, run_name):
    """Melatih model dan mencatat semuanya dengan MLflow."""
    mlflow.set_tracking_uri("http://127.0.0.1:5000")
    mlflow.set_experiment(experiment_name)
    
    with mlflow.start_run(run_name=run_name) as run:
        print(f"Memulai run: {run_name} di bawah eksperimen: {experiment_name}")
        
        model = RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced')
        mlflow.log_params(model.get_params())
        
        model.fit(X_train, y_train)
        
        y_pred_train = model.predict(X_train)
        signature = infer_signature(X_train, y_pred_train)
        
        y_pred_test = model.predict(X_test)
        
        metrics = {
            "accuracy": accuracy_score(y_test, y_pred_test),
            "recall": recall_score(y_test, y_pred_test),
            "precision": precision_score(y_test, y_pred_test),
            "f1_score": f1_score(y_test, y_pred_test)
        }
        mlflow.log_metrics(metrics)
        print(f"Metrik dievaluasi: {metrics}")
        
        mlflow.sklearn.log_model(
            sk_model=model,
            artifact_path="model",
            signature=signature,
            registered_model_name="HeartDiseaseClassifier"
        )
        
        # Simpan dan log preprocessor sebagai artefak
        joblib.dump(scaler, "scaler.joblib")
        joblib.dump(imputer, "imputer.joblib")
        mlflow.log_artifact("scaler.joblib")
        mlflow.log_artifact("imputer.joblib")
        print("Scaler dan Imputer berhasil disimpan dan dicatat.")
        
        print(f"\n✅ Eksperimen '{run_name}' berhasil dicatat.")
        print(f"Run ID: {run.info.run_id}")

# --- 4. BLOK EKSEKUSI UTAMA ---
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Pipeline Pelatihan Model Penyakit Jantung")
    parser.add_argument("--dataset", type=str, required=True, help="Path ke file dataset CSV.")
    parser.add_argument("--experiment_name", type=str, default="Prediksi Penyakit Jantung v2", help="Nama eksperimen MLflow.")
    parser.add_argument("--run_name", type=str, required=True, help="Nama run MLflow.")
    args = parser.parse_args()

    # Menjalankan pipeline
    raw_df = load_data(args.dataset)
    X_train, X_test, y_train, y_test, scaler, imputer = preprocess_data(raw_df)
    train_and_log_model(X_train, y_train, X_test, y_test, scaler, imputer, args.experiment_name, args.run_name)