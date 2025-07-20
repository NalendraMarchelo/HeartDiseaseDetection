# train.py (Versi DagsHub)
import pandas as pd
import argparse
import mlflow
import mlflow.sklearn
import joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from mlflow.models import infer_signature
import os

# --- 1. FUNGSI-FUNGSI (load_data, preprocess_data) ---
def load_data(file_path):
    print(f"Memuat data dari: {file_path}")
    df = pd.read_csv(file_path)
    if 'Heart Disease' in df.columns and df['Heart Disease'].dtype == 'object':
        df['Heart Disease'] = df['Heart Disease'].map({'Presence': 1, 'Absence': 0})
    return df

def preprocess_data(df):
    TARGET_COLUMN = 'Heart Disease'
    df.dropna(subset=[TARGET_COLUMN], inplace=True)
    X = df.drop(TARGET_COLUMN, axis=1)
    y = df[TARGET_COLUMN]
    imputer = SimpleImputer(strategy='median')
    scaler = StandardScaler()
    X_imputed = imputer.fit_transform(X)
    X_scaled = scaler.fit_transform(X_imputed)
    X_preprocessed = pd.DataFrame(X_scaled, columns=X.columns)
    X_train, X_test, y_train, y_test = train_test_split(
        X_preprocessed, y, test_size=0.2, random_state=42, stratify=y)
    return X_train, X_test, y_train, y_test, scaler, imputer


# --- 2. KONFIGURASI DAN FUNGSI TRAINING ---
def setup_mlflow_tracking():
    """Mengatur koneksi ke MLflow Tracking Server (DagsHub atau lokal)."""
    
    # --- GANTI DENGAN INFORMASI ANDA ---
    DAGSHUB_USER = "NalendraMarchelo"  # Ganti dengan username DagsHub Anda
    DAGSHUB_REPO = "HeartDiseaseDetection" # Ganti dengan nama repository DagsHub Anda
    
    # Prioritaskan koneksi ke DagsHub jika kredensial ada
    if os.getenv("DAGSHUB_TOKEN"):
        print("Menggunakan DagsHub MLflow Tracking Server...")
        os.environ['MLFLOW_TRACKING_USERNAME'] = DAGSHUB_USER
        os.environ['MLFLOW_TRACKING_PASSWORD'] = os.getenv("DAGSHUB_TOKEN")
        mlflow.set_tracking_uri(f"https://dagshub.com/{DAGSHUB_USER}/{DAGSHUB_REPO}.mlflow")
    elif os.getenv("MLFLOW_TRACKING_URI"):
        # Digunakan saat berjalan di dalam Docker
        print(f"Menggunakan MLflow Tracking Server dari environment: {os.getenv('MLFLOW_TRACKING_URI')}")
        mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI"))
    else:
        # Fallback ke server lokal jika tidak ada konfigurasi lain
        print("Menggunakan MLflow Tracking Server lokal...")
        mlflow.set_tracking_uri("http://localhost:5000")

def train_and_log_model(X_train, y_train, X_test, y_test, scaler, imputer, experiment_name, run_name):
    """Melatih model dan mencatat semuanya ke MLflow."""
    setup_mlflow_tracking()
    mlflow.set_experiment(experiment_name)
    
    with mlflow.start_run(run_name=run_name) as run:
        print(f"Memulai run: {run_name}")
        model = RandomForestClassifier(n_estimators=100, random_state=42)
        model.fit(X_train, y_train)
        
        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        mlflow.log_metric("accuracy", accuracy)
        print(f"Akurasi: {accuracy}")

        signature = infer_signature(X_train, model.predict(X_train))
        mlflow.sklearn.log_model(sk_model=model, artifact_path="model", signature=signature)
        
        run_id = run.info.run_id
        model_uri = f"runs:/{run_id}/model"
        mlflow.register_model(model_uri=model_uri, name="HeartDiseaseClassifier")
        print("Model berhasil diregistrasi.")
        
        joblib.dump(scaler, "scaler.joblib")
        joblib.dump(imputer, "imputer.joblib")
        mlflow.log_artifact("scaler.joblib")
        mlflow.log_artifact("imputer.joblib")
        print("Scaler dan Imputer berhasil dicatat.")

# --- 3. BLOK EKSEKUSI UTAMA ---
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, required=True)
    parser.add_argument("--experiment_name", type=str, default="Prediksi Penyakit Jantung")
    parser.add_argument("--run_name", type=str, default="TrainingRun")
    args = parser.parse_args()
    
    raw_df = load_data(args.dataset)
    X_train, X_test, y_train, y_test, scaler, imputer = preprocess_data(raw_df)
    train_and_log_model(X_train, y_train, X_test, y_test, scaler, imputer, args.experiment_name, args.run_name)