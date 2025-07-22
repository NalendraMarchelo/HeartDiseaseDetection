# 225150207111001_1 MUHAMMAD NADHIF_1
# 225150201111002_2 NALENDRA MARCHELO_2
# 225150200111005_3 NARENDRA ATHA ABHINAYA_3
# 225150200111003_4 YOSUA SAMUEL EDLYN SINAGA_4

# promote_model.py
import mlflow
import os
from mlflow.tracking import MlflowClient

def promote_latest_model():
    """
    Mencari versi model terbaru di stage 'None' dan mempromosikannya 
    ke 'Production', serta mengarsipkan versi 'Production' yang lama.
    """
    # --- Konfigurasi Koneksi ke DagsHub ---
    DAGSHUB_USER = "NalendraMarchelo"
    DAGSHUB_REPO = "HeartDiseaseDetection"
    
    # Mengambil token dari environment variable yang diatur oleh GitHub Actions
    DAGSHUB_TOKEN = os.getenv("DAGSHUB_TOKEN")

    if not DAGSHUB_TOKEN:
        print("Error: DAGSHUB_TOKEN tidak ditemukan. Pastikan sudah diatur di GitHub Secrets.")
        exit(1)
        
    os.environ['MLFLOW_TRACKING_USERNAME'] = DAGSHUB_USER
    os.environ['MLFLOW_TRACKING_PASSWORD'] = DAGSHUB_TOKEN
    mlflow.set_tracking_uri(f"https://dagshub.com/{DAGSHUB_USER}/{DAGSHUB_REPO}.mlflow")

    client = MlflowClient()
    model_name = "HeartDiseaseClassifier"

    try:
        # 1. Dapatkan versi model terbaru yang ada di stage 'None'
        latest_versions = client.get_latest_versions(name=model_name, stages=["None"])
        if not latest_versions:
            print("Tidak ada model baru di stage 'None' untuk dipromosikan.")
            return

        new_model_version = latest_versions[0].version
        print(f"Mempromosikan Version {new_model_version} dari model '{model_name}' ke Production...")

        # 2. Pindahkan versi baru ke Production dan arsipkan versi lama
        client.transition_model_version_stage(
            name=model_name,
            version=new_model_version,
            stage="Production",
            archive_existing_versions=True
        )
        print("✅ Model berhasil dipromosikan ke Production.")
        print(f"Versi lama dari '{model_name}' di stage Production telah diarsipkan.")

    except Exception as e:
        print(f"Gagal mempromosikan model: {e}")
        exit(1)

if __name__ == "__main__":
    promote_latest_model()