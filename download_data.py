# download_data.py
import requests
import os

os.makedirs('data', exist_ok=True)

# Ganti URL di bawah ini dengan URL raw dari DagsHub Anda
urls = {
    "old_data.csv": "https://dagshub.com/NalendraMarchelo/HeartDiseaseDetection/raw/main/data/old_data.csv",
    "synthetic_data.csv": "https://dagshub.com/NalendraMarchelo/HeartDiseaseDetection/raw/main/data/synthetic_data.csv"
}

for filename, url in urls.items():
    try:
        print(f"Mengunduh {filename} dari {url}...")
        response = requests.get(url)
        response.raise_for_status()

        with open(os.path.join('data', filename), 'wb') as f:
            f.write(response.content)
        print(f"Berhasil menyimpan {filename}")

    except requests.exceptions.RequestException as e:
        print(f"Gagal mengunduh {filename}. Error: {e}")
        exit(1)