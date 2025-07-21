# 225150207111001_1 MUHAMMAD NADHIF_1
# 225150201111002_2 NALENDRA MARCHELO_2
# 225150200111005_3 NARENDRA ATHA ABHINAYA_3
# 225150200111003_4 YOSUA SAMUEL EDLYN SINAGA_4

import pandas as pd
import os

data_dir = 'data'
old_data_path = os.path.join(data_dir, 'old_data.csv')
synthetic_data_path = os.path.join(data_dir, 'synthetic_data.csv')
combined_data_path = os.path.join(data_dir, 'combined_data.csv')

try:
    old_data = pd.read_csv(old_data_path)
    synthetic_data = pd.read_csv(synthetic_data_path)
    print("Berhasil memuat data lama dan data sintetis.")

    # Gabungkan kedua dataset
    combined_data = pd.concat([old_data, synthetic_data], ignore_index=True)
    print(f"Data berhasil digabungkan. Total baris data gabungan: {len(combined_data)}")

    # Simpan dataset gabungan ke file CSV baru
    combined_data.to_csv(combined_data_path, index=False)
    print(f"Data gabungan berhasil disimpan di: {combined_data_path}")

except FileNotFoundError as e:
    print(f"Error: File tidak ditemukan. Pastikan path file sudah benar. Detail: {e}")
except Exception as e:
    print(f"Terjadi error: {e}")