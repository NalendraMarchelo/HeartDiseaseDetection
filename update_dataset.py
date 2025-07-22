# 225150207111001_1 MUHAMMAD NADHIF_1
# 225150201111002_2 NALENDRA MARCHELO_2
# 225150200111005_3 NARENDRA ATHA ABHINAYA_3
# 225150200111003_4 YOSUA SAMUEL EDLYN SINAGA_4

# update_dataset.py
import pandas as pd
import os

data_dir = 'data'
combined_file = os.path.join(data_dir, 'combined_data.csv')
logs_file = os.path.join(data_dir, 'new_logs.csv')
old_file = os.path.join(data_dir, 'old_data.csv') # File awal

# Cek apakah file log baru ada
if not os.path.exists(logs_file):
    print("Tidak ada log data baru untuk ditambahkan.")
    exit()

# Gunakan combined_data.csv jika ada, jika tidak mulai dari old_data.csv
if os.path.exists(combined_file):
    base_data = pd.read_csv(combined_file)
else:
    base_data = pd.read_csv(old_file)

new_logs = pd.read_csv(logs_file)

# Gabungkan
updated_data = pd.concat([base_data, new_logs], ignore_index=True)

# Timpa file combined_data.csv dengan versi baru
updated_data.to_csv(combined_file, index=False)

# Hapus file log lama setelah digabungkan
os.remove(logs_file)

print(f"Dataset berhasil diperbarui. Total baris sekarang: {len(updated_data)}")