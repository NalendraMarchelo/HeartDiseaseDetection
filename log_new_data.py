# log_new_data.py
import pandas as pd
import os

def log_prediction_data(new_data_dict):
    """
    Menerima data baru dalam bentuk dictionary, mengubahnya menjadi DataFrame,
    dan menyimpannya ke file CSV.
    """
    log_file = 'data/new_prediction_logs.csv'
    
    # Ubah dictionary menjadi DataFrame
    new_df = pd.DataFrame([new_data_dict])

    # Jika file log sudah ada, tambahkan data baru. Jika tidak, buat file baru.
    if os.path.exists(log_file):
        new_df.to_csv(log_file, mode='a', header=False, index=False)
    else:
        new_df.to_csv(log_file, index=False)
        
    print(f"Data baru berhasil dicatat di {log_file}")