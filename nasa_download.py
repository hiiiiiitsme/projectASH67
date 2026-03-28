import os
import requests
import pandas as pd
from dotenv import load_dotenv

# .env dosyasındaki NASA_API_KEY'i yükle
load_dotenv()
API_KEY = os.getenv("NASA_API_KEY", "DEMO_KEY")  # Key yoksa demo_key kullanır


def fetch_nasa_data(endpoint, start_date="2023-01-01", end_date="2026-03-27"):
    base_url = f"https://api.nasa.gov/DONKI/{endpoint}"
    params = {
        "startDate": start_date,
        "endDate": end_date,
        "api_key": API_KEY
    }

    print(f"📡 {endpoint} verileri çekiliyor...")
    response = requests.get(base_url, params=params)

    if response.status_code == 200:
        return response.json()
    else:
        print(f"❌ Hata: {response.status_code}")
        return []


def main():
    # main() fonksiyonunun içine Flare ve CME bloklarının altına ekle:

    # 3. GST (Geomagnetic Storms - Jeomanyetik Fırtınalar)
    storms = fetch_nasa_data("GST")
    if storms:
        df_storms = pd.DataFrame(storms)
        df_storms.to_csv("data/raw/nasa_geomagnetic_storms.csv", index=False)
        print(f"✅ {len(df_storms)} adet jeomanyetik fırtına kaydedildi.")

if __name__ == "__main__":
    main()