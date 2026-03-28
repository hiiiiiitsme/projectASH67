# ☀️ Solar Flare AI Prediction System

> **Güneş patlamalarını 7 gün önceden tahmin ederek uydu, GPS ve elektrik şebekelerini koruyoruz.**

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.10+-blue.svg" alt="Python">
  <img src="https://img.shields.io/badge/PyTorch-2.0-red.svg" alt="PyTorch">
  <img src="https://img.shields.io/badge/Streamlit-1.x-ff4b4b.svg" alt="Streamlit">
  <img src="https://img.shields.io/badge/License-MIT-green.svg" alt="License">
  <img src="https://img.shields.io/badge/Hackathon-2025-orange.svg" alt="Hackathon">
</p>

---

## 🎯 Problem

1859 Carrington Olayı'nda dev bir güneş fırtınası dünya çapında telgraf sistemlerini çökertdi. Bugün aynı büyüklükte bir olay olsaydı:

| Etki | Büyüklük |
|------|----------|
| 🛰️ Hasar görebilecek uydu | **2,000+** |
| 📡 GPS kesintisi | **Saatlerce** |
| ⚡ Elektrik şebekesi hasarı | **Aylarca tamir** |
| 💰 Ekonomik kayıp | **~2.6 trilyon USD** |

**Mevcut uyarı süresi:** 15–60 dakika
**Bizim hedefimiz:** **7 gün önceden tahmin**

---

## 💡 Çözümümüz

Yapay zeka tabanlı **3 katmanlı hibrit sistem**:

### 1. LSTM Derin Öğrenme Modeli
- Son 7 günün güneş aktivitesini 7-adımlı zaman serisi olarak işler
- EMA (üstel hareketli ortalama) özellikleriyle zenginleştirilmiş girdi
- Binary classification: "7 gün sonra X-class flare olur mu?"

### 2. 27-Günlük Rotasyon Tahmini ⭐ (İNOVASYON)
- Güneşin kendi etrafında dönüş periyodunu kullanır
- **Arka yüz tahmini**: Şu an görünmeyen güneş yüzeyindeki aktiviteyi tahmin eder
- Diferansiyel rotasyon formülü: `ω(λ) = 14.713 − 2.396·sin²(λ) − 1.787·sin⁴(λ)` derece/gün

### 3. Ensemble Yaklaşım
```
Final Tahmin = 0.55 × RF + 0.45 × LSTM   (threshold optimize edildi)
```

---

## 📊 Model Performansı

| Model | Accuracy | Precision | Recall | F1-Score | AUC |
|-------|----------|-----------|--------|----------|-----|
| Persistence Baseline | 0.812 | — | — | 0.000 | — |
| Random Forest | 0.653 | 0.242 | 0.333 | 0.281 | 0.604 |
| **LSTM** | **0.734** | **0.433** | **0.875** | **0.579** | **0.811** |
| **Ensemble (RF + LSTM)** | **0.677** | **0.392** | **0.979** | **0.560** | **0.816** |

> Test seti: 2025-08-03 → 2026-03-26 (236 gün, 20.3% pozitif oran)

### Ensemble Confusion Matrix
```
                  Tahmin: Yok   Tahmin: Var
Gerçek: Yok  [     108           73    ]
Gerçek: Var  [       1           47    ]

True Negatives:  108  ✅ Doğru "patlama yok" tahmini
True Positives:   47  ✅ Doğru "patlama var" tahmini
False Positives:  73  ⚠️ Yanlış alarm
False Negatives:   1  🚨 Kaçan patlama (kritik hata — minimize edildi)
```

### 🔑 En Önemli Özellikler
1. `cycle_phase` — Solar Döngü 25 fazı
2. `ema7_n_m` — 7-günlük EMA M-sınıfı sayısı
3. `day_of_year` — Mevsimsel örüntü
4. `ema14_intensity` — 14-günlük EMA X-ray yoğunluğu
5. `roll14_intensity` — 14-günlük kayan ortalama yoğunluk

---

## 🌟 İnovasyon: Arka Yüz Tahmini

Güneşin Dünya'ya dönük olmayan yüzü **arka yüz** olarak adlandırılır ve doğrudan gözlemlenemez.

**Bizim Yaklaşımımız:**
```python
# 1. 27-Günlük Rotasyon Döngüsü
backside_estimate = historical_data.shift(27)

# 2. Diferansiyel Rotasyon Formülü
ω(λ) = 14.713 - 2.396·sin²(λ) - 1.787·sin⁴(λ)  # derece/gün
# Ekvator ~25.4 gün, kutuplar ~38 günde döner

# 3. Risk Tahmini
risk_score = lstm_predict(backside_estimate)
```

**Sonuç:** Tahmin süresini 2–3 günden **10–14 güne** çıkarıyoruz 🎯

---

## 🏗️ Proje Yapısı

```
solar-flare-prediction/
├── data/
│   ├── raw/
│   │   ├── nasa_solar_flares.csv       # 2,212 patlama olayı (2023-2026)
│   │   ├── nasa_cme.csv                # 4,823 CME olayı
│   │   └── nasa_geomagnetic_storms.csv # 66 jeomanyetik fırtına
│   └── processed/
│       └── daily_features.csv          # 1,178 günlük özellik seti
├── models/
│   ├── lstm_model.pth                  # LSTM ağırlıkları (AUC=0.811)
│   └── random_forest.pkl               # RF modeli + scaler + threshold
├── scripts/
│   ├── data_preprocessing.py           # NASA CSV yükleme, konum parse
│   ├── feature_engineering.py          # 34 özellik: rolling, EMA, lag-27
│   └── train_model.py                  # RF + LSTM + Ensemble eğitimi
├── results/
│   └── metrics.json                    # Test metrikleri
├── demo.py                             # Streamlit demo (5 tab)
├── requirements.txt
├── .env.example
└── README.md
```

---

## 🚀 Hızlı Başlangıç

### Gereksinimler
```
Python 3.10+
GPU (opsiyonel, CPU'da da çalışır)
```

### Kurulum
```bash
# 1. Repo'yu klonla
git clone https://github.com/your-team/solar-flare-prediction.git
cd solar-flare-prediction

# 2. Virtual environment oluştur
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# 3. Bağımlılıkları kur
pip install -r requirements.txt

# 4. NASA API key al (ücretsiz)
# https://api.nasa.gov/ adresinden al

# 5. .env dosyası oluştur
cp .env.example .env
# .env içine: NASA_API_KEY=your_key_here
```

### Veri İndirme
```bash
python scripts/nasa_download.py
# Çıktı: data/raw/nasa_solar_flares.csv, nasa_cme.csv, nasa_geomagnetic_storms.csv
```

### Model Eğitimi
```bash
python scripts/train_model.py
# Çıktı: models/lstm_model.pth, models/random_forest.pkl, results/metrics.json
```

### Demo Çalıştırma
```bash
streamlit run demo.py
# Tarayıcıda aç: http://localhost:8501
```

---

## 📺 Demo Özellikleri (5 Tab)

### 🌟 Dashboard
- Gerçek zamanlı risk seviyesi göstergesi (gauge)
- 7 günlük X-class patlama tahmin barları
- Son patlamalar tablosu
- Son 90 gün aktivite stack chart

### 🔮 AI Tahmin
- Tarih seçici ile özelleştirilebilir tahmin ufku (1–14 gün)
- RF / Rotasyon / Ensemble karşılaştırma grafiği
- Günlük risk tablosu + uyarı bayrakları
- Top 15 özellik önemi

### 🌑 Arka Yüz Analizi ⭐
- Solar disk görselleştirmesi (görünen yüz vs arka yüz)
- 27-gün rotasyon döngüsü analizi
- Diferansiyel rotasyon formülü grafiği

### 📈 Tarihsel Analiz
- 2023–2026 aktivite zaman serisi
- Sınıf dağılımı (C / M / X) pasta grafiği
- Aylık aktivite heatmap
- Confusion matrix + ROC eğrisi

### ⚡ Etki Senaryoları
- Patlama sınıfına göre etki hesaplayıcı (C5 → X45)
- Radar chart: GPS / Uydu / Şebeke / Radyo / Aurora
- Ekonomik etki karşılaştırması
- Uydu risk tablosu

---

## 🛠️ Teknoloji Stack

| Katman | Teknoloji |
|--------|-----------|
| **Derin Öğrenme** | PyTorch 2.0 — LSTM (2 katman, hidden=64) |
| **Makine Öğrenimi** | scikit-learn — Random Forest (500 ağaç) |
| **Veri İşleme** | pandas, numpy |
| **Görselleştirme** | Streamlit, Plotly |
| **Veri Kaynağı** | NASA DONKI API, NOAA SWPC |

---

## 📊 Veri Kaynakları

| Kaynak | Veri Tipi | Kayıt | Dönem |
|--------|-----------|-------|-------|
| **NASA DONKI API** | Solar flare olayları | 2,212 | 2023–2026 |
| **NASA DONKI API** | CME olayları | 4,823 | 2023–2026 |
| **NASA DONKI API** | Jeomanyetik fırtınalar | 66 | 2023–2026 |

**Toplam:** ~7,100 olay — Solar Cycle 25 (zirve dönemi dahil)

---

## 🔬 Feature Engineering

Toplam **34 özellik**, 3 kategoride:

```
Temel Sayımlar (13):    n_c, n_m, n_x, n_b, max_intensity, sum_intensity,
                        mean_intensity, n_active_regions, mean_lat,
                        n_cme, max_cme_speed, mean_cme_speed, max_kp

Rolling Windows (12):   roll3/7/14 × n_x, n_m, intensity, n_cme

EMA + Trend (4):        ema5_intensity, ema14_intensity, ema7_n_m, ema_trend

27-Gün Lag (3):         lag27_n_x, lag27_n_m, lag27_intensity

Solar Döngü (2):        cycle_phase, day_of_year
```

---

## 🚀 Gelecek Planlar

**Kısa Vadeli (3–6 ay)**
- [ ] Real-time data pipeline (canlı tahmin, her 6 saatte güncelleme)
- [ ] Multi-class classification (C / M / X ayrı tahmin)
- [ ] ESA Helioseismic veri entegrasyonu
- [ ] Mobile app (iOS / Android)

**Orta Vadeli (6–12 ay)**
- [ ] SDO FITS görüntüleri ile CNN modeli
- [ ] Attention mechanism (Transformer) ekleme
- [ ] Physics-informed neural networks (PINN)
- [ ] TUA (Türkiye Uzay Ajansı) verisi entegrasyonu 🇹🇷

**Uzun Vadeli (1–2 yıl)**
- [ ] ESA Vigil 2029 entegrasyonu (arka yüz görüntüleme)
- [ ] Multi-view 3D reconstruction
- [ ] Operational deployment (NOAA / ESA ile işbirliği)

---

## 📚 Referanslar

1. Bobra, M. G., & Couvidat, S. (2015). *Solar Flare Prediction Using SDO/HMI Vector Magnetic Field Data with a Machine-Learning Algorithm.* The Astrophysical Journal.
2. Nishizuka, N., et al. (2017). *Solar Flare Prediction Model with Three Machine-Learning Algorithms using Ultraviolet Brightening and Vector Magnetograms.* The Astrophysical Journal.

**Veri Kaynakları**
- NASA DONKI: https://ccmc.gsfc.nasa.gov/tools/DONKI/
- NOAA SWPC: https://www.swpc.noaa.gov/
- Space Weather Prediction Center: https://www.swpc.noaa.gov/

---

## 🎓 Ekip — Team Solar Sentinels

| Rol | Sorumluluk |
|-----|-----------|
| Data Engineer | NASA/NOAA veri toplama, pipeline |
| Data Scientist | Feature engineering, EMA analizi |
| ML Engineer 1 | LSTM model geliştirme, threshold opt. |
| ML Engineer 2 | Random Forest, ensemble |
| Full-Stack Dev | Streamlit demo, deployment |

---

## 📄 Lisans

MIT License — Detaylar için `LICENSE` dosyasına bakın.

---

## 🙏 Teşekkürler

- **NASA** — DONKI API'si için
- **NOAA** — Gerçek zamanlı veri paylaşımı için
- **Hackathon Organizatörleri** — Bu fırsatı sunduğu için

---

<p align="center">
  <b>Made with ☀️ by Team Solar Sentinels</b><br>
  <a href="https://github.com/your-team/solar-flare-prediction">GitHub</a> ·
  <a href="https://solar-flare-demo.streamlit.app">Live Demo</a>
</p>
