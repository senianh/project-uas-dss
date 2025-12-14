# 🎯 Marketing Promotion Response DSS

Sistem Pendukung Keputusan (Decision Support System / DSS) berbasis **Data Mining dan Machine Learning** untuk **memprediksi respons promosi pelanggan** serta **menentukan prioritas target promosi** secara objektif dan berbasis data.

Aplikasi ini dikembangkan sebagai **Proyek UAS Decision Support System**, dengan studi kasus pemasaran (marketing analytics) menggunakan dataset *Customer Personality Analysis*.

---

## 👩‍💻 Anggota

| Senia Nur Hasanah | Martha Meslina Florencia |Keyna Fatima Abinalibrata |
|-------------------|-------------------|-------------------|
| 140810230021 | 140810230037 | 140810230067 |

---

## 📌 Fitur Utama

✅ **Prediksi Respons Promosi**
Menggunakan model **XGBoost Classifier** untuk memprediksi probabilitas pelanggan merespons promosi.

✅ **Rekomendasi Produk Dinamis**
Memberikan rekomendasi produk berdasarkan kecocokan profil pelanggan (usia, pendapatan, keluarga) dan pola pembelian historis.

✅ **Analisis Segmentasi Produk**
Menyajikan insight berbasis data aktual seperti:

* Range umur ideal pelanggan
* Level pendapatan dominan
* Pola keluarga
* Statistik pembeli per produk

✅ **Ranking Target Pelanggan (TOPSIS)**
Menggabungkan:

* Probabilitas respons
* Total pengeluaran
* Pendapatan
* Recency

untuk menghasilkan **daftar prioritas pelanggan promosi**.

✅ **Dashboard Interaktif**
Visualisasi insight bisnis menggunakan Plotly dan Streamlit.

---

## 🧠 Metode dan Teknologi

### 1. Metode Analisis

* **Data Mining**
* **Machine Learning (Supervised Classification)**
* **Multi-Criteria Decision Making (TOPSIS)**

### 2. Model Utama

* **XGBoost Classifier**

  * Alasan: akurat, efisien, dan kuat terhadap data tabular

### 3. Feature Engineering

Beberapa fitur turunan yang digunakan:

* `Age`
* `Total_Spent`
* `Total_Purchases`
* `Avg_Order_Value`
* `Child_Category`
* `Spending_vs_Income`
* `Web_vs_Store_Ratio`
* `Customer_Age_Days`

---

## 📂 Dataset

Dataset yang digunakan adalah:

**Customer Personality Analysis Dataset**
Sumber: Kaggle (gratis dan publik)

🔗 [https://www.kaggle.com/datasets/imakash3011/customer-personality-analysis](https://www.kaggle.com/datasets/imakash3011/customer-personality-analysis)

Dataset mencakup:

* Data demografi pelanggan
* Pola pembelian multi-channel
* Total pengeluaran per kategori produk
* Respons pelanggan terhadap promosi (target)

---

## 🖥️ Teknologi yang Digunakan

* **Python 3.9+**
* **Streamlit** (Frontend & UI)
* **Pandas, NumPy** (Data Processing)
* **Scikit-learn** (Pipeline & evaluasi)
* **XGBoost** (Model prediksi)
* **Plotly** (Visualisasi interaktif)

---

## 🚀 User Guide

#### 1. Aplikasi dapat diakses melalui browser menggunakan tautan berikut:

[Marketing Promotion Response DSS:](https://project-uas-dss.streamlit.app/)

##$# 2. Pemrosesan Data Pelanggan
Sistem akan memproses data pelanggan yang tersedia, meliputi:
- Informasi demografi
- Pendapatan
- Aktivitas dan riwayat pembelian
- Recency transaksi

#### 3. Prediksi Respons Pelanggan
Model **XGBoost** akan menghasilkan nilai:
- **Probability_Respond (0–1)**  
  Nilai ini menunjukkan peluang pelanggan merespons promosi.

> Pelanggan dengan **Probability_Respond ≥ 0.40** akan diproses ke tahap berikutnya.

#### 4. Visualisasi Feature Importance
Aplikasi menampilkan grafik feature importance untuk menunjukkan faktor-faktor yang paling berpengaruh terhadap respons pelanggan, seperti:
- Recency
- Total_Spent
- Income
- Total_Purchases
- Aktivitas web pelanggan

#### 5. Perankingan Pelanggan (TOPSIS)
Pelanggan potensial akan diperingkat menggunakan metode **TOPSIS** berdasarkan kriteria berikut:

| Criteria | Type | Weight |
|--------|------|--------|
| Probability_Respond | Benefit | 0.40 |
| Total_Spent | Benefit | 0.30 |
| Income | Benefit | 0.20 |
| Recency | Cost | 0.10 |

Output yang dihasilkan:
- Skor TOPSIS
- Urutan prioritas pelanggan
- Rekomendasi target promosi utama

---

### 📊 Interpretasi Hasil
- **Probability_Respond tinggi** → peluang respons promosi besar
- **Skor TOPSIS tinggi** → pelanggan prioritas utama

Hasil ini dapat digunakan sebagai dasar penentuan strategi promosi yang lebih efektif dan efisien.

---

## 🚀 Cara Menjalankan Aplikasi dari GitHub

### 1. Clone / Siapkan Project

Pastikan file berikut berada dalam satu direktori:

```
- dss_app.py
- model_manager.py
- marketing_data.csv
- styles.css
- README.md
```

### 2. Install Dependency

```bash
pip install streamlit pandas numpy scikit-learn xgboost plotly
```

### 3. Jalankan Aplikasi

```bash
streamlit run dss_app.py
```

Aplikasi akan terbuka otomatis di browser.

---


## 📊 Modul Aplikasi

1️⃣ **Prediksi Respons Promosi**
Input data pelanggan → prediksi probabilitas → rekomendasi promosi

2️⃣ **Dashboard Analisis**
Insight bisnis berbasis segmentasi pelanggan dan feature importance

3️⃣ **Target List Pelanggan**
Ranking prioritas pelanggan menggunakan metode **TOPSIS**

---

## 🎓 Konteks Akademik

Proyek ini dikembangkan untuk memenuhi tugas:

**Mata Kuliah:** Decision Support System
**Program Studi:** Teknik Informatika
**Institusi:** Universitas Padjadjaran

Pendekatan dan implementasi telah disesuaikan dengan:

* Prinsip DSS
* Landasan Data Mining
* Metodologi ilmiah

---

## 📖 Referensi Utama

* Apampa, O. (2016). Evaluation of Classification and Ensemble Algorithms for Bank Customer 
Marketing Response Prediction Evaluation of Classification and Ensemble Algorithms for 
Bank Customer Marketing Response Prediction. Journal of International Technology and 
Information Management, 25(4).  
* Imakash3011. (2018). *Customer Personality Analysis Dataset*. Kaggle.

---

## ✨ Catatan Akhir

Aplikasi ini bersifat **data-driven, adaptif, dan reusable**, serta dapat dikembangkan lebih lanjut dengan:

* Integrasi data real-time
* Model deep learning
* Sistem rekomendasi personalisasi lanjutan

---

📌 *Marketing Promotion Response DSS — Intelligent, Data-Driven, Decision-Oriented*
