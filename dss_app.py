# dss_app.py - Aplikasi DSS Marketing Promotion Response
# Versi Dinamis: Insight dan Rekomendasi Berbasis Data Aktual

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime

# Pastikan model_manager.py berada di direktori yang sama
from model_manager import (
    load_and_prepare_data, train_best_model, 
    predict_single_customer, categorize_response, 
    get_feature_importances, evaluate_model,
    recommend_promotion, calculate_topsis_score
)

# --- KONSTANTA KONVERSI RUPIAH ---
EXCHANGE_RATE = 15000  # 1 USD = 15,000 IDR
# --- Konstanta Tahun ---
CURRENT_YEAR = 2025

# --- Fungsi Utility Format Rupiah ---
def format_rupiah(value):
    if pd.isna(value) or value is None:
        return "Rp 0"
    try:
        # Mengkonversi ke Rupiah dan format
        rupiah_value = int(round(value * EXCHANGE_RATE))
        return f"Rp {rupiah_value:,.0f}".replace(",", ".")
    except (TypeError, ValueError):
        return "Rp 0"

# --- Setup Awal ---
st.set_page_config(
    layout="wide", 
    page_title="Marketing Promotion Response DSS", 
    initial_sidebar_state="expanded",
    page_icon="🎯"
)

# Load external CSS
with open("styles.css") as css:
    st.markdown(f"<style>{css.read()}</style>", unsafe_allow_html=True)

# =============================================
# FUNGSI UNTUK ANALISIS PRODUK (DINAMIS)
# =============================================

def analyze_product_age_range(data_full, product_col):
    """Menganalisis range umur yang cocok untuk produk tertentu."""
    if product_col not in data_full.columns:
        return None
    
    # Filter hanya yang pernah beli produk ini (spending > 0)
    buyers = data_full[data_full[product_col] > 0].copy()
    
    if len(buyers) < 5:
        return None
    
    # Hitung statistik umur
    age_stats = {
        'min_age': buyers['Age'].min(),
        'max_age': buyers['Age'].max(),
        'avg_age': buyers['Age'].mean(),
        'median_age': buyers['Age'].median(),
        'q25_age': buyers['Age'].quantile(0.25),
        'q75_age': buyers['Age'].quantile(0.75),
        'total_buyers': len(buyers),
        'avg_spending': buyers[product_col].mean(),
        'total_spending': buyers[product_col].sum()
    }
    
    # Segmentasi umur
    age_bins = [18, 25, 35, 45, 55, 65, 100]
    age_labels = ['18-24', '25-34', '35-44', '45-54', '55-64', '65+']
    
    buyers['Age_Group'] = pd.cut(buyers['Age'], bins=age_bins, labels=age_labels, right=False)
    age_group_counts = buyers['Age_Group'].value_counts().sort_index()
    
    age_stats['age_groups'] = age_group_counts
    age_stats['best_age_group'] = age_group_counts.idxmax() if not age_group_counts.empty else None
    
    return age_stats

def get_product_display_name(product_code):
    """Mengubah kode produk menjadi nama yang mudah dibaca."""
    product_names = {
        'wines': '🍷 Wine / Anggur',
        'fruits': '🍎 Buah-buahan', 
        'meatproducts': '🥩 Produk Daging',
        'fishproducts': '🐟 Produk Ikan',
        'sweetproducts': '🍬 Makanan Manis',
        'goldprods': '💎 Produk Premium'
    }
    return product_names.get(product_code, product_code)

def get_product_column_name(product_code):
    """Mengubah kode produk menjadi nama kolom di dataset."""
    product_cols = {
        'wines': 'MntWines',
        'fruits': 'MntFruits',
        'meatproducts': 'MntMeatProducts',
        'fishproducts': 'MntFishProducts',
        'sweetproducts': 'MntSweetProducts',
        'goldprods': 'MntGoldProds'
    }
    return product_cols.get(product_code)

def generate_dynamic_insights(data_full, product_col, product_name):
    """
    Generate insight DINAMIS berdasarkan data aktual.
    Tidak mengandalkan hardcoded values.
    """
    if product_col not in data_full.columns:
        return None
    
    # Filter pembeli produk ini
    buyers = data_full[data_full[product_col] > 0].copy()
    
    if len(buyers) < 5:
        return None
    
    # Analisis demografis pembeli
    age_mean = buyers['Age'].mean()
    age_median = buyers['Age'].median()
    age_q25 = buyers['Age'].quantile(0.25)
    age_q75 = buyers['Age'].quantile(0.75)
    
    income_mean = buyers['Income'].mean()
    income_median = buyers['Income'].median()
    
    # Analisis keluarga
    avg_children = buyers['Child_Category'].mean()
    
    # Analisis pendidikan (jika ada)
    if 'Education' in buyers.columns:
        top_education = buyers['Education'].mode()[0] if not buyers['Education'].mode().empty else 'Unknown'
    else:
        top_education = 'Unknown'
    
    # Analisis status pernikahan (jika ada)
    if 'Marital_Status' in buyers.columns:
        top_marital = buyers['Marital_Status'].mode()[0] if not buyers['Marital_Status'].mode().empty else 'Unknown'
    else:
        top_marital = 'Unknown'
    
    # Tentukan best age range berdasarkan data
    if age_q75 - age_q25 < 10:
        best_age_range = f"{int(age_q25)}-{int(age_q75)}"
        age_description = f"Target spesifik pada umur {best_age_range} tahun"
    else:
        best_age_range = f"{int(age_q25)}-{int(age_q75)}"
        age_description = f"Target luas dari {int(age_q25)} hingga {int(age_q75)} tahun"
    
    # Tentukan income level berdasarkan data
    if income_mean < 40000:
        income_level = "Rendah-Menengah"
        income_description = f"Target dengan pendapatan di bawah ${income_mean:,.0f}"
    elif income_mean < 70000:
        income_level = "Menengah"
        income_description = f"Target dengan pendapatan sedang (~${income_mean:,.0f})"
    else:
        income_level = "Menengah-Tinggi"
        income_description = f"Target dengan pendapatan tinggi (~${income_mean:,.0f})"
    
    # Tentukan family impact berdasarkan korelasi
    if avg_children > 1:
        family_insight = "Produk ini SANGAT POPULER di kalangan keluarga dengan banyak anak"
        family_impact = "Tinggi (Positif)"
    elif avg_children > 0.5:
        family_insight = "Produk ini cukup populer di kalangan keluarga dengan anak"
        family_impact = "Sedang (Positif)"
    else:
        family_insight = "Produk ini lebih populer di kalangan single/DINK (Dual Income No Kids)"
        family_impact = "Rendah/Negatif"
    
    insights = {
        'total_buyers': len(buyers),
        'best_age_range': best_age_range,
        'age_description': age_description,
        'age_mean': age_mean,
        'age_median': age_median,
        'age_q25': age_q25,
        'age_q75': age_q75,
        'income_level': income_level,
        'income_description': income_description,
        'income_mean': income_mean,
        'income_median': income_median,
        'avg_spending': buyers[product_col].mean(),
        'top_education': top_education,
        'top_marital': top_marital,
        'avg_children': avg_children,
        'family_insight': family_insight,
        'family_impact': family_impact,
        'key_factor': f"Umur {best_age_range}, Pendapatan {income_level}, Status: {top_marital}"
    }
    
    return insights

def calculate_product_suitability(age, income, kidhome, teenhome, product_code, data_full):
    """
    Menghitung kecocokan produk berdasarkan profil pelanggan.
    Menggunakan data dari dataset untuk membuat scoring yang adaptif.
    """
    product_col = get_product_column_name(product_code)
    
    if product_col not in data_full.columns:
        return 0.5
    
    # Dapatkan insights dinamis
    insights = generate_dynamic_insights(data_full, product_col, product_code)
    if not insights:
        return 0.5
    
    # Normalisasi skor (0-1)
    def normalize(value, min_val, max_val):
        if max_val - min_val == 0:
            return 0.5
        return (value - min_val) / (max_val - min_val)
    
    # Bobot untuk setiap faktor
    weights = {
        'age': 0.4,
        'income': 0.3,
        'family': 0.2,
        'product_specific': 0.1
    }
    
    # Hitung skor usia berdasarkan data ideal dari dataset
    age_q25 = insights['age_q25']
    age_q75 = insights['age_q75']
    age_range = age_q75 - age_q25
    
    if age_q25 <= age <= age_q75:
        age_score = 1.0
    elif age < age_q25:
        age_score = max(0.1, 1 - (age_q25 - age) / (age_range + 10))
    else:
        age_score = max(0.1, 1 - (age - age_q75) / (age_range + 10))
    
    # Hitung skor pendapatan berdasarkan income median dari dataset
    income_median = insights['income_median']
    income_range = income_median * 0.5  # Range 50% dari median
    
    income_score = normalize(income, income_median - income_range, income_median + income_range)
    income_score = min(1.0, max(0.1, income_score))
    
    # Hitung skor keluarga berdasarkan data
    total_children = kidhome + teenhome
    avg_children = insights['avg_children']
    
    if avg_children < 0.5:  # Produk lebih populer di kalangan DINK
        if total_children == 0:
            family_score = 1.0
        else:
            family_score = max(0.3, 1 - (total_children * 0.2))
    else:  # Produk lebih populer di kalangan keluarga
        if total_children == 0:
            family_score = 0.5
        else:
            family_score = min(1.0, 0.5 + (total_children * 0.15))
    
    # Skor produk spesifik
    product_specific_score = 0.7  # Base score
    
    # Hitung total skor
    total_score = (
        age_score * weights['age'] +
        income_score * weights['income'] +
        family_score * weights['family'] +
        product_specific_score * weights['product_specific']
    )
    
    return min(1.0, max(0.0, total_score))

def rank_products_by_suitability(age, income, kidhome, teenhome, data_full):
    """Mengurutkan produk berdasarkan kecocokan dengan profil pelanggan."""
    product_codes = ['wines', 'fruits', 'meatproducts', 'fishproducts', 'sweetproducts', 'goldprods']
    
    suitability_scores = {}
    for product_code in product_codes:
        score = calculate_product_suitability(age, income, kidhome, teenhome, product_code, data_full)
        suitability_scores[product_code] = score
    
    # Urutkan dari tertinggi ke terendah
    ranked_products = sorted(suitability_scores.items(), key=lambda x: x[1], reverse=True)
    
    # Konversi ke format yang lebih baik
    result = []
    for product_code, score in ranked_products:
        display_name = get_product_display_name(product_code)
        result.append({
            'product': display_name,
            'code': product_code,
            'score': score,
            'percentage': f"{score * 100:.1f}%",
            'recommendation': get_recommendation_based_on_score(score)
        })
    
    return result

def get_recommendation_based_on_score(score):
    """Memberikan rekomendasi berdasarkan skor."""
    if score >= 0.8:
        return "Sangat direkomendasikan! 🎯"
    elif score >= 0.6:
        return "Cukup direkomendasikan ✅"
    elif score >= 0.4:
        return "Bisa dicoba ⚠️"
    else:
        return "Kurang direkomendasikan ❌"

def get_dynamic_recommendation(suitability_score, ml_prob):
    """
    Generate rekomendasi DINAMIS berdasarkan suitability dan ML probability.
    Tidak hardcoded berdasarkan nilai tertentu.
    """
    suit_weight = 0.6
    ml_weight = 0.4
    
    combined_score = (suitability_score * suit_weight) + (ml_prob * ml_weight)
    
    if combined_score >= 0.75:
        return {
            'verdict': '✅ KOMBINASI IDEAL!',
            'description': 'Produk sangat cocok dengan profil pelanggan dan peluang respons tinggi',
            'action': 'Lakukan promosi langsung dengan diskon menarik',
            'type': 'high'
        }
    elif combined_score >= 0.55:
        return {
            'verdict': '⚠️ KOMBINASI MENARIK',
            'description': 'Produk cukup cocok dengan profil dan peluang respons sedang',
            'action': 'Uji dengan promosi terbatas atau bundling',
            'type': 'medium'
        }
    elif combined_score >= 0.35:
        return {
            'verdict': '⚠️ PERLU PERTIMBANGAN',
            'description': 'Produk kurang cocok atau peluang respons rendah',
            'action': 'Fokus pada engagement atau produk alternatif',
            'type': 'low'
        }
    else:
        return {
            'verdict': '❌ TIDAK DIREKOMENDASIKAN',
            'description': 'Kombinasi kecocokan dan peluang respons sangat rendah',
            'action': 'Pertimbangkan segmen customer atau produk lain',
            'type': 'very_low'
        }

# =============================================
# INISIALISASI DATA DAN MODEL DSS UTAMA
# =============================================

# Inisialisasi State dan Load Data/Model
@st.cache_resource
def load_data_and_train_model():
    """Fungsi untuk memuat data dan melatih model (di-cache)."""
    # Mengharapkan 4 nilai dari load_and_prepare_data()
    X_model, y, data_id_response, data_full = load_and_prepare_data()
    
    if X_model.empty or y.empty:
        # Mengembalikan 6 nilai kosong
        return pd.DataFrame(), pd.Series(), pd.DataFrame(), pd.DataFrame(), None, 0.5 
        
    pipeline = train_best_model(X_model, y)
    auc = evaluate_model(pipeline, X_model, y)
    
    # Mengembalikan 6 nilai
    return X_model, y, data_id_response, data_full, pipeline, auc

# MEMUAT 6 NILAI DARI CACHE
X, y, data_id_response, data_full, dss_pipeline, auc = load_data_and_train_model() 

if dss_pipeline is None:
    st.error("🚨 **Error:** Gagal memuat atau melatih model. Pastikan file `marketing_data.csv` ada di direktori yang sama dan memiliki format yang benar.")
    st.stop()

# =============================================
# HALAMAN 1: HOME - PREDIKSI RESPONS PROMOSI
# =============================================
def page_home():
    """Halaman Home / Prediksi Respons Promosi dengan Analisis Produk."""
    st.markdown("""
<div class='hero-banner'>
    <div class='hero-title'>🎯 Marketing Promotion Response DSS</div>
    <div class='hero-subtitle'>Sistem cerdas untuk memprediksi respons pelanggan & memberikan rekomendasi produk</div>
</div>
""", unsafe_allow_html=True)

    st.title("🎯 Sistem Prediksi Respons Promosi")
    st.markdown("### Analisis Potensi Pelanggan & Rekomendasi Produk (Berbasis Data)")
    
    # Tampilkan metrik sistem
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Akurasi Model (AUC)", f"{auc:.3f}")
    with col2:
        st.metric("Total Pelanggan", f"{len(X):,}")
    with col3:
        st.metric("Model Utama", "XGBoost")
    
    st.markdown("---")
    
    # Container utama dengan tabs
    tab1, tab2 = st.tabs(["👤 Prediksi Individual", "📊 Analisis Produk"])
    
    # TAB 1: PREDIKSI INDIVIDUAL
    with tab1:
        st.subheader("👤 Analisis Kecocokan Produk & Prediksi Respons")
        # Container utama input data pelanggan
        with st.container(border=True):
            # Pilihan produk yang ingin dipapp.pyromosikan
            st.markdown("### 🎯 Produk yang Ingin Dipromosikan")
            product_options = {
                "🍷 Wine / Anggur": "wines",
                "🍎 Buah-buahan": "fruits", 
                "🥩 Produk Daging": "meatproducts",
                "🐟 Produk Ikan": "fishproducts",
                "🍬 Makanan Manis": "sweetproducts",
                "💎 Produk Premium": "goldprods"
            }
            
            selected_product_display = st.selectbox(
                "Pilih produk yang ingin dipromosikan:",
                options=list(product_options.keys()),
                index=0,
                help="Pilih produk yang ingin Anda tawarkan kepada pelanggan"
            )
            selected_product = product_options[selected_product_display]
            
            st.markdown("---")
            st.markdown("### 📋 Data Demografi Pelanggan")
            
            # Section 1: Data Demografi
            col_age, col_inc, col_rec = st.columns(3)
            
            # Nilai default dari data training
            default_age = int(X['Age'].mean()) if not X.empty else 35
            default_income = X['Income'].median() if not X.empty else 50000
            default_recency = X['Recency'].mean() if not X.empty else 50
            
            age = col_age.number_input("Usia (Tahun)", min_value=18, max_value=100, value=default_age)
            income_idr = col_inc.number_input(
                f"Pendapatan Tahunan (Rp)", 
                min_value=0, 
                max_value=200000*EXCHANGE_RATE, 
                value=int(default_income * EXCHANGE_RATE),
                help=f"Contoh: {format_rupiah(default_income)}"
            )
            recency = col_rec.number_input(
                "Recency (Hari sejak Pembelian Terakhir)", 
                min_value=0, 
                max_value=100, 
                value=int(default_recency)
            )
            
            col_edu, col_mar, col_dt = st.columns(3)
            
            # Ambil nilai unik untuk dropdown
            education_options = X['Education'].unique().tolist() if not X.empty else ['Graduation', 'PhD', 'Master', 'Basic', '2n Cycle']
            marital_options = X['Marital_Status'].unique().tolist() if not X.empty else ['Married', 'Single', 'Together', 'Divorced', 'Widow']
            
            education = col_edu.selectbox("Pendidikan", education_options)
            marital_status = col_mar.selectbox("Status Pernikahan", marital_options)
            
            # Tanggal bergabung (default: 3 tahun yang lalu)
            dt_customer = col_dt.date_input(
                "Tanggal Pelanggan Bergabung", 
                value=datetime(CURRENT_YEAR, 1, 1).date().replace(year=CURRENT_YEAR - 3),
                max_value=datetime(CURRENT_YEAR, 1, 1).date()
            )
            
            st.markdown("---")
            st.markdown("### 🛒 Data Pembelian & Keluarga")
            
            # Section 2: Data Pembelian & Keluarga
            # Nilai default
            default_total_spent = X['Total_Spent'].median() if not X.empty else 500
            default_web_purchases = X['NumWebPurchases'].median() if not X.empty else 4
            default_store_purchases = X['NumStorePurchases'].median() if not X.empty else 5
            default_catalog_purchases = X['NumCatalogPurchases'].median() if not X.empty else 2
            default_deals = X['NumDealsPurchases'].median() if not X.empty else 2
            default_web_visits = X['NumWebVisitsMonth'].median() if not X.empty else 5
            default_kidhome = X['Kidhome'].mode()[0] if not X.empty and not X['Kidhome'].mode().empty else 0
            default_teenhome = X['Teenhome'].mode()[0] if not X.empty and not X['Teenhome'].mode().empty else 0
            
            col_spent, col_w, col_s = st.columns(3)
            total_spent_idr = col_spent.number_input(
                f"Total Pengeluaran Tahunan (Rp)", 
                min_value=0, 
                max_value=int(10000*EXCHANGE_RATE), 
                value=int(default_total_spent * EXCHANGE_RATE), 
                help=f"Contoh: {format_rupiah(default_total_spent)}"
            )
            num_web_purchases = col_w.number_input(
                "Pembelian via Web (Frekuensi)", 
                min_value=0, 
                value=int(default_web_purchases)
            )
            num_store_purchases = col_s.number_input(
                "Pembelian via Store (Frekuensi)", 
                min_value=0, 
                value=int(default_store_purchases)
            )
            
            col_cat, col_k, col_t = st.columns(3)
            num_catalog_purchases = col_cat.number_input(
                "Pembelian via Catalog (Frekuensi)", 
                min_value=0, 
                value=int(default_catalog_purchases)
            )
            kidhome = col_k.number_input(
                "Jumlah Anak Kecil (<10 tahun)", 
                min_value=0, 
                max_value=3, 
                value=int(default_kidhome)
            )
            teenhome = col_t.number_input(
                "Jumlah Remaja (10-18 tahun)", 
                min_value=0, 
                max_value=3, 
                value=int(default_teenhome)
            )
            
            # Informasi fitur otomatis
            with st.expander("📊 Nilai Default Berdasarkan Data Training"):
                st.info(f"""
                **Fitur yang diisi otomatis:**
                - Jumlah Pembelian Diskon: **{int(default_deals)}**
                - Kunjungan Web per Bulan: **{int(default_web_visits)}**
                
                *Nilai ini diambil dari median data training untuk mengisi data yang tidak diinput.*
                """)
            
            # Tombol prediksi
            submitted = st.button(
                "🚀 ANALISIS KECOCOKAN & PREDIKSI RESPONS", 
                use_container_width=True, 
                type="primary"
            )
            
            if submitted:
                # KONVERSI INPUT RUPIAH KEMBALI KE USD UNTUK MODEL
                income_usd = income_idr / EXCHANGE_RATE
                total_spent_usd = total_spent_idr / EXCHANGE_RATE

                # Hitung fitur turunan
                total_purchases = num_web_purchases + num_catalog_purchases + num_store_purchases
                
                if isinstance(dt_customer, datetime):
                    dt_customer = dt_customer.date()

                customer_age_days = (datetime(CURRENT_YEAR, 1, 1).date() - dt_customer).days 
                
                # Siapkan data input untuk model
                input_data = {
                    'Age': age, 
                    'Income': income_usd, 
                    'Recency': recency, 
                    'Total_Spent': total_spent_usd,
                    'Education': education, 
                    'Marital_Status': marital_status if marital_status not in ['Absurd', 'YOLO', 'Alone'] else 'Niche_Other',
                    'NumWebPurchases': num_web_purchases, 
                    'NumStorePurchases': num_store_purchases, 
                    'NumCatalogPurchases': num_catalog_purchases,
                    'Kidhome': kidhome, 
                    'Teenhome': teenhome, 
                    'NumDealsPurchases': int(default_deals), 
                    'NumWebVisitsMonth': int(default_web_visits), 
                    'Total_Purchases': total_purchases, 
                    'Avg_Order_Value': total_spent_usd / (total_purchases if total_purchases > 0 else 1),
                    'Child_Category': kidhome + teenhome,
                    'Spending_vs_Income': total_spent_usd / (income_usd + 1000),
                    'Web_vs_Store_Ratio': num_web_purchases / (num_store_purchases if num_store_purchases > 0 else 1),
                    'Customer_Age_Days': customer_age_days
                }
                
                # ANALISIS KECOCOKAN PRODUK (DINAMIS)
                st.markdown("---")
                st.subheader("📊 Analisis Kecocokan Produk (Berbasis Data Aktual)")
                
                # 1. Hitung kecocokan untuk produk yang dipilih (DINAMIS dari data)
                suitability_score = calculate_product_suitability(age, income_usd, kidhome, teenhome, selected_product, data_full)
                
                col_suit1, col_suit2 = st.columns(2)
                
                with col_suit1:
                    st.markdown(f"### 🎯 {selected_product_display}")
                    st.metric(
                        label="Tingkat Kecocokan", 
                        value=f"{suitability_score * 100:.1f}%",
                        delta=f"{'Tinggi' if suitability_score >= 0.6 else 'Rendah'}"
                    )
                    
                    # Berikan penjelasan
                    if suitability_score >= 0.8:
                        st.success("✅ **Sangat Cocok!** Produk ini sangat sesuai dengan profil pelanggan.")
                    elif suitability_score >= 0.6:
                        st.info("⚠️ **Cukup Cocok** Produk ini cukup sesuai dengan profil pelanggan.")
                    elif suitability_score >= 0.4:
                        st.warning("⚠️ **Kurang Cocok** Mungkin perlu pertimbangan ulang.")
                    else:
                        st.error("❌ **Tidak Cocok** Produk ini tidak sesuai dengan profil pelanggan.")
                
                with col_suit2:
                    # Insight berdasarkan usia (dari data actual)
                    st.markdown("#### 📈 Insight Berdasarkan Data Aktual")
                    
                    # Dapatkan insights dinamis
                    product_col = get_product_column_name(selected_product)
                    product_insights = generate_dynamic_insights(data_full, product_col, selected_product)
                    
                    if product_insights:
                        st.info(f"""
                        **Profil Ideal Pembeli {selected_product_display}:**
                        - Range Umur: {product_insights['best_age_range']} tahun
                        - Pendapatan: {product_insights['income_level']}
                        - Total Pembeli: {product_insights['total_buyers']:,} orang
                        """)
                
                # 2. Ranking semua produk berdasarkan kecocokan
                st.markdown("---")
                st.subheader("🏆 Ranking Produk Berdasarkan Kecocokan")
                st.markdown(f"**Urutan produk yang paling cocok untuk usia {age} tahun:**")
                
                ranked_products = rank_products_by_suitability(age, income_usd, kidhome, teenhome, data_full)
                
                # Tampilkan dalam dataframe
                df_ranked = pd.DataFrame(ranked_products)
                df_ranked.index = df_ranked.index + 1  # Mulai dari 1
                
                # Tampilkan dataframe
                st.dataframe(
                    df_ranked[['product', 'percentage', 'recommendation']]
                    .rename(columns={'product': 'Produk', 'percentage': 'Tingkat Kecocokan', 'recommendation': 'Rekomendasi'}),
                    use_container_width=True
                )
                
                # PREDIKSI RESPONS PROMOSI
                st.markdown("---")
                st.subheader("🤖 Prediksi Respons Promosi (XGBoost)")
                
                try:
                    # Prediksi dengan model XGBoost
                    proba = predict_single_customer(dss_pipeline, input_data)
                    keterangan = categorize_response(proba)
                    rekomendasi = recommend_promotion(proba)
                    
                    # Tampilkan hasil
                    col_pred1, col_pred2, col_pred3 = st.columns(3)
                    
                    with col_pred1:
                        st.metric(
                            label="Probabilitas Respons", 
                            value=f"{proba*100:.1f}%",
                            delta=f"{'Tinggi' if proba >= 0.5 else 'Rendah'}",
                            help="Kemungkinan pelanggan merespons promosi apapun"
                        )
                    
                    with col_pred2:
                        st.metric(
                            label="Kategori Potensi", 
                            value=keterangan,
                            help="Kategorisasi berdasarkan probabilitas respons"
                        )

                    with col_pred3:
                        st.metric(
                            label="Rekomendasi Promosi", 
                            value=rekomendasi,
                            help="Jenis promosi yang direkomendasikan"
                        )
                    
                    # Visualisasi probabilitas
                    fig_gauge = go.Figure(go.Indicator(
                        mode="gauge+number",
                        value=proba*100,
                        domain={'x': [0, 1], 'y': [0, 1]},
                        title={'text': "Skor Respons Promosi"},
                        gauge={
                            'axis': {'range': [0, 100]},
                            'bar': {'color': "darkblue"},
                            'steps': [
                                {'range': [0, 40], 'color': "lightgray"},
                                {'range': [40, 70], 'color': "yellow"},
                                {'range': [70, 100], 'color': "lightgreen"}
                            ],
                            'threshold': {
                                'line': {'color': "red", 'width': 4},
                                'thickness': 0.75,
                                'value': 70
                            }
                        }
                    ))
                    
                    fig_gauge.update_layout(height=250)
                    st.plotly_chart(fig_gauge, use_container_width=True)
                    
                    # Kesimpulan akhir (DINAMIS dari kedua model)
                    st.markdown("---")
                    st.subheader("💡 Kesimpulan & Rekomendasi (Berbasis TOPSIS + XGBoost)")
                    
                    # Dapatkan rekomendasi dinamis
                    dynamic_rec = get_dynamic_recommendation(suitability_score, proba)
                    
                    col_concl1, col_concl2 = st.columns(2)
                    
                    with col_concl1:
                        st.markdown("#### 📋 Ringkasan Analisis")
                        
                        if dynamic_rec['type'] == 'high':
                            st.success(f"""
                            **{dynamic_rec['verdict']}**
                            
                            - {dynamic_rec['description']}
                            - **Rekomendasi:** {dynamic_rec['action']}
                            """)
                        elif dynamic_rec['type'] == 'medium':
                            st.info(f"""
                            **{dynamic_rec['verdict']}**
                            
                            - {dynamic_rec['description']}
                            - **Rekomendasi:** {dynamic_rec['action']}
                            """)
                        elif dynamic_rec['type'] == 'low':
                            st.warning(f"""
                            **{dynamic_rec['verdict']}**
                            
                            - {dynamic_rec['description']}
                            - **Rekomendasi:** {dynamic_rec['action']}
                            """)
                        else:
                            st.error(f"""
                            **{dynamic_rec['verdict']}**
                            
                            - {dynamic_rec['description']}
                            - **Rekomendasi:** {dynamic_rec['action']}
                            """)
                    
                    with col_concl2:
                        st.markdown("#### 🎯 Produk Alternatif Terbaik")
                        
                        # Tampilkan 3 produk terbaik selain yang dipilih
                        top_alternatives = [p for p in ranked_products if p['code'] != selected_product][:3]
                        
                        for i, product in enumerate(top_alternatives[:3], 1):
                            st.markdown(f"**{i}. {product['product']}**")
                            st.markdown(f"  Kecocokan: {product['percentage']} - {product['recommendation']}")
                            
                    # Analisis berdasarkan data historis
                    st.markdown("---")
                    st.subheader("📈 Analisis Data Historis")
                    
                    # Analisis produk berdasarkan data
                    product_col = get_product_column_name(selected_product)
                    if product_col and product_col in data_full.columns:
                        age_stats = analyze_product_age_range(data_full, product_col)
                        
                        if age_stats:
                            col_hist1, col_hist2 = st.columns(2)
                            
                            with col_hist1:
                                st.markdown("#### 📊 Statistik Pembeli")
                                st.info(f"""
                                **Data aktual pembeli {selected_product_display}:**
                                - Total pembeli: {age_stats['total_buyers']:,}
                                - Umur rata-rata: {age_stats['avg_age']:.1f} tahun
                                - Range umur ideal: {age_stats['q25_age']:.0f}-{age_stats['q75_age']:.0f} tahun
                                """)
                            
                            with col_hist2:
                                st.markdown("#### 🎯 Perbandingan")
                                if age_stats['q25_age'] <= age <= age_stats['q75_age']:
                                    st.success(f"✅ Usia {age} tahun **MASUK** dalam range ideal pembeli")
                                else:
                                    if age < age_stats['q25_age']:
                                        st.warning(f"⚠️ Usia {age} tahun **TERLALU MUDA** dari target ideal")
                                    else:
                                        st.warning(f"⚠️ Usia {age} tahun **TERLALU TUA** dari target ideal")
                        else:
                            st.info("Tidak ada data historis yang cukup untuk produk ini.")
                            
                except Exception as e:
                    st.error(f"❌ Error saat melakukan prediksi: {str(e)}")
                    st.info("Pastikan semua input data sudah diisi dengan benar.")
    
    # TAB 2: ANALISIS PRODUK
    with tab2:
        st.subheader("📈 Analisis Segmentasi Pelanggan per Produk (Berbasis Data)")
        
        # Pilihan produk untuk dianalisis
        product_options = {
            "🍷 Wine / Anggur": "wines",
            "🍎 Buah-buahan": "fruits", 
            "🥩 Produk Daging": "meatproducts",
            "🐟 Produk Ikan": "fishproducts",
            "🍬 Makanan Manis": "sweetproducts",
            "💎 Produk Premium": "goldprods"
        }
        
        selected_product_display = st.selectbox(
            "Pilih Produk untuk Analisis Segmentasi:",
            options=list(product_options.keys()),
            index=0,
            key="analisis_produk_selectbox"
        )
        selected_product = product_options[selected_product_display]
        
        # Analisis produk yang dipilih
        product_col = get_product_column_name(selected_product)
        
        if product_col:
            age_stats = analyze_product_age_range(data_full, product_col)
            
            if age_stats:
                # Tampilkan insight produk (DINAMIS dari data)
                insights = generate_dynamic_insights(data_full, product_col, selected_product)
                
                col_insight1, col_insight2 = st.columns(2)
                
                with col_insight1:
                    st.markdown("#### 🎯 Profil Target Ideal (Berbasis Data Aktual)")
                    st.info(f"**Insight:** {insights['family_insight']}")
                    st.info(f"**Range Umur Terbaik:** {insights['best_age_range']} tahun")
                    st.info(f"**Level Pendapatan:** {insights['income_level']}")
                    st.info(f"**Faktor Kunci:** {insights['key_factor']}")
                
                with col_insight2:
                    st.markdown("#### 📊 Statistik Pembeli Aktual")
                    st.metric("Total Pembeli", f"{insights['total_buyers']:,}")
                    st.metric("Rata-rata Pengeluaran", f"${insights['avg_spending']:,.0f}")
                    st.metric("Umur Rata-rata", f"{insights['age_mean']:.1f} tahun")
                    st.metric("Umur Median", f"{insights['age_median']:.1f} tahun")
                
                # Visualisasi distribusi umur
                st.markdown("#### 📈 Distribusi Umur Pembeli")
                
                if 'age_groups' in age_stats and not age_stats['age_groups'].empty:
                    fig_age = px.bar(
                        x=age_stats['age_groups'].index,
                        y=age_stats['age_groups'].values,
                        labels={'x': 'Kelompok Umur', 'y': 'Jumlah Pembeli'},
                        color=age_stats['age_groups'].values,
                        color_continuous_scale='blues'
                    )
                    fig_age.update_layout(title=f"Distribusi Umur Pembeli {selected_product_display}")
                    st.plotly_chart(fig_age, use_container_width=True)
                
                # Box plot pengeluaran vs umur
                st.markdown("#### 📦 Pengeluaran per Kelompok Umur")
                
                buyers = data_full[data_full[product_col] > 0].copy()
                if len(buyers) > 0:
                    buyers['Age_Group'] = pd.cut(buyers['Age'], bins=[18, 25, 35, 45, 55, 65, 100], 
                                                labels=['18-24', '25-34', '35-44', '45-54', '55-64', '65+'], right=False)
                    
                    fig_box = px.box(
                        buyers,
                        x='Age_Group',
                        y=product_col,
                        points='all',
                        title=f"Distribusi Pengeluaran {selected_product_display} per Kelompok Umur"
                    )
                    st.plotly_chart(fig_box, use_container_width=True)
                
                # Rekomendasi berdasarkan analisis
                st.markdown("#### 💡 Rekomendasi Pemasaran (Berbasis Data)")
                best_group = age_stats.get('best_age_group')
                if best_group:
                    st.success(f"""
                    **Fokuskan kampanye untuk {selected_product_display} pada kelompok umur {best_group} tahun!**
                    
                    Berdasarkan data aktual:
                    - Kelompok {best_group} tahun memiliki **{age_stats['age_groups'].get(best_group, 0)} pembeli**
                    - Range umur ideal: **{age_stats.get('q25_age', 0):.0f}-{age_stats.get('q75_age', 0):.0f} tahun**
                    - Status dominan: **{insights['top_marital']}**
                    - Pendidikan dominan: **{insights['top_education']}**
                    - Rata-rata anak di rumah: **{insights['avg_children']:.1f}**
                    - Strategi: Sesuaikan channel iklan dengan preferensi kelompok umur ini
                    """)
            else:
                st.warning(f"Data pembeli untuk {selected_product_display} tidak cukup untuk analisis mendetail.")
        else:
            st.warning(f"Kolom data untuk produk {selected_product} tidak ditemukan.")

# =============================================
# HALAMAN 2: DASHBOARD ANALISIS
# =============================================
def page_analysis():
    """Halaman Analisis / Dashboard Utama."""
    st.title("🏠 Dashboard Analisis Marketing")
    st.markdown("### Sistem Pendukung Keputusan untuk Penargetan Promosi E-Commerce")
    st.info("DSS ini mengombinasikan **Model Prediktif (XGBoost)** dengan **Analisis Data Dinamis** untuk memberikan insight yang adaptif terhadap perubahan dataset.")
    
    st.markdown("---")

    col_metrics, col_chart = st.columns([1, 2])
    
    with col_metrics:
        st.subheader("🎯 Kinerja Sistem")
        st.metric(label="Model Utama", value="XGBoost Classifier")
        st.metric(label="Metrik AUC (Akurasi Prediksi)", value=f"{auc:.4f}", help="Area Under the Curve, menunjukkan kemampuan model membedakan Responden vs Non-Responden.")
        st.metric(label="Total Pelanggan Analisis", value=f"{len(X):,}")
    
    with col_chart:
        st.subheader("📈 Top 10 Feature Importances (XGBoost)")
        
        importance_df = get_feature_importances(dss_pipeline, X)
        
        if not importance_df.empty:
            top_features = importance_df.head(10)
            fig = px.bar(top_features.iloc[::-1], x='Importance', y='Feature', orientation='h',
                         color='Importance', color_continuous_scale=px.colors.sequential.Teal)
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.warning("Gagal memuat Feature Importances.")

    st.markdown("---")
    st.subheader("🔍 Analisis Insight Bisnis Utama")
    
    # Periksa apakah X tidak kosong
    if X.empty:
        st.warning("Data tidak tersedia untuk analisis.")
        return
    
    # --- ROW 1: Status Pernikahan & Jumlah Anak ---
    col_ratio, col_income = st.columns(2)
    
    # VISUALISASI 1: Pengeluaran Rata-rata vs. Status Pernikahan
    with col_ratio:
        st.markdown("#### Rata-rata Pengeluaran (Rp) berdasarkan Status Pernikahan")
        df_spent_marital = X.groupby('Marital_Status')['Total_Spent'].mean().reset_index()
        df_spent_marital['Total_Spent_IDR'] = df_spent_marital['Total_Spent'] * EXCHANGE_RATE
        
        fig_spent = px.bar(df_spent_marital.sort_values(by='Total_Spent_IDR'), 
                           x='Total_Spent_IDR', y='Marital_Status', orientation='h',
                           color='Total_Spent_IDR', color_continuous_scale=px.colors.sequential.Sunset)
        st.plotly_chart(fig_spent, use_container_width=True)

    # VISUALISASI 2: Distribusi Total Pengeluaran vs. Jumlah Anak
    with col_income:
        st.markdown("#### Total Pengeluaran (Rp) vs. Jumlah Anak di Rumah")
        X['Total_Spent_IDR'] = X['Total_Spent'] * EXCHANGE_RATE
        
        fig_child = px.box(X, x='Child_Category', y='Total_Spent_IDR', 
                            labels={'Child_Category': 'Jumlah Anak Total', 'Total_Spent_IDR': 'Total Pengeluaran (Rp)'},
                            notched=True)
        st.plotly_chart(fig_child, use_container_width=True)

    st.markdown("---")

    # --- ROW 2: Pendidikan & Kategori Produk ---
    col_edu, col_prod = st.columns(2)

    # VISUALISASI 3: Pengeluaran vs. Pendidikan
    with col_edu:
        st.markdown("#### Distribusi Pengeluaran (Rp) vs. Tingkat Pendidikan")
        fig_edu = px.box(X, x='Education', y='Total_Spent_IDR', 
                        labels={'Education': 'Pendidikan', 'Total_Spent_IDR': 'Total Pengeluaran (Rp)'},
                        color='Education',
                        notched=True)
        st.plotly_chart(fig_edu, use_container_width=True)
        
    # VISUALISASI 4: Pembelian vs. Kategori Produk
    with col_prod:
        st.markdown("#### Rata-rata Pengeluaran ($) Berdasarkan Kategori Produk")
        
        product_cols = ['MntWines', 'MntFruits', 'MntMeatProducts', 'MntFishProducts', 'MntSweetProducts', 'MntGoldProds']
        
        # Periksa kolom yang ada di data_full
        available_cols = [col for col in product_cols if col in data_full.columns]
        
        if available_cols:
            df_products = data_full[available_cols].mean().reset_index()
            df_products.columns = ['Produk', 'Rata_Rata_Pengeluaran_USD']
            
            fig_prod = px.bar(df_products.sort_values(by='Rata_Rata_Pengeluaran_USD'), 
                              x='Rata_Rata_Pengeluaran_USD', y='Produk', orientation='h',
                              color='Produk')
            st.plotly_chart(fig_prod, use_container_width=True)
        else:
            st.warning("Data pengeluaran produk tidak tersedia.")

# =============================================
# HALAMAN 3: TARGET LIST (TOPSIS RANKING)
# =============================================
def page_target_list():
    """Halaman List Target Pelanggan dengan Ranking TOPSIS."""
    st.title("🥇 Ranking Prioritas Pelanggan (TOPSIS)")
    st.write("Daftar pelanggan yang diurutkan berdasarkan skor **TOPSIS** yang menggabungkan prediksi model dan kriteria bisnis.")

    # Periksa apakah data tersedia
    if X.empty:
        st.warning("Data tidak tersedia untuk analisis.")
        return
    
    # 1. Prediksi Probabilitas untuk semua pelanggan
    try:
        all_probabilities = dss_pipeline.predict_proba(X)[:, 1]
    except Exception as e:
        st.error(f"❌ Error saat prediksi: {str(e)}")
        return
    
    df_result = X.copy()
    df_result['CustomerID'] = data_id_response['ID'].values if not data_id_response.empty else range(len(X))
    df_result['Probability_Respond'] = all_probabilities
    df_result['Keterangan'] = df_result['Probability_Respond'].apply(categorize_response)
    df_result['Rekomendasi_Promosi'] = df_result['Probability_Respond'].apply(recommend_promotion)
    
    # 2. Penerapan TOPSIS
    try:
        df_ranked, df_weights = calculate_topsis_score(df_result)
    except Exception as e:
        st.error(f"❌ Error saat menghitung TOPSIS: {str(e)}")
        return

    st.markdown("---")
    
    # Bobot TOPSIS
    st.subheader("⚖️ Kriteria & Bobot TOPSIS")
    
    col_weights, col_pie = st.columns([2, 1])
    
    with col_weights:
        st.dataframe(df_weights.style.format({'Bobot': "{:.2f}"}), use_container_width=True)
        st.caption("""
        **Penjelasan Bobot:**
        - Probabilitas (40%): Kemungkinan merespons promosi
        - Total Pengeluaran (30%): Nilai ekonomi pelanggan
        - Pendapatan (20%): Potensi belanja masa depan
        - Recency (10%): Keaktifan terakhir (semakin kecil semakin baik)
        """)
    
    with col_pie:
        st.subheader("Distribusi Potensi")
        count_df = df_ranked['Keterangan'].value_counts().reset_index()
        count_df.columns = ['Keterangan', 'Jumlah']
        
        fig_pie = px.pie(count_df, values='Jumlah', names='Keterangan', hole=0.3,
                         color='Keterangan',
                         color_discrete_map={'Sangat Berpotensi':'#1e8449',
                                             'Berpotensi':'#e67e22',
                                             'Tidak Disarankan Target':'#3498db'})
        st.plotly_chart(fig_pie, use_container_width=True)

    st.markdown("---")
    st.subheader("🥇 Top Target Pelanggan")
    
    # Slider untuk jumlah pelanggan yang ditampilkan
    filter_limit = st.slider(
        "Tampilkan Pelanggan Top", 
        min_value=10, 
        max_value=min(200, len(df_ranked)), 
        value=50, 
        step=10
    )
    
    # Tampilkan tabel
    df_final_target = df_ranked[[
        'CustomerID', 'TOPSIS_Score', 'Probability_Respond', 'Keterangan',
        'Rekomendasi_Promosi', 'Total_Spent', 'Income', 'Recency', 'Age'
    ]].head(filter_limit).copy()

    # Format nilai uang
    df_final_target['Total_Spent_Rp'] = df_final_target['Total_Spent'].apply(format_rupiah)
    df_final_target['Income_Rp'] = df_final_target['Income'].apply(format_rupiah)
    
    # Tampilkan tabel dengan styling
    st.dataframe(
        df_final_target[['CustomerID', 'TOPSIS_Score', 'Probability_Respond', 'Keterangan', 
                         'Rekomendasi_Promosi', 'Total_Spent_Rp', 'Income_Rp', 'Recency', 'Age']]
        .rename(columns={
            'Total_Spent_Rp': 'Total Spent (Rp)',
            'Income_Rp': 'Income (Rp)',
            'Probability_Respond': 'Prob. Respons'
        })
        .style.format({
            'TOPSIS_Score': "{:.4f}", 
            'Probability_Respond': "{:.2%}", 
        })
        .bar(subset=['TOPSIS_Score'], color='#58d68d')
        .applymap(lambda x: 'color: green' if x == 'Sangat Berpotensi' else 
                 ('color: orange' if x == 'Berpotensi' else 'color: blue'), 
                 subset=['Keterangan']), 
        use_container_width=True,
        height=400
    )
    
    st.info("💡 **Catatan:** Skor TOPSIS 0.0000 berarti pelanggan tidak memenuhi ambang batas potensi (Probabilitas < 0.40) untuk ranking MCDM.")

# =============================================
# NAVIGASI SIDEBAR
# =============================================
st.sidebar.title("🎯 Marketing DSS")
st.sidebar.markdown("---")

# Informasi Sistem
st.sidebar.info("""
**Sistem Pendukung Keputusan** untuk:
1. Prediksi respons promosi
2. Analisis segmentasi produk  
3. Ranking target pelanggan

**Fitur Utama:**
- ✅ Insight berbasis data aktual
- ✅ Rekomendasi dinamis (TOPSIS + XGBoost)
- ✅ Adaptif terhadap perubahan dataset
""")

st.sidebar.markdown("---")

# Navigasi Halaman
st.sidebar.subheader("Navigasi Modul")

selection = st.sidebar.radio(
    "Pilih Modul",
    ["Prediksi Respons Promosi", "Dashboard Analisis", "Target List Pelanggan"],
    index=0  # Default ke halaman pertama (Prediksi Respons Promosi)
)

# Pemetaan halaman
if selection == "Prediksi Respons Promosi":
    page_home()
elif selection == "Dashboard Analisis":
    page_analysis()
elif selection == "Target List Pelanggan":
    page_target_list()

st.sidebar.markdown("---")
st.sidebar.caption("Dikembangkan oleh: Tim DSS Marketing")
st.sidebar.caption("Teknik Informatika UNPAD")
st.sidebar.caption("**Mode:** Dinamis Berbasis Data")