# model_manager.py (FINAL - Sinkronisasi Data untuk DSS)

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from xgboost import XGBClassifier
from sklearn.metrics import roc_auc_score
from datetime import datetime

# --- Konstanta ---
CURRENT_YEAR = 2025
RANDOM_STATE = 42

# --- GLOBAL FEATURE LISTS (UNTUK SINKRONISASI) ---
NUMERICAL_FEATURES = [
    'Age', 'Income', 'Recency', 'Total_Spent', 'NumDealsPurchases', 'NumWebVisitsMonth', 
    'NumWebPurchases', 'NumStorePurchases', 'NumCatalogPurchases', 
    'Kidhome', 'Teenhome', 'Total_Purchases', 
    'Avg_Order_Value', 'Child_Category', 'Spending_vs_Income', 'Web_vs_Store_Ratio', 
    'Customer_Age_Days'
]
CATEGORICAL_FEATURES = ['Education', 'Marital_Status']
# --------------------------------------------------

def load_and_prepare_data(filepath='marketing_data.csv'):
    """Memuat data, menangani missing values, dan melakukan feature engineering."""
    try:
        data = pd.read_csv(filepath)
    except FileNotFoundError:
        print("❌ Error: File 'marketing_data.csv' tidak ditemukan. Pastikan file ada.")
        return pd.DataFrame(), pd.Series(), pd.DataFrame(), pd.DataFrame() 
    
    
    # 🌟 LANGKAH 1: Pembersihan dan Konversi Tipe Data
    
    data['Income'] = data['Income'].astype(str).str.replace('$', '', regex=False).str.replace(',', '', regex=False).str.strip()
    data['Income'] = pd.to_numeric(data['Income'], errors='coerce')
    
    mnt_cols = ['MntWines', 'MntFruits', 'MntMeatProducts', 'MntFishProducts', 'MntSweetProducts', 'MntGoldProds']
    purchase_cols = ['NumDealsPurchases', 'NumWebPurchases', 'NumCatalogPurchases', 'NumStorePurchases', 'NumWebVisitsMonth']

    for col in mnt_cols + purchase_cols:
        data[col] = pd.to_numeric(data[col], errors='coerce').fillna(0)
        
    data['Response'] = pd.to_numeric(data['Response'], errors='coerce').fillna(0).astype(int)
    data['ID'] = pd.to_numeric(data['ID'], errors='coerce')


    # Penanganan Missing Values: Isi NaN di 'Income' dengan median
    median_income = data['Income'].median()
    data['Income'] = data['Income'].fillna(median_income) 
    
    # Menghapus Outlier
    data = data[data['Income'] < 200000].copy()
    
    if data.empty:
        print("❌ Peringatan: Data kosong setelah pembersihan. Membatalkan training.")
        return pd.DataFrame(), pd.Series(), pd.DataFrame(), pd.DataFrame()

    # -----------------------------------------------------------
    # MITIGASI: KONSOLIDASI KATEGORI MARITAL_STATUS YANG JARANG (NICHE)
    # -----------------------------------------------------------
    niche_statuses = ['Absurd', 'YOLO', 'Alone'] 
    data['Marital_Status'] = data['Marital_Status'].replace(niche_statuses, 'Niche_Other')
    # -----------------------------------------------------------

    # 🌟 LANGKAH 2: Feature Engineering Dasar
    data['Age'] = CURRENT_YEAR - data['Year_Birth']
    data['Total_Spent'] = data[mnt_cols].sum(axis=1)
    data['Total_Purchases'] = data[['NumWebPurchases', 'NumCatalogPurchases', 'NumStorePurchases']].sum(axis=1)

    # 🌟 LANGKAH 3: Feature Engineering TINGKAT LANJUT
    
    # 3a. Average Order Value (AOV)
    data['Avg_Order_Value'] = data['Total_Spent'] / (data['Total_Purchases'].replace(0, 1))

    # 3b. Child Category (Tanggung Jawab Anak)
    data['Child_Category'] = data['Kidhome'] + data['Teenhome']

    # 3c. Spending vs. Income Ratio
    data['Spending_vs_Income'] = data['Total_Spent'] / (data['Income'] + 1000)

    # 3d. Proporsi Belanja Digital (Web vs. Store Ratio)
    data['Web_vs_Store_Ratio'] = data['NumWebPurchases'] / (data['NumStorePurchases'].replace(0, 1))
    
    # 3e. Usia Akun (Customer Age/Loyalty)
    data['Dt_Customer'] = pd.to_datetime(data['Dt_Customer'], format='%m/%d/%Y', errors='coerce')
    data['Dt_Customer'] = data['Dt_Customer'].fillna(datetime(CURRENT_YEAR, 1, 1)) 
    data['Customer_Age_Days'] = (datetime(CURRENT_YEAR, 1, 1) - data['Dt_Customer']).dt.days

    # 🌟 LANGKAH 4: Tentukan Fitur Input (X) dan Target (y)
    features = NUMERICAL_FEATURES + CATEGORICAL_FEATURES
    
    # X_model: DataFrame khusus untuk input model (hanya fitur yang terdaftar)
    X_model = data[features].copy() 
    y = data['Response']
    data_id_response = data[['ID', 'Response']].copy()
    
    # RETURN 4 NILAI: X_model (untuk training), y, data_id_response, dan data (full)
    return X_model, y, data_id_response, data 

def setup_preprocessor():
    """Mendefinisikan pipeline preprocessing untuk normalisasi, imputasi, dan encoding."""
    numerical_features = NUMERICAL_FEATURES
    categorical_features = CATEGORICAL_FEATURES

    # Pipeline untuk Numerik: Imputasi (Median) -> Scaling
    numerical_pipeline = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')), 
        ('scaler', StandardScaler())
    ])
    
    # Pipeline untuk Kategorikal: Imputasi (Modus) -> One-Hot Encoding
    categorical_pipeline = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')), 
        ('onehot', OneHotEncoder(handle_unknown='ignore'))
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numerical_pipeline, numerical_features),
            ('cat', categorical_pipeline, categorical_features)
        ],
        remainder='passthrough'
    )
    return preprocessor

def train_best_model(X_model, y):
    """Melatih model terbaik (XGBoost) dan mengembalikan pipeline."""
    preprocessor = setup_preprocessor()
    X_train, _, y_train, _ = train_test_split(X_model, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y)
    
    # Mengatasi class imbalance
    scale_pos_weight_value = (y_train == 0).sum() / (y_train == 1).sum() 

    best_model = XGBClassifier(random_state=RANDOM_STATE, 
                               use_label_encoder=False, 
                               eval_metric='logloss', 
                               n_estimators=100, 
                               learning_rate=0.1,
                               scale_pos_weight=scale_pos_weight_value
                               ) 
    
    pipeline = Pipeline(steps=[('preprocessor', preprocessor),
                               ('classifier', best_model)])
    
    pipeline.fit(X_train, y_train)
    return pipeline

def get_feature_importances(pipeline, X_model):
    """Mendapatkan Feature Importances dari model XGBoost."""
    try:
        numerical_features = NUMERICAL_FEATURES
        categorical_features = CATEGORICAL_FEATURES
        
        ohe = pipeline.named_steps['preprocessor'].named_transformers_['cat'].named_steps['onehot']
        
        try:
            ohe_names = ohe.get_feature_names_out(categorical_features)
        except AttributeError:
            ohe_names = ohe.get_feature_names(categorical_features)
            
        feature_names = list(numerical_features) + list(ohe_names)
        importances = pipeline.named_steps['classifier'].feature_importances_
        
        if len(feature_names) != len(importances):
             raise ValueError(f"Sinkronisasi Gagal: Jumlah nama fitur ({len(feature_names)}) tidak sesuai dengan jumlah importances ({len(importances)}).")

        feature_importance_df = pd.DataFrame({
            'Feature': feature_names,
            'Importance': importances
        }).sort_values(by='Importance', ascending=False)
        
        return feature_importance_df
    except Exception as e:
        print(f"Error fatal saat menghitung Feature Importances: {e}") 
        return pd.DataFrame()


def calculate_topsis_score(df_target):
    """
    Menerapkan algoritma TOPSIS pada pelanggan prioritas untuk mendapatkan peringkat akhir.
    """
    
    CRITERIA_COLS = ['Probability_Respond', 'Total_Spent', 'Income', 'Recency']
    WEIGHTS = np.array([0.4, 0.3, 0.2, 0.1]) 
    CRITERIA_TYPE = np.array(['benefit', 'benefit', 'benefit', 'cost'])
    
    # Hanya TOPSIS pada pelanggan yang memenuhi ambang batas potensi (prob > 0.40)
    df_mcdm = df_target[df_target['Probability_Respond'] >= 0.40].copy() 
    
    if df_mcdm.empty:
        df_target['TOPSIS_Score'] = 0.0
        return df_target, pd.DataFrame({'Kriteria': CRITERIA_COLS, 'Bobot': WEIGHTS})
        
    X_mcdm = df_mcdm[CRITERIA_COLS].values
    
    norm_sq = np.sqrt(np.sum(X_mcdm**2, axis=0))
    norm_sq[norm_sq == 0] = 1e-6
    X_normalized = X_mcdm / norm_sq
    
    X_weighted = X_normalized * WEIGHTS
    
    A_plus = np.zeros(len(CRITERIA_COLS))
    A_minus = np.zeros(len(CRITERIA_COLS))
    
    for i, type in enumerate(CRITERIA_TYPE):
        if type == 'benefit':
            A_plus[i] = np.max(X_weighted[:, i])
            A_minus[i] = np.min(X_weighted[:, i])
        else:
            A_plus[i] = np.min(X_weighted[:, i])
            A_minus[i] = np.max(X_weighted[:, i])
            
    D_plus = np.sqrt(np.sum((X_weighted - A_plus)**2, axis=1))
    D_minus = np.sqrt(np.sum((X_weighted - A_minus)**2, axis=1))
    
    Skor_TOPSIS = D_minus / (D_minus + D_plus)
    
    df_mcdm['TOPSIS_Score'] = Skor_TOPSIS
    
    df_target = pd.merge(df_target, df_mcdm[['CustomerID', 'TOPSIS_Score']], on='CustomerID', how='left')
    df_target['TOPSIS_Score'] = df_target['TOPSIS_Score'].fillna(0)
    
    df_target.sort_values(by='TOPSIS_Score', ascending=False, inplace=True)
    
    return df_target, pd.DataFrame({'Kriteria': CRITERIA_COLS, 'Bobot': WEIGHTS})

def predict_single_customer(pipeline, input_data):
    """Membuat prediksi probabilitas untuk satu input pelanggan."""
    input_df = pd.DataFrame([input_data])
    proba = pipeline.predict_proba(input_df)[:, 1]
    return proba[0]

def categorize_response(prob):
    """Mengkategorikan probabilitas respon."""
    if prob >= 0.70: 
        return "Sangat Berpotensi"
    elif prob >= 0.40: 
        return "Berpotensi"
    else:
        return "Tidak Disarankan Target"

def recommend_promotion(prob):
    """Memberikan rekomendasi aksi promosi berdasarkan probabilitas respon."""
    if prob >= 0.70:
        return "Diskon Eksklusif (High Value)"
    elif prob >= 0.40:
        return "Promosi Taktis (Free Shipping/Min. Order)"
    else:
        return "Konten / Promosi General"

def evaluate_model(pipeline, X_model, y):
    """Menghitung metrik evaluasi AUC."""
    if X_model.empty or y.empty:
        return 0.5
        
    _, X_test, _, y_test = train_test_split(X_model, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y)
    
    y_proba = pipeline.predict_proba(X_test)[:, 1]
    auc = roc_auc_score(y_test, y_proba)
    return auc