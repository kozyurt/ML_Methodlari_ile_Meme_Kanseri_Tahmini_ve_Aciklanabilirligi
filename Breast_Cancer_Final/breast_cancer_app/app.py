import streamlit as st
import pandas as pd
import numpy as np
import pickle
import os
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

# ─── Sayfa Yapılandırması ────────────────────────────────────────────────────
st.set_page_config(
    page_title="Meme Kanseri Tahmin Sistemi",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ─── CSS ─────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

    html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

    .main { background-color: #0f1117; }

    /* Sidebar */
    section[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #1a1d2e 0%, #16213e 100%);
        border-right: 1px solid #2d3561;
    }

    /* Metrik kartları */
    .metric-card {
        background: linear-gradient(135deg, #1a1d2e 0%, #16213e 100%);
        border: 1px solid #2d3561;
        border-radius: 12px;
        padding: 20px;
        text-align: center;
        margin-bottom: 12px;
    }
    .metric-value {
        font-size: 2rem;
        font-weight: 700;
        background: linear-gradient(135deg, #667eea, #764ba2);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    .metric-label {
        font-size: 0.8rem;
        color: #8892b0;
        text-transform: uppercase;
        letter-spacing: 1px;
        margin-top: 4px;
    }

    /* Sonuç kutusu */
    .result-benign {
        background: linear-gradient(135deg, #0d3b2e 0%, #0a2e22 100%);
        border: 2px solid #00b894;
        border-radius: 16px;
        padding: 28px;
        text-align: center;
    }
    .result-malignant {
        background: linear-gradient(135deg, #3b0d0d 0%, #2e0a0a 100%);
        border: 2px solid #d63031;
        border-radius: 16px;
        padding: 28px;
        text-align: center;
    }
    .result-title {
        font-size: 1.8rem;
        font-weight: 700;
        margin-bottom: 8px;
    }
    .result-subtitle {
        font-size: 0.9rem;
        color: #b2bec3;
    }

    /* Bölüm başlığı */
    .section-header {
        font-size: 1.4rem;
        font-weight: 600;
        color: #ccd6f6;
        border-left: 4px solid #667eea;
        padding-left: 12px;
        margin: 24px 0 16px 0;
    }

    /* Info banner */
    .info-banner {
        background: linear-gradient(135deg, #1a1d2e 0%, #16213e 100%);
        border: 1px solid #667eea;
        border-radius: 10px;
        padding: 16px 20px;
        color: #ccd6f6;
        margin-bottom: 20px;
    }

    /* Streamlit bileşen düzenlemeleri */
    .stSlider > div > div { color: #ccd6f6; }
    div[data-testid="metric-container"] {
        background: linear-gradient(135deg, #1a1d2e 0%, #16213e 100%);
        border: 1px solid #2d3561;
        border-radius: 10px;
        padding: 12px 16px;
    }
    div[data-testid="metric-container"] label { color: #8892b0 !important; }
    div[data-testid="metric-container"] div { color: #ccd6f6 !important; }

    /* Tab stilleri */
    .stTabs [data-baseweb="tab-list"] {
        background-color: #1a1d2e;
        border-radius: 10px;
        padding: 4px;
        gap: 4px;
    }
    .stTabs [data-baseweb="tab"] {
        background-color: transparent;
        color: #8892b0;
        border-radius: 8px;
        padding: 8px 20px;
    }
    .stTabs [aria-selected="true"] {
        background: linear-gradient(135deg, #667eea, #764ba2) !important;
        color: white !important;
    }

    /* Uyarı kutuları */
    .disclaimer {
        background-color: #1a1218;
        border: 1px solid #d63031;
        border-radius: 8px;
        padding: 12px 16px;
        color: #fab1a0;
        font-size: 0.8rem;
    }
</style>
""", unsafe_allow_html=True)

# ─── Feature Engineering Transformer (pickle için zorunlu) ───────────────────
from sklearn.base import BaseEstimator, TransformerMixin

class FeatureEngineeringTransformer(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.feature_names_ = None

    def fit(self, X, y=None):
        if isinstance(X, pd.DataFrame):
            self.feature_names_ = X.columns.tolist()
        return self

    def transform(self, X):
        if not isinstance(X, pd.DataFrame):
            if self.feature_names_:
                X = pd.DataFrame(X, columns=self.feature_names_)
            else:
                X = pd.DataFrame(X)
        X = X.copy()
        if 'radius_mean' in X.columns and 'area_mean' in X.columns:
            X['radius_area_ratio'] = X['radius_mean'] / (X['area_mean'] + 1e-6)
        if 'perimeter_mean' in X.columns and 'area_mean' in X.columns:
            X['perimeter_area_ratio'] = X['perimeter_mean'] / (X['area_mean'] + 1e-6)
        if 'concavity_mean' in X.columns and 'concave points_mean' in X.columns:
            X['concavity_points_product'] = X['concavity_mean'] * X['concave points_mean']
        return X

# ─── Model & Bilgi Yükleme ────────────────────────────────────────────────────
@st.cache_resource
def load_model():
    model_path = "breast_cancer_pipeline.pkl"
    if os.path.exists(model_path):
        with open(model_path, "rb") as f:
            return pickle.load(f)
    return None

@st.cache_data
def load_model_info():
    info_path = "model_info.pkl"
    if os.path.exists(info_path):
        with open(info_path, "rb") as f:
            return pickle.load(f)
    return {
        "model_name": "Logistic Regression",
        "accuracy": 0.9737,
        "f1_score": 0.9639,
        "roc_auc": 0.9861,
        "feature_names": [
            'radius_mean','texture_mean','perimeter_mean','area_mean',
            'smoothness_mean','compactness_mean','concavity_mean','concave points_mean',
            'symmetry_mean','fractal_dimension_mean','radius_se','texture_se',
            'perimeter_se','area_se','smoothness_se','compactness_se','concavity_se',
            'concave points_se','symmetry_se','fractal_dimension_se','radius_worst',
            'texture_worst','perimeter_worst','area_worst','smoothness_worst',
            'compactness_worst','concavity_worst','concave points_worst',
            'symmetry_worst','fractal_dimension_worst'
        ]
    }

pipeline = load_model()
model_info = load_model_info()
FEATURE_NAMES = model_info["feature_names"]

# ─── Feature açıklamaları ─────────────────────────────────────────────────────
FEATURE_DESCRIPTIONS = {
    'radius_mean': ('Ortalama Yarıçap', 'Tümör hücresinin ortalama yarıçapı (mm)'),
    'texture_mean': ('Ortalama Doku', 'Gri tonu standart sapması'),
    'perimeter_mean': ('Ortalama Çevre', 'Tümör çevresinin ortalaması (mm)'),
    'area_mean': ('Ortalama Alan', 'Tümör alanının ortalaması (mm²)'),
    'smoothness_mean': ('Ortalama Pürüzsüzlük', 'Yarıçap uzunluklarındaki yerel varyasyon'),
    'compactness_mean': ('Ortalama Kompaktlık', 'Çevre² / Alan - 1.0'),
    'concavity_mean': ('Ortalama Çukurluk', 'Konturun içbükey kısımlarının şiddeti'),
    'concave points_mean': ('Ort. Çukur Nokta Sayısı', 'Konturun içbükey kısımlarının sayısı'),
    'symmetry_mean': ('Ortalama Simetri', 'Hücrenin simetri ölçüsü'),
    'fractal_dimension_mean': ('Ort. Fraktal Boyut', '"Kıyı şeridi yaklaşımı" - 1'),
    'radius_se': ('Yarıçap Std. Hatası', 'Yarıçap ölçümlerinin standart hatası'),
    'texture_se': ('Doku Std. Hatası', 'Doku ölçümlerinin standart hatası'),
    'perimeter_se': ('Çevre Std. Hatası', 'Çevre ölçümlerinin standart hatası'),
    'area_se': ('Alan Std. Hatası', 'Alan ölçümlerinin standart hatası'),
    'smoothness_se': ('Pürüzsüzlük Std. Hatası', 'Pürüzsüzlük standart hatası'),
    'compactness_se': ('Kompaktlık Std. Hatası', 'Kompaktlık standart hatası'),
    'concavity_se': ('Çukurluk Std. Hatası', 'Çukurluk standart hatası'),
    'concave points_se': ('Çukur Nokta Std. Hatası', 'Çukur nokta sayısı standart hatası'),
    'symmetry_se': ('Simetri Std. Hatası', 'Simetri standart hatası'),
    'fractal_dimension_se': ('Fraktal Boyut Std. Hatası', 'Fraktal boyut standart hatası'),
    'radius_worst': ('En Kötü Yarıçap', 'En büyük 3 ölçümün ortalaması - yarıçap'),
    'texture_worst': ('En Kötü Doku', 'En büyük 3 ölçümün ortalaması - doku'),
    'perimeter_worst': ('En Kötü Çevre', 'En büyük 3 ölçümün ortalaması - çevre'),
    'area_worst': ('En Kötü Alan', 'En büyük 3 ölçümün ortalaması - alan'),
    'smoothness_worst': ('En Kötü Pürüzsüzlük', 'En büyük 3 ölçümün ortalaması - pürüzsüzlük'),
    'compactness_worst': ('En Kötü Kompaktlık', 'En büyük 3 ölçümün ortalaması - kompaktlık'),
    'concavity_worst': ('En Kötü Çukurluk', 'En büyük 3 ölçümün ortalaması - çukurluk'),
    'concave points_worst': ('En Kötü Çukur Nokta', 'En büyük 3 ölçümün ortalaması - çukur nokta'),
    'symmetry_worst': ('En Kötü Simetri', 'En büyük 3 ölçümün ortalaması - simetri'),
    'fractal_dimension_worst': ('En Kötü Fraktal Boyut', 'En büyük 3 ölçümün ortalaması - fraktal boyut'),
}

# Tipik değer aralıkları (Wisconsin veri setinden)
FEATURE_RANGES = {
    'radius_mean': (6.98, 28.11, 14.13),
    'texture_mean': (9.71, 39.28, 19.29),
    'perimeter_mean': (43.79, 188.5, 91.97),
    'area_mean': (143.5, 2501.0, 654.89),
    'smoothness_mean': (0.05, 0.16, 0.096),
    'compactness_mean': (0.02, 0.35, 0.104),
    'concavity_mean': (0.0, 0.43, 0.089),
    'concave points_mean': (0.0, 0.20, 0.049),
    'symmetry_mean': (0.11, 0.30, 0.181),
    'fractal_dimension_mean': (0.05, 0.097, 0.063),
    'radius_se': (0.11, 2.87, 0.405),
    'texture_se': (0.36, 4.88, 1.217),
    'perimeter_se': (0.76, 21.98, 2.866),
    'area_se': (6.8, 542.2, 40.34),
    'smoothness_se': (0.002, 0.031, 0.007),
    'compactness_se': (0.002, 0.135, 0.025),
    'concavity_se': (0.0, 0.396, 0.032),
    'concave points_se': (0.0, 0.053, 0.012),
    'symmetry_se': (0.008, 0.079, 0.021),
    'fractal_dimension_se': (0.001, 0.03, 0.004),
    'radius_worst': (7.93, 36.04, 16.27),
    'texture_worst': (12.02, 49.54, 25.68),
    'perimeter_worst': (50.41, 251.2, 107.26),
    'area_worst': (185.2, 4254.0, 880.58),
    'smoothness_worst': (0.07, 0.22, 0.132),
    'compactness_worst': (0.027, 1.058, 0.254),
    'concavity_worst': (0.0, 1.252, 0.272),
    'concave points_worst': (0.0, 0.291, 0.115),
    'symmetry_worst': (0.16, 0.664, 0.290),
    'fractal_dimension_worst': (0.055, 0.208, 0.084),
}

# ─── Sidebar ─────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style='text-align:center; padding: 20px 0 10px 0;'>
        <div style='font-size:2.5rem;'>🔬</div>
        <div style='font-size:1.1rem; font-weight:700; color:#ccd6f6; margin-top:8px;'>
            Meme Kanseri<br>Tahmin Sistemi
        </div>
        <div style='font-size:0.75rem; color:#8892b0; margin-top:4px;'>
            Wisconsin Breast Cancer Dataset
        </div>
    </div>
    <hr style='border-color:#2d3561; margin:16px 0;'>
    """, unsafe_allow_html=True)

    st.markdown("**📊 Model Bilgisi**")
    st.markdown(f"""
    <div class='metric-card'>
        <div class='metric-value'>{model_info['accuracy']*100:.1f}%</div>
        <div class='metric-label'>Doğruluk (Accuracy)</div>
    </div>
    <div class='metric-card'>
        <div class='metric-value'>{model_info['f1_score']*100:.1f}%</div>
        <div class='metric-label'>F1-Score</div>
    </div>
    <div class='metric-card'>
        <div class='metric-value'>{model_info['roc_auc']*100:.1f}%</div>
        <div class='metric-label'>ROC-AUC</div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown(f"""
    <hr style='border-color:#2d3561; margin:16px 0;'>
    <div style='font-size:0.8rem; color:#8892b0;'>
        <b style='color:#ccd6f6;'>Model:</b> {model_info['model_name']}<br>
        <b style='color:#ccd6f6;'>Eğitim:</b> {model_info.get('created_date', '2026-06-26')}<br>
        <b style='color:#ccd6f6;'>Özellik:</b> {len(FEATURE_NAMES)} girdi değişkeni
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <hr style='border-color:#2d3561; margin:16px 0;'>
    <div class='disclaimer'>
        ⚠️ <b>Uyarı:</b> Bu uygulama yalnızca akademik/araştırma amaçlıdır.
        Tıbbi teşhis için kullanılamaz.
    </div>
    """, unsafe_allow_html=True)

# ─── Ana Başlık ───────────────────────────────────────────────────────────────
st.markdown("""
<div style='text-align:center; padding:30px 0 10px 0;'>
    <h1 style='font-size:2.2rem; font-weight:700; color:#ccd6f6; margin:0;'>
        🔬 Meme Kanseri Tahmin & Açıklanabilirlik Sistemi
    </h1>
    <p style='color:#8892b0; margin-top:8px; font-size:1rem;'>
        Makine öğrenmesi modelleri ile tümör karakterizasyonu · SHAP açıklanabilirliği
    </p>
</div>
<hr style='border-color:#2d3561; margin:20px 0;'>
""", unsafe_allow_html=True)

# ─── Sekmeler ─────────────────────────────────────────────────────────────────
tab1, tab2, tab3, tab4 = st.tabs([
    "🎯 Tahmin Yap",
    "📊 Model Analizi",
    "🔍 Veri Keşfi (EDA)",
    "📖 Proje Hakkında"
])

# ═══════════════════════════════════════════════════════════════════════════════
# TAB 1 — TAHMİN YAP
# ═══════════════════════════════════════════════════════════════════════════════
with tab1:
    st.markdown("""
    <div class='info-banner'>
        💡 Aşağıdaki değerleri hasta biyopsi raporuna göre doldurun. Her özellik için kaydırıcıyı kullanabilir
        veya değeri doğrudan girebilirsiniz. Değerler Wisconsin veri setinin tipik aralığına göre ayarlanmıştır.
    </div>
    """, unsafe_allow_html=True)

    # Özellik grupları
    groups = {
        "🔵 Ortalama Değerler (Mean)": [f for f in FEATURE_NAMES if f.endswith('_mean') or f == 'concave points_mean'],
        "🟡 Standart Hata (SE)": [f for f in FEATURE_NAMES if f.endswith('_se') or f == 'concave points_se'],
        "🔴 En Kötü Değerler (Worst)": [f for f in FEATURE_NAMES if f.endswith('_worst') or f == 'concave points_worst'],
    }

    input_values = {}

    for group_name, features in groups.items():
        st.markdown(f"<div class='section-header'>{group_name}</div>", unsafe_allow_html=True)
        cols = st.columns(3)
        for idx, feat in enumerate(features):
            col = cols[idx % 3]
            with col:
                mn, mx, default = FEATURE_RANGES.get(feat, (0.0, 1.0, 0.5))
                label, tooltip = FEATURE_DESCRIPTIONS.get(feat, (feat, ""))
                val = st.number_input(
                    label,
                    min_value=float(mn * 0.5),
                    max_value=float(mx * 2.0),
                    value=float(default),
                    format="%.4f",
                    help=tooltip,
                    key=f"input_{feat}"
                )
                input_values[feat] = val

    st.markdown("<br>", unsafe_allow_html=True)

    col_btn1, col_btn2, col_btn3 = st.columns([1, 2, 1])
    with col_btn2:
        predict_btn = st.button("🔬 Tahmin Et", use_container_width=True, type="primary")
        reset_btn = st.button("🔄 Değerleri Sıfırla", use_container_width=True)

    if predict_btn:
        if pipeline is None:
            st.error("❌ Model dosyası bulunamadı! `breast_cancer_pipeline.pkl` dosyasının app.py ile aynı klasörde olduğundan emin olun.")
        else:
            input_df = pd.DataFrame([input_values])

            with st.spinner("Model tahmin yapıyor..."):
                prediction = pipeline.predict(input_df)[0]
                probability = pipeline.predict_proba(input_df)[0]

            benign_prob = probability[0] * 100
            malignant_prob = probability[1] * 100

            st.markdown("<br>", unsafe_allow_html=True)
            st.markdown("<div class='section-header'>📋 Tahmin Sonucu</div>", unsafe_allow_html=True)

            col_res, col_gauge = st.columns([1, 1])

            with col_res:
                if prediction == 0:
                    st.markdown(f"""
                    <div class='result-benign'>
                        <div class='result-title' style='color:#00b894;'>✅ İyi Huylu (Benign)</div>
                        <div style='font-size:2.5rem; font-weight:800; color:#00b894; margin:12px 0;'>
                            {benign_prob:.1f}%
                        </div>
                        <div class='result-subtitle'>İyi huylu tümör olasılığı</div>
                    </div>
                    """, unsafe_allow_html=True)
                else:
                    st.markdown(f"""
                    <div class='result-malignant'>
                        <div class='result-title' style='color:#d63031;'>⚠️ Kötü Huylu (Malignant)</div>
                        <div style='font-size:2.5rem; font-weight:800; color:#d63031; margin:12px 0;'>
                            {malignant_prob:.1f}%
                        </div>
                        <div class='result-subtitle'>Malignant tümör olasılığı</div>
                    </div>
                    """, unsafe_allow_html=True)

            with col_gauge:
                fig_gauge = go.Figure(go.Indicator(
                    mode="gauge+number",
                    value=malignant_prob,
                    title={"text": "Malignant Olasılığı (%)", "font": {"color": "#ccd6f6", "size": 14}},
                    number={"font": {"color": "#ccd6f6", "size": 28}, "suffix": "%"},
                    gauge={
                        "axis": {"range": [0, 100], "tickfont": {"color": "#8892b0"}},
                        "bar": {"color": "#d63031" if prediction == 1 else "#00b894"},
                        "bgcolor": "#1a1d2e",
                        "bordercolor": "#2d3561",
                        "steps": [
                            {"range": [0, 30], "color": "#0d3b2e"},
                            {"range": [30, 70], "color": "#2d2a1e"},
                            {"range": [70, 100], "color": "#3b0d0d"},
                        ],
                        "threshold": {
                            "line": {"color": "white", "width": 2},
                            "thickness": 0.75,
                            "value": 50
                        }
                    }
                ))
                fig_gauge.update_layout(
                    paper_bgcolor="rgba(0,0,0,0)",
                    plot_bgcolor="rgba(0,0,0,0)",
                    height=220,
                    margin=dict(t=40, b=0, l=20, r=20),
                    font={"color": "#ccd6f6"}
                )
                st.plotly_chart(fig_gauge, use_container_width=True)

            # Olasılık çubuğu
            st.markdown("<br>", unsafe_allow_html=True)
            fig_bar = go.Figure(data=[
                go.Bar(name='İyi Huylu (Benign)', x=['Olasılık'], y=[benign_prob],
                       marker_color='#00b894', text=[f'{benign_prob:.1f}%'], textposition='inside',
                       textfont=dict(size=16, color='white')),
                go.Bar(name='Kötü Huylu (Malignant)', x=['Olasılık'], y=[malignant_prob],
                       marker_color='#d63031', text=[f'{malignant_prob:.1f}%'], textposition='inside',
                       textfont=dict(size=16, color='white')),
            ])
            fig_bar.update_layout(
                barmode='stack',
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                height=100,
                margin=dict(t=10, b=10, l=0, r=0),
                showlegend=True,
                legend=dict(font=dict(color='#ccd6f6'), bgcolor='rgba(0,0,0,0)'),
                xaxis=dict(visible=False),
                yaxis=dict(visible=False),
            )
            st.plotly_chart(fig_bar, use_container_width=True)

            # Girilen değerlerin özeti
            with st.expander("📋 Girilen Değerlerin Özeti", expanded=False):
                summary_df = pd.DataFrame({
                    "Özellik": [FEATURE_DESCRIPTIONS.get(f, (f,))[0] for f in FEATURE_NAMES],
                    "Değer": [f"{input_values[f]:.4f}" for f in FEATURE_NAMES],
                })
                st.dataframe(summary_df, use_container_width=True, hide_index=True)


# ═══════════════════════════════════════════════════════════════════════════════
# TAB 2 — MODEL ANALİZİ
# ═══════════════════════════════════════════════════════════════════════════════
with tab2:
    st.markdown("<div class='section-header'>🏆 Model Performans Karşılaştırması</div>", unsafe_allow_html=True)

    model_data = {
        "Model": ["Decision Tree", "Naive Bayes", "KNN", "LightGBM", "Random Forest",
                  "XGBoost", "Gradient Boosting", "SVM", "Logistic Regression"],
        "Accuracy": [0.904, 0.921, 0.939, 0.947, 0.947, 0.956, 0.956, 0.956, 0.974],
        "Precision": [0.900, 0.872, 0.961, 1.000, 0.951, 0.974, 0.951, 0.974, 0.976],
        "Recall": [0.857, 0.952, 0.881, 0.881, 0.929, 0.929, 0.952, 0.929, 0.952],
        "F1-Score": [0.878, 0.910, 0.919, 0.937, 0.940, 0.951, 0.951, 0.951, 0.964],
        "ROC-AUC": [0.887, 0.993, 0.963, 0.994, 0.993, 0.990, 0.994, 0.997, 0.986],
        "CV_Mean": [0.900, 0.927, 0.935, 0.945, 0.944, 0.944, 0.949, 0.962, 0.969],
    }
    df_models = pd.DataFrame(model_data).sort_values("F1-Score", ascending=False)

    # Renk skalası
    colors = ['#667eea' if m == 'Logistic Regression' else '#4a5568' for m in df_models['Model']]

    col_m1, col_m2 = st.columns(2)
    with col_m1:
        fig_acc = go.Figure(go.Bar(
            x=df_models['Accuracy'], y=df_models['Model'],
            orientation='h', marker_color=colors,
            text=[f"{v:.3f}" for v in df_models['Accuracy']], textposition='outside',
            textfont=dict(color='#ccd6f6', size=11)
        ))
        fig_acc.add_vline(x=0.974, line_dash="dash", line_color="#d63031", line_width=1.5)
        fig_acc.update_layout(
            title="Accuracy Karşılaştırması", title_font_color='#ccd6f6',
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(26,29,46,0.5)",
            xaxis=dict(range=[0.85, 1.0], tickfont=dict(color='#8892b0'), gridcolor='#2d3561'),
            yaxis=dict(tickfont=dict(color='#ccd6f6')),
            height=320, margin=dict(t=40, b=10, l=10, r=60),
        )
        st.plotly_chart(fig_acc, use_container_width=True)

    with col_m2:
        fig_roc = go.Figure(go.Bar(
            x=df_models['ROC-AUC'], y=df_models['Model'],
            orientation='h', marker_color=['#764ba2' if m == 'Logistic Regression' else '#4a5568' for m in df_models['Model']],
            text=[f"{v:.3f}" for v in df_models['ROC-AUC']], textposition='outside',
            textfont=dict(color='#ccd6f6', size=11)
        ))
        fig_roc.update_layout(
            title="ROC-AUC Karşılaştırması", title_font_color='#ccd6f6',
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(26,29,46,0.5)",
            xaxis=dict(range=[0.85, 1.01], tickfont=dict(color='#8892b0'), gridcolor='#2d3561'),
            yaxis=dict(tickfont=dict(color='#ccd6f6')),
            height=320, margin=dict(t=40, b=10, l=10, r=60),
        )
        st.plotly_chart(fig_roc, use_container_width=True)

    # Tablo
    st.markdown("<div class='section-header'>📋 Detaylı Metrik Tablosu</div>", unsafe_allow_html=True)
    styled_df = df_models[["Model","Accuracy","Precision","Recall","F1-Score","ROC-AUC","CV_Mean"]].copy()
    for col in ["Accuracy","Precision","Recall","F1-Score","ROC-AUC","CV_Mean"]:
        styled_df[col] = styled_df[col].apply(lambda x: f"{x:.4f}")
    st.dataframe(styled_df.reset_index(drop=True), use_container_width=True, hide_index=True)

    # Model görselleri
    st.markdown("<div class='section-header'>📈 Model Grafikleri</div>", unsafe_allow_html=True)
    col_g1, col_g2 = st.columns(2)
    with col_g1:
        if os.path.exists("graphs/model_comparison_eng.png"):
            st.image("graphs/model_comparison_eng.png", caption="Model Karşılaştırması", use_container_width=True)
    with col_g2:
        if os.path.exists("graphs/final_model_confusion_matrix.png"):
            st.image("graphs/final_model_confusion_matrix.png", caption="Confusion Matrix & ROC Eğrisi", use_container_width=True)

    # SHAP grafikler
    st.markdown("<div class='section-header'>🧠 SHAP Açıklanabilirlik Analizi</div>", unsafe_allow_html=True)
    col_s1, col_s2 = st.columns(2)
    with col_s1:
        if os.path.exists("graphs/shap_importance_eng.png"):
            st.image("graphs/shap_importance_eng.png", caption="SHAP Feature Importance", use_container_width=True)
    with col_s2:
        if os.path.exists("graphs/shap_summary_eng.png"):
            st.image("graphs/shap_summary_eng.png", caption="SHAP Summary Plot", use_container_width=True)

    st.markdown("""
    <div class='info-banner'>
        🧠 <b>SHAP Yorumu:</b> <code>texture_worst</code> ve <code>radius_se</code> modelin kararını en çok etkileyen özelliklerdir.
        Yüksek değerler (kırmızı noktalar) malignant yönünde, düşük değerler (mavi noktalar) benign yönünde etki eder.
    </div>
    """, unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════════════════
# TAB 3 — VERİ KEŞFİ (EDA)
# ═══════════════════════════════════════════════════════════════════════════════
with tab3:
    st.markdown("<div class='section-header'>📊 Hedef Değişken Dağılımı</div>", unsafe_allow_html=True)

    col_e1, col_e2 = st.columns([1, 2])
    with col_e1:
        labels = ['Benign (İyi Huylu)', 'Malignant (Kötü Huylu)']
        values = [357, 212]
        fig_pie = go.Figure(go.Pie(
            labels=labels, values=values,
            hole=0.5,
            marker_colors=['#00b894', '#d63031'],
            textinfo='label+percent',
            textfont=dict(color='white', size=12),
        ))
        fig_pie.update_layout(
            paper_bgcolor="rgba(0,0,0,0)",
            showlegend=False,
            height=280,
            margin=dict(t=10, b=10, l=10, r=10),
            annotations=[dict(text='569<br>örnek', x=0.5, y=0.5, font_size=14,
                               showarrow=False, font_color='#ccd6f6')]
        )
        st.plotly_chart(fig_pie, use_container_width=True)

        col_stat1, col_stat2 = st.columns(2)
        col_stat1.metric("Benign", "357", "62.7%")
        col_stat2.metric("Malignant", "212", "37.3%")

    with col_e2:
        if os.path.exists("graphs/diagnosis_distribution_eng.png"):
            st.image("graphs/diagnosis_distribution_eng.png", use_container_width=True)

    st.markdown("<div class='section-header'>🔗 Özellikler Arası Korelasyon Matrisi</div>", unsafe_allow_html=True)
    if os.path.exists("graphs/correlation_matrix_eng.png"):
        st.image("graphs/correlation_matrix_eng.png", caption="Korelasyon Matrisi", use_container_width=True)

    col_e3, col_e4 = st.columns(2)
    with col_e3:
        st.markdown("<div class='section-header'>🎯 Hedef ile Özellik Korelasyonu</div>", unsafe_allow_html=True)
        if os.path.exists("graphs/target_correlation_eng.png"):
            st.image("graphs/target_correlation_eng.png", use_container_width=True)
    with col_e4:
        st.markdown("<div class='section-header'>📦 Aykırı Değer Analizi (Boxplot)</div>", unsafe_allow_html=True)
        if os.path.exists("graphs/top10_outliers_boxplot_eng.png"):
            st.image("graphs/top10_outliers_boxplot_eng.png", use_container_width=True)

    # Özellik istatistikleri
    st.markdown("<div class='section-header'>📐 Özellik Aralıkları</div>", unsafe_allow_html=True)
    stats_data = []
    for feat in FEATURE_NAMES:
        mn, mx, default = FEATURE_RANGES.get(feat, (0, 1, 0.5))
        label, _ = FEATURE_DESCRIPTIONS.get(feat, (feat, ""))
        stats_data.append({"Özellik": label, "Ham Adı": feat, "Min": mn, "Max": mx, "Ortalama": default})
    df_stats = pd.DataFrame(stats_data)
    for col in ["Min", "Max", "Ortalama"]:
        df_stats[col] = df_stats[col].apply(lambda x: f"{x:.4f}")
    st.dataframe(df_stats, use_container_width=True, hide_index=True)


# ═══════════════════════════════════════════════════════════════════════════════
# TAB 4 — PROJE HAKKINDA
# ═══════════════════════════════════════════════════════════════════════════════
with tab4:
    st.markdown("""
    <div class='section-header'>📖 Proje Özeti</div>
    """, unsafe_allow_html=True)

    col_a1, col_a2 = st.columns(2)
    with col_a1:
        st.markdown("""
        <div style='background:#1a1d2e; border:1px solid #2d3561; border-radius:12px; padding:20px; margin-bottom:16px;'>
            <h4 style='color:#ccd6f6; margin:0 0 12px 0;'>🎯 Proje Amacı</h4>
            <p style='color:#8892b0; font-size:0.9rem; line-height:1.7;'>
            Bu proje, Wisconsin Breast Cancer veri seti kullanılarak meme kanseri tümörlerini
            <b style='color:#ccd6f6;'>iyi huylu (benign)</b> ve <b style='color:#ccd6f6;'>kötü huylu (malignant)</b>
            olarak sınıflandırmayı amaçlamaktadır. Çeşitli makine öğrenmesi yöntemleri karşılaştırılmış,
            SHAP ile model açıklanabilirliği sağlanmıştır.
            </p>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("""
        <div style='background:#1a1d2e; border:1px solid #2d3561; border-radius:12px; padding:20px;'>
            <h4 style='color:#ccd6f6; margin:0 0 12px 0;'>📚 Veri Seti</h4>
            <ul style='color:#8892b0; font-size:0.9rem; line-height:2; margin:0; padding-left:18px;'>
                <li><b style='color:#ccd6f6;'>Kaynak:</b> UCI Machine Learning Repository</li>
                <li><b style='color:#ccd6f6;'>Örnek Sayısı:</b> 569</li>
                <li><b style='color:#ccd6f6;'>Özellik Sayısı:</b> 30 (+ 3 türetilmiş)</li>
                <li><b style='color:#ccd6f6;'>Benign:</b> 357 (%62.7)</li>
                <li><b style='color:#ccd6f6;'>Malignant:</b> 212 (%37.3)</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)

    with col_a2:
        st.markdown("""
        <div style='background:#1a1d2e; border:1px solid #2d3561; border-radius:12px; padding:20px; margin-bottom:16px;'>
            <h4 style='color:#ccd6f6; margin:0 0 12px 0;'>⚙️ Kullanılan Yöntemler</h4>
            <ul style='color:#8892b0; font-size:0.9rem; line-height:2; margin:0; padding-left:18px;'>
                <li>Logistic Regression <span style='color:#00b894;'>★ Final Model</span></li>
                <li>Random Forest</li>
                <li>XGBoost & LightGBM</li>
                <li>Gradient Boosting</li>
                <li>SVM (RBF Kernel)</li>
                <li>KNN, Naive Bayes, Decision Tree</li>
                <li>Ensemble (VotingClassifier)</li>
                <li>SHAP Açıklanabilirlik Analizi</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("""
        <div style='background:#1a1d2e; border:1px solid #2d3561; border-radius:12px; padding:20px;'>
            <h4 style='color:#ccd6f6; margin:0 0 12px 0;'>🛠️ Teknik Altyapı</h4>
            <ul style='color:#8892b0; font-size:0.9rem; line-height:2; margin:0; padding-left:18px;'>
                <li>Python 3.x · Scikit-learn · Pandas · NumPy</li>
                <li>XGBoost · LightGBM</li>
                <li>SHAP · Matplotlib · Seaborn</li>
                <li>Streamlit · Plotly</li>
                <li>Feature Engineering Pipeline</li>
                <li>GridSearchCV Hiperparametre Optimizasyonu</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<div class='section-header'>🔄 Pipeline Akışı</div>", unsafe_allow_html=True)
    st.markdown("""
    <div style='background:#1a1d2e; border:1px solid #2d3561; border-radius:12px; padding:24px;'>
        <div style='display:flex; align-items:center; justify-content:center; gap:8px; flex-wrap:wrap; text-align:center;'>
            <div style='background:#2d3561; border-radius:8px; padding:10px 16px; color:#ccd6f6; font-size:0.85rem;'>
                📂<br>Ham Veri
            </div>
            <div style='color:#667eea; font-size:1.5rem;'>→</div>
            <div style='background:#2d3561; border-radius:8px; padding:10px 16px; color:#ccd6f6; font-size:0.85rem;'>
                🔧<br>Feature Eng.
            </div>
            <div style='color:#667eea; font-size:1.5rem;'>→</div>
            <div style='background:#2d3561; border-radius:8px; padding:10px 16px; color:#ccd6f6; font-size:0.85rem;'>
                📏<br>StandardScaler
            </div>
            <div style='color:#667eea; font-size:1.5rem;'>→</div>
            <div style='background:#2d3561; border-radius:8px; padding:10px 16px; color:#ccd6f6; font-size:0.85rem;'>
                🤖<br>Logistic Reg.
            </div>
            <div style='color:#667eea; font-size:1.5rem;'>→</div>
            <div style='background:#0d3b2e; border:1px solid #00b894; border-radius:8px; padding:10px 16px; color:#00b894; font-size:0.85rem;'>
                ✅<br>Tahmin
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <br>
    <div style='text-align:center; color:#4a5568; font-size:0.8rem;'>
        Bu uygulama yalnızca akademik ve araştırma amaçlıdır. Tıbbi teşhis için kullanılamaz.
    </div>
    """, unsafe_allow_html=True)