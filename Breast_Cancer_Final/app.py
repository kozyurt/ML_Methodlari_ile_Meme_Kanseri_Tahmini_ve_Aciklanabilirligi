import streamlit as st
import pandas as pd
import numpy as np
import pickle
import os
import io
import plotly.graph_objects as go
import plotly.express as px
from sklearn.base import BaseEstimator, TransformerMixin

# ─── Sayfa Yapılandırması ─────────────────────────────────────────────────────
st.set_page_config(
    page_title="MedPredict · Meme Kanseri Tanı Sistemi",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ─── FeatureEngineeringTransformer (pickle için zorunlu) ──────────────────────
class FeatureEngineeringTransformer(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.feature_names_ = None
    def fit(self, X, y=None):
        if isinstance(X, pd.DataFrame):
            self.feature_names_ = X.columns.tolist()
        return self
    def transform(self, X):
        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X, columns=self.feature_names_) if self.feature_names_ else pd.DataFrame(X)
        X = X.copy()
        if 'radius_mean' in X.columns and 'area_mean' in X.columns:
            X['radius_area_ratio'] = X['radius_mean'] / (X['area_mean'] + 1e-6)
        if 'perimeter_mean' in X.columns and 'area_mean' in X.columns:
            X['perimeter_area_ratio'] = X['perimeter_mean'] / (X['area_mean'] + 1e-6)
        if 'concavity_mean' in X.columns and 'concave points_mean' in X.columns:
            X['concavity_points_product'] = X['concavity_mean'] * X['concave points_mean']
        return X

# ─── CSS ──────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&family=DM+Serif+Display&display=swap');

*, html, body, [class*="css"] {
    font-family: 'Inter', sans-serif !important;
}

/* Sayfa arka planı */
.main, .block-container {
    background-color: #F7F9FC !important;
    padding-top: 0 !important;
}

/* Sidebar gizle */
section[data-testid="stSidebar"] { display: none; }

/* Streamlit header gizle */
header[data-testid="stHeader"] { background: transparent; }

/* ── Hero banner ── */
.hero {
    background: linear-gradient(135deg, #0F4C81 0%, #1a6fb5 60%, #2196F3 100%);
    border-radius: 0 0 24px 24px;
    padding: 36px 48px 32px 48px;
    margin: -80px -80px 32px -80px;
    color: white;
}
.hero-badge {
    display: inline-block;
    background: rgba(255,255,255,0.15);
    border: 1px solid rgba(255,255,255,0.3);
    border-radius: 100px;
    padding: 4px 14px;
    font-size: 0.75rem;
    font-weight: 600;
    letter-spacing: 1.5px;
    text-transform: uppercase;
    margin-bottom: 12px;
}
.hero h1 {
    font-family: 'DM Serif Display', serif !important;
    font-size: 2.4rem;
    font-weight: 400;
    margin: 0 0 8px 0;
    line-height: 1.2;
}
.hero p {
    font-size: 1rem;
    opacity: 0.85;
    margin: 0;
    max-width: 600px;
}
.hero-stats {
    display: flex;
    gap: 32px;
    margin-top: 24px;
}
.hero-stat-val {
    font-size: 1.8rem;
    font-weight: 800;
    line-height: 1;
}
.hero-stat-lbl {
    font-size: 0.72rem;
    opacity: 0.7;
    letter-spacing: 1px;
    text-transform: uppercase;
    margin-top: 4px;
}

/* ── Nav tabs ── */
.nav-container {
    display: flex;
    gap: 4px;
    background: white;
    border-radius: 12px;
    padding: 6px;
    margin-bottom: 28px;
    box-shadow: 0 1px 4px rgba(0,0,0,0.06);
    border: 1px solid #E8EDF4;
}

/* ── Kart ── */
.card {
    background: white;
    border-radius: 14px;
    padding: 24px;
    border: 1px solid #E8EDF4;
    box-shadow: 0 1px 6px rgba(15,76,129,0.06);
    margin-bottom: 16px;
}
.card-title {
    font-size: 0.75rem;
    font-weight: 700;
    letter-spacing: 1.2px;
    text-transform: uppercase;
    color: #6B7A99;
    margin-bottom: 16px;
    display: flex;
    align-items: center;
    gap: 8px;
}

/* ── Metrik kutu ── */
.kpi-row { display: flex; gap: 12px; margin-bottom: 20px; }
.kpi {
    flex: 1;
    background: white;
    border-radius: 12px;
    padding: 18px 20px;
    border: 1px solid #E8EDF4;
    box-shadow: 0 1px 4px rgba(0,0,0,0.05);
    text-align: center;
}
.kpi-val {
    font-size: 1.7rem;
    font-weight: 800;
    color: #0F4C81;
    line-height: 1;
}
.kpi-lbl {
    font-size: 0.72rem;
    color: #8896A8;
    text-transform: uppercase;
    letter-spacing: 1px;
    margin-top: 6px;
}

/* ── Sonuç kutusu ── */
.result-box {
    border-radius: 16px;
    padding: 28px 32px;
    text-align: center;
    margin: 8px 0;
}
.result-benign {
    background: linear-gradient(135deg, #E8FBF3 0%, #D4F5E9 100%);
    border: 2px solid #00C875;
}
.result-malignant {
    background: linear-gradient(135deg, #FFF0EE 0%, #FFE0DC 100%);
    border: 2px solid #E03C31;
}
.result-icon { font-size: 2.8rem; line-height: 1; margin-bottom: 8px; }
.result-label {
    font-size: 1.5rem;
    font-weight: 800;
    margin-bottom: 4px;
}
.result-benign .result-label { color: #007A4D; }
.result-malignant .result-label { color: #B52A1D; }
.result-prob {
    font-size: 2.8rem;
    font-weight: 900;
    line-height: 1.1;
}
.result-benign .result-prob { color: #00C875; }
.result-malignant .result-prob { color: #E03C31; }
.result-sub { font-size: 0.85rem; color: #6B7A99; margin-top: 6px; }

/* ── Section başlık ── */
.sec-header {
    font-size: 1rem;
    font-weight: 700;
    color: #1A2B4A;
    margin: 28px 0 14px 0;
    padding-bottom: 8px;
    border-bottom: 2px solid #E8EDF4;
    display: flex;
    align-items: center;
    gap: 8px;
}

/* ── Özellik grup başlığı ── */
.feat-group {
    background: linear-gradient(90deg, #EEF4FB 0%, transparent 100%);
    border-left: 3px solid #0F4C81;
    border-radius: 0 8px 8px 0;
    padding: 8px 14px;
    font-size: 0.8rem;
    font-weight: 700;
    color: #0F4C81;
    letter-spacing: 0.5px;
    text-transform: uppercase;
    margin: 20px 0 12px 0;
}

/* ── Uyarı banner ── */
.warning-strip {
    background: #FFF8E6;
    border: 1px solid #F5C842;
    border-radius: 10px;
    padding: 10px 16px;
    font-size: 0.82rem;
    color: #7A5A00;
    margin-bottom: 20px;
    display: flex;
    align-items: center;
    gap: 10px;
}

/* ── CSV sonuç tablosu ── */
.batch-result-benign { background: #E8FBF3; color: #007A4D; font-weight: 600;
    padding: 3px 10px; border-radius: 20px; font-size: 0.8rem; }
.batch-result-malignant { background: #FFF0EE; color: #B52A1D; font-weight: 600;
    padding: 3px 10px; border-radius: 20px; font-size: 0.8rem; }

/* Streamlit widget düzenlemeleri */
.stNumberInput label { font-size: 0.78rem !important; font-weight: 600 !important; color: #3D5270 !important; }
.stSlider label { font-size: 0.78rem !important; font-weight: 600 !important; color: #3D5270 !important; }
.stButton > button {
    border-radius: 10px !important;
    font-weight: 600 !important;
    font-size: 0.9rem !important;
    transition: all 0.2s !important;
}
.stButton > button[kind="primary"] {
    background: linear-gradient(135deg, #0F4C81, #2196F3) !important;
    border: none !important;
    color: white !important;
    box-shadow: 0 4px 14px rgba(15,76,129,0.3) !important;
}
.stButton > button[kind="primary"]:hover {
    transform: translateY(-1px) !important;
    box-shadow: 0 6px 18px rgba(15,76,129,0.4) !important;
}
div[data-testid="stTabs"] > div > div[role="tablist"] {
    background: white;
    border-radius: 12px;
    padding: 6px;
    border: 1px solid #E8EDF4;
    gap: 4px;
}
div[data-testid="stTabs"] button[role="tab"] {
    border-radius: 8px !important;
    font-weight: 600 !important;
    font-size: 0.85rem !important;
    color: #6B7A99 !important;
    padding: 8px 18px !important;
}
div[data-testid="stTabs"] button[role="tab"][aria-selected="true"] {
    background: linear-gradient(135deg, #0F4C81, #2196F3) !important;
    color: white !important;
}
div[data-testid="metric-container"] {
    background: white !important;
    border: 1px solid #E8EDF4 !important;
    border-radius: 12px !important;
    padding: 16px !important;
}
</style>
""", unsafe_allow_html=True)

# ─── Sabitler ────────────────────────────────────────────────────────────────
FEATURE_NAMES = [
    'radius_mean','texture_mean','perimeter_mean','area_mean','smoothness_mean',
    'compactness_mean','concavity_mean','concave points_mean','symmetry_mean',
    'fractal_dimension_mean','radius_se','texture_se','perimeter_se','area_se',
    'smoothness_se','compactness_se','concavity_se','concave points_se',
    'symmetry_se','fractal_dimension_se','radius_worst','texture_worst',
    'perimeter_worst','area_worst','smoothness_worst','compactness_worst',
    'concavity_worst','concave points_worst','symmetry_worst','fractal_dimension_worst'
]
FEAT_LABELS = {
    'radius_mean':('Ort. Yarıçap','mm'),'texture_mean':('Ort. Doku',''),'perimeter_mean':('Ort. Çevre','mm'),
    'area_mean':('Ort. Alan','mm²'),'smoothness_mean':('Ort. Pürüzsüzlük',''),'compactness_mean':('Ort. Kompaktlık',''),
    'concavity_mean':('Ort. Çukurluk',''),'concave points_mean':('Ort. Çukur Nokta',''),'symmetry_mean':('Ort. Simetri',''),
    'fractal_dimension_mean':('Ort. Fraktal Boyut',''),'radius_se':('Yarıçap SE',''),'texture_se':('Doku SE',''),
    'perimeter_se':('Çevre SE',''),'area_se':('Alan SE',''),'smoothness_se':('Pürüzsüzlük SE',''),
    'compactness_se':('Kompaktlık SE',''),'concavity_se':('Çukurluk SE',''),'concave points_se':('Çukur Nokta SE',''),
    'symmetry_se':('Simetri SE',''),'fractal_dimension_se':('Fraktal Boyut SE',''),
    'radius_worst':('En Kötü Yarıçap','mm'),'texture_worst':('En Kötü Doku',''),'perimeter_worst':('En Kötü Çevre','mm'),
    'area_worst':('En Kötü Alan','mm²'),'smoothness_worst':('En Kötü Pürüzsüzlük',''),
    'compactness_worst':('En Kötü Kompaktlık',''),'concavity_worst':('En Kötü Çukurluk',''),
    'concave points_worst':('En Kötü Çukur Nokta',''),'symmetry_worst':('En Kötü Simetri',''),
    'fractal_dimension_worst':('En Kötü Fraktal Boyut',''),
}
FEAT_RANGES = {
    'radius_mean':(6.98,28.11,14.13),'texture_mean':(9.71,39.28,19.29),'perimeter_mean':(43.79,188.5,91.97),
    'area_mean':(143.5,2501.0,654.89),'smoothness_mean':(0.05,0.16,0.096),'compactness_mean':(0.02,0.35,0.104),
    'concavity_mean':(0.0,0.43,0.089),'concave points_mean':(0.0,0.20,0.049),'symmetry_mean':(0.11,0.30,0.181),
    'fractal_dimension_mean':(0.05,0.097,0.063),'radius_se':(0.11,2.87,0.405),'texture_se':(0.36,4.88,1.217),
    'perimeter_se':(0.76,21.98,2.866),'area_se':(6.8,542.2,40.34),'smoothness_se':(0.002,0.031,0.007),
    'compactness_se':(0.002,0.135,0.025),'concavity_se':(0.0,0.396,0.032),'concave points_se':(0.0,0.053,0.012),
    'symmetry_se':(0.008,0.079,0.021),'fractal_dimension_se':(0.001,0.03,0.004),
    'radius_worst':(7.93,36.04,16.27),'texture_worst':(12.02,49.54,25.68),'perimeter_worst':(50.41,251.2,107.26),
    'area_worst':(185.2,4254.0,880.58),'smoothness_worst':(0.07,0.22,0.132),'compactness_worst':(0.027,1.058,0.254),
    'concavity_worst':(0.0,1.252,0.272),'concave points_worst':(0.0,0.291,0.115),
    'symmetry_worst':(0.16,0.664,0.290),'fractal_dimension_worst':(0.055,0.208,0.084),
}
GROUPS = {
    "🔵 Ortalama Değerler": [f for f in FEATURE_NAMES if '_mean' in f],
    "🟡 Standart Hata (SE)": [f for f in FEATURE_NAMES if '_se' in f],
    "🔴 En Kötü Değerler": [f for f in FEATURE_NAMES if '_worst' in f],
}

# ─── Model yükleme ────────────────────────────────────────────────────────────
@st.cache_resource
def load_model():
    path = "breast_cancer_pipeline.pkl"
    if os.path.exists(path):
        with open(path, "rb") as f:
            return pickle.load(f)
    return None

@st.cache_data
def load_info():
    path = "model_info.pkl"
    if os.path.exists(path):
        with open(path, "rb") as f:
            return pickle.load(f)
    return {"model_name":"Logistic Regression","accuracy":0.9737,"f1_score":0.9639,"roc_auc":0.9861}

pipeline  = load_model()
model_info = load_info()

# ─── Hero Banner ─────────────────────────────────────────────────────────────
st.markdown(f"""
<div class="hero">
    <div class="hero-badge">🩺 MedPredict AI</div>
    <h1>Meme Kanseri Tanı Destek Sistemi</h1>
    <p>Wisconsin Breast Cancer veri seti ile eğitilmiş makine öğrenmesi modeli.
       Biyopsi ölçümlerini girerek anlık sınıflandırma yapın.</p>
    <div class="hero-stats">
        <div>
            <div class="hero-stat-val">{model_info['accuracy']*100:.1f}%</div>
            <div class="hero-stat-lbl">Doğruluk</div>
        </div>
        <div>
            <div class="hero-stat-val">{model_info['f1_score']*100:.1f}%</div>
            <div class="hero-stat-lbl">F1-Score</div>
        </div>
        <div>
            <div class="hero-stat-val">{model_info['roc_auc']*100:.1f}%</div>
            <div class="hero-stat-lbl">ROC-AUC</div>
        </div>
        <div>
            <div class="hero-stat-val">569</div>
            <div class="hero-stat-lbl">Eğitim Örneği</div>
        </div>
    </div>
</div>
""", unsafe_allow_html=True)

st.markdown("""
<div class="warning-strip">
    ⚠️ <b>Tıbbi Uyarı:</b> Bu sistem yalnızca akademik/araştırma amaçlıdır.
    Klinik teşhis kararlarında kullanılamaz, bir uzman doktora danışın.
</div>
""", unsafe_allow_html=True)

# ─── Ana Sekmeler ─────────────────────────────────────────────────────────────
tab1, tab2, tab3, tab4 = st.tabs([
    "🎯  Tekil Tahmin",
    "📂  Toplu Analiz (CSV)",
    "📊  Model & Sonuçlar",
    "🔬  Veri Keşfi"
])

# ══════════════════════════════════════════════════════════════════════════════
#  TAB 1 — TEKİL TAHMİN
# ══════════════════════════════════════════════════════════════════════════════
with tab1:
    left, right = st.columns([3, 2], gap="large")

    with left:
        st.markdown("<div class='sec-header'>📋 Hasta Biyopsi Değerleri</div>", unsafe_allow_html=True)
        input_values = {}
        for group_name, features in GROUPS.items():
            st.markdown(f"<div class='feat-group'>{group_name}</div>", unsafe_allow_html=True)
            cols = st.columns(3)
            for i, feat in enumerate(features):
                label, unit = FEAT_LABELS.get(feat, (feat, ""))
                mn, mx, default = FEAT_RANGES.get(feat, (0.0, 1.0, 0.5))
                display = f"{label}" + (f" ({unit})" if unit else "")
                with cols[i % 3]:
                    input_values[feat] = st.number_input(
                        display,
                        min_value=float(mn * 0.4),
                        max_value=float(mx * 2.5),
                        value=float(default),
                        format="%.4f",
                        key=f"t1_{feat}"
                    )

        col_b1, col_b2 = st.columns([2, 1])
        with col_b1:
            predict_btn = st.button("🔬 Tahmin Et", use_container_width=True, type="primary")
        with col_b2:
            reset_btn = st.button("↺ Sıfırla", use_container_width=True)

    with right:
        st.markdown("<div class='sec-header'>📈 Sonuç Paneli</div>", unsafe_allow_html=True)

        if "prediction_result" not in st.session_state:
            st.markdown("""
            <div class="card" style="text-align:center; padding:48px 24px; color:#9BAABB;">
                <div style="font-size:3rem;">🔬</div>
                <div style="font-size:1rem; font-weight:600; margin-top:12px; color:#3D5270;">
                    Tahmin bekleniyor
                </div>
                <div style="font-size:0.82rem; margin-top:8px;">
                    Sol taraftaki değerleri doldurup<br>"Tahmin Et" butonuna basın.
                </div>
            </div>
            """, unsafe_allow_html=True)
        else:
            res = st.session_state["prediction_result"]
            pred      = res["pred"]
            ben_pct   = res["ben_pct"]
            mal_pct   = res["mal_pct"]

            if pred == 0:
                st.markdown(f"""
                <div class="result-box result-benign">
                    <div class="result-icon">✅</div>
                    <div class="result-label">İyi Huylu (Benign)</div>
                    <div class="result-prob">{ben_pct:.1f}%</div>
                    <div class="result-sub">Benign olasılığı</div>
                </div>""", unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div class="result-box result-malignant">
                    <div class="result-icon">⚠️</div>
                    <div class="result-label">Kötü Huylu (Malignant)</div>
                    <div class="result-prob">{mal_pct:.1f}%</div>
                    <div class="result-sub">Malignant olasılığı</div>
                </div>""", unsafe_allow_html=True)

            # Olasılık çubuğu
            fig_prob = go.Figure()
            fig_prob.add_trace(go.Bar(
                x=[ben_pct], y=[""], orientation='h',
                marker_color='#00C875', name='Benign',
                text=f"Benign {ben_pct:.1f}%", textposition='inside',
                textfont=dict(color='white', size=13, family='Inter'),
            ))
            fig_prob.add_trace(go.Bar(
                x=[mal_pct], y=[""], orientation='h',
                marker_color='#E03C31', name='Malignant',
                text=f"Malignant {mal_pct:.1f}%", textposition='inside',
                textfont=dict(color='white', size=13, family='Inter'),
            ))
            fig_prob.update_layout(
                barmode='stack', height=72,
                margin=dict(t=8, b=8, l=0, r=0),
                paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
                showlegend=False, xaxis=dict(visible=False), yaxis=dict(visible=False),
            )
            st.plotly_chart(fig_prob, use_container_width=True)

            # Güven göstergesi
            confidence = max(ben_pct, mal_pct)
            conf_label = "Çok Yüksek" if confidence > 90 else "Yüksek" if confidence > 75 else "Orta" if confidence > 60 else "Düşük"
            conf_color = "#007A4D" if confidence > 90 else "#0F4C81" if confidence > 75 else "#E67E22" if confidence > 60 else "#E03C31"

            st.markdown(f"""
            <div class="card" style="margin-top:8px;">
                <div class="card-title">🎯 Model Güveni</div>
                <div style="display:flex; justify-content:space-between; align-items:center;">
                    <div style="font-size:1.6rem; font-weight:800; color:{conf_color};">{confidence:.1f}%</div>
                    <div style="background:{conf_color}22; color:{conf_color}; font-size:0.78rem;
                                font-weight:700; padding:4px 12px; border-radius:20px;">{conf_label}</div>
                </div>
            </div>
            """, unsafe_allow_html=True)

            # PDF rapor butonu
            st.markdown("<div class='sec-header'>📄 Rapor</div>", unsafe_allow_html=True)
            report_lines = [
                "MedPredict AI — Hasta Tahmin Raporu",
                "=" * 45,
                f"Model          : {model_info['model_name']}",
                f"Model Doğruluğu: {model_info['accuracy']*100:.2f}%",
                "",
                "─── TAHMİN SONUCU ───────────────────────",
                f"Sonuç          : {'İyi Huylu (Benign)' if pred==0 else 'Kötü Huylu (Malignant)'}",
                f"Benign Olasılığı   : {ben_pct:.2f}%",
                f"Malignant Olasılığı: {mal_pct:.2f}%",
                f"Model Güveni   : {conf_label} ({confidence:.1f}%)",
                "",
                "─── GİRİLEN DEĞERLERİN ÖZETİ ────────────",
            ]
            for feat in FEATURE_NAMES:
                lbl, _ = FEAT_LABELS.get(feat, (feat, ""))
                report_lines.append(f"  {lbl:<30}: {input_values.get(feat, 0):.4f}")
            report_lines += [
                "",
                "─── UYARI ────────────────────────────────",
                "Bu rapor yalnızca akademik/araştırma amaçlıdır.",
                "Tıbbi teşhis için kullanılamaz.",
            ]
            report_text = "\n".join(report_lines)
            st.download_button(
                label="⬇️ TXT Raporu İndir",
                data=report_text.encode("utf-8"),
                file_name="medpredict_rapor.txt",
                mime="text/plain",
                use_container_width=True,
            )

        # Tahmin işlemi
        if predict_btn:
            if pipeline is None:
                st.error("❌ `breast_cancer_pipeline.pkl` bulunamadı!")
            else:
                df_in = pd.DataFrame([input_values])
                pred   = pipeline.predict(df_in)[0]
                proba  = pipeline.predict_proba(df_in)[0]
                st.session_state["prediction_result"] = {
                    "pred": int(pred),
                    "ben_pct": float(proba[0] * 100),
                    "mal_pct": float(proba[1] * 100),
                }
                st.rerun()

        if reset_btn:
            if "prediction_result" in st.session_state:
                del st.session_state["prediction_result"]
            st.rerun()

# ══════════════════════════════════════════════════════════════════════════════
#  TAB 2 — TOPLU ANALİZ (CSV)
# ══════════════════════════════════════════════════════════════════════════════
with tab2:
    st.markdown("<div class='sec-header'>📂 CSV ile Toplu Tahmin</div>", unsafe_allow_html=True)

    st.markdown("""
    <div class="card">
        <div class="card-title">📌 Kullanım Talimatları</div>
        CSV dosyanız şu 30 sütunu içermelidir (sıra önemli değil):<br><br>
        <code style="font-size:0.78rem; background:#F0F4F9; padding:8px 12px; border-radius:8px; display:block; line-height:2;">
        radius_mean · texture_mean · perimeter_mean · area_mean · smoothness_mean · compactness_mean · concavity_mean ·
        concave points_mean · symmetry_mean · fractal_dimension_mean · radius_se · texture_se · perimeter_se · area_se ·
        smoothness_se · compactness_se · concavity_se · concave points_se · symmetry_se · fractal_dimension_se ·
        radius_worst · texture_worst · perimeter_worst · area_worst · smoothness_worst · compactness_worst ·
        concavity_worst · concave points_worst · symmetry_worst · fractal_dimension_worst
        </code>
    </div>
    """, unsafe_allow_html=True)

    # Örnek CSV indirme
    sample_data = {}
    for f in FEATURE_NAMES:
        mn, mx, default = FEAT_RANGES[f]
        sample_data[f] = [round(default + np.random.uniform(-0.1, 0.1) * default, 4) for _ in range(5)]
    sample_df = pd.DataFrame(sample_data)
    st.download_button(
        "⬇️ Örnek CSV Şablonu İndir",
        data=sample_df.to_csv(index=False).encode("utf-8"),
        file_name="ornek_veri.csv",
        mime="text/csv"
    )

    uploaded = st.file_uploader("CSV dosyası yükleyin", type=["csv"], label_visibility="collapsed")

    if uploaded and pipeline:
        try:
            df_csv = pd.read_csv(uploaded)
            missing = [c for c in FEATURE_NAMES if c not in df_csv.columns]

            if missing:
                st.error(f"❌ Eksik sütunlar: {', '.join(missing)}")
            else:
                df_feat = df_csv[FEATURE_NAMES].copy()
                preds  = pipeline.predict(df_feat)
                probas = pipeline.predict_proba(df_feat)

                df_result = df_csv.copy()
                df_result["Tahmin"]              = ["Benign" if p == 0 else "Malignant" for p in preds]
                df_result["Benign Olasılığı %"]  = (probas[:, 0] * 100).round(2)
                df_result["Malignant Olasılığı %"] = (probas[:, 1] * 100).round(2)
                df_result["Güven %"]             = (np.max(probas, axis=1) * 100).round(2)

                # Özet metrikler
                n_total   = len(preds)
                n_benign  = (preds == 0).sum()
                n_malig   = (preds == 1).sum()
                avg_conf  = (np.max(probas, axis=1) * 100).mean()

                c1, c2, c3, c4 = st.columns(4)
                c1.metric("Toplam Örnek", n_total)
                c2.metric("İyi Huylu", n_benign, f"{n_benign/n_total*100:.1f}%")
                c3.metric("Kötü Huylu", n_malig, f"{n_malig/n_total*100:.1f}%")
                c4.metric("Ort. Güven", f"{avg_conf:.1f}%")

                # Dağılım grafiği
                fig_dist = go.Figure(data=[
                    go.Bar(
                        x=["Benign (İyi Huylu)", "Malignant (Kötü Huylu)"],
                        y=[n_benign, n_malig],
                        marker_color=["#00C875", "#E03C31"],
                        text=[n_benign, n_malig], textposition="outside",
                        textfont=dict(size=16, color=["#007A4D", "#B52A1D"], family="Inter"),
                        width=0.4,
                    )
                ])
                fig_dist.update_layout(
                    paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                    height=260, margin=dict(t=20, b=10, l=10, r=10),
                    xaxis=dict(tickfont=dict(color="#3D5270", size=13)),
                    yaxis=dict(gridcolor="#E8EDF4", tickfont=dict(color="#9BAABB")),
                    showlegend=False,
                )
                st.plotly_chart(fig_dist, use_container_width=True)

                # Sonuç tablosu
                st.markdown("<div class='sec-header'>📋 Tahmin Sonuçları</div>", unsafe_allow_html=True)
                display_cols = ["Tahmin", "Benign Olasılığı %", "Malignant Olasılığı %", "Güven %"]
                if "id" in df_result.columns:
                    display_cols = ["id"] + display_cols
                st.dataframe(
                    df_result[display_cols].style.applymap(
                        lambda v: "background-color:#E8FBF3; color:#007A4D; font-weight:600" if v == "Benign"
                        else ("background-color:#FFF0EE; color:#B52A1D; font-weight:600" if v == "Malignant" else ""),
                        subset=["Tahmin"]
                    ),
                    use_container_width=True, height=400
                )

                st.download_button(
                    "⬇️ Sonuçları CSV Olarak İndir",
                    data=df_result.to_csv(index=False).encode("utf-8"),
                    file_name="medpredict_sonuclar.csv",
                    mime="text/csv",
                    type="primary"
                )

        except Exception as e:
            st.error(f"❌ Hata: {e}")

    elif uploaded and pipeline is None:
        st.error("❌ Model dosyası (`breast_cancer_pipeline.pkl`) bulunamadı.")


# ══════════════════════════════════════════════════════════════════════════════
#  TAB 3 — MODEL & SONUÇLAR
# ══════════════════════════════════════════════════════════════════════════════
with tab3:
    # KPI'lar
    st.markdown("<div class='sec-header'>🏅 Final Model Performansı</div>", unsafe_allow_html=True)
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Accuracy",  f"{model_info['accuracy']*100:.2f}%")
    c2.metric("F1-Score",  f"{model_info['f1_score']*100:.2f}%")
    c3.metric("ROC-AUC",   f"{model_info['roc_auc']*100:.2f}%")
    c4.metric("Model",     model_info['model_name'])

    # Model karşılaştırma tablosu
    st.markdown("<div class='sec-header'>📊 9 Model Karşılaştırması</div>", unsafe_allow_html=True)
    model_data = {
        "Model":["Logistic Regression","SVM","Gradient Boosting","XGBoost","Random Forest",
                 "LightGBM","KNN","Naive Bayes","Decision Tree"],
        "Accuracy":[0.9737,0.9561,0.9561,0.9561,0.9474,0.9474,0.9386,0.9211,0.9035],
        "Precision":[0.9756,0.9744,0.9512,0.9744,0.9512,1.000,0.9608,0.8718,0.9000],
        "Recall":[0.9524,0.9286,0.9524,0.9286,0.9286,0.8810,0.8810,0.9524,0.8571],
        "F1-Score":[0.9639,0.9510,0.9517,0.9510,0.9397,0.9367,0.9189,0.9104,0.8780],
        "ROC-AUC":[0.9861,0.9974,0.9940,0.9901,0.9928,0.9938,0.9630,0.9930,0.8870],
    }
    df_m = pd.DataFrame(model_data)

    # Tooltip rengi
    def highlight_best(col):
        if col.name == "Model":
            return [""]*len(col)
        best = col.max()
        return ["background-color:#EEF9F3; color:#007A4D; font-weight:700"
                if v == best else "" for v in col]

    for c in ["Accuracy","Precision","Recall","F1-Score","ROC-AUC"]:
        df_m[c] = df_m[c].apply(lambda x: f"{x:.4f}")
    st.dataframe(df_m.style.apply(highlight_best), use_container_width=True, hide_index=True)

    # Grafik
    df_chart = pd.DataFrame(model_data)
    fig_comp = go.Figure()
    metrics  = ["Accuracy","F1-Score","ROC-AUC"]
    colors_m = ["#0F4C81","#2196F3","#64B5F6"]
    for metric, color in zip(metrics, colors_m):
        fig_comp.add_trace(go.Bar(
            x=df_chart["Model"], y=df_chart[metric], name=metric,
            marker_color=color, opacity=0.85,
        ))
    fig_comp.update_layout(
        barmode="group", height=360,
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        xaxis=dict(tickangle=-30, tickfont=dict(size=11, color="#3D5270"), gridcolor="#E8EDF4"),
        yaxis=dict(range=[0.85, 1.0], gridcolor="#E8EDF4", tickfont=dict(color="#9BAABB")),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1,
                    font=dict(color="#3D5270")),
        margin=dict(t=16, b=16, l=0, r=0),
    )
    st.plotly_chart(fig_comp, use_container_width=True)

    # Grafik görselleri
    st.markdown("<div class='sec-header'>🖼️ Analiz Görselleri</div>", unsafe_allow_html=True)
    img_pairs = [
        ("graphs/final_model_confusion_matrix.png", "Confusion Matrix & ROC"),
        ("graphs/model_comparison.png",          "Model Karşılaştırması"),
        ("graphs/shap_importance.png",            "SHAP Özellik Önemi"),
        ("graphs/shap_summary.png",               "SHAP Özet Analizi"),
    ]
    row1 = st.columns(2)
    for i, (path, caption) in enumerate(img_pairs):
        with row1[i % 2]:
            if os.path.exists(path):
                st.image(path, caption=caption, use_container_width=True)


# ══════════════════════════════════════════════════════════════════════════════
#  TAB 4 — VERİ KEŞFİ
# ══════════════════════════════════════════════════════════════════════════════
with tab4:
    st.markdown("<div class='sec-header'>🗂️ Veri Seti Özeti</div>", unsafe_allow_html=True)
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Toplam Örnek", "569")
    c2.metric("Özellik Sayısı", "30")
    c3.metric("Benign", "357  (62.7%)")
    c4.metric("Malignant", "212  (37.3%)")

    # Dağılım
    col_p, col_i = st.columns([1, 2])
    with col_p:
        fig_pie = go.Figure(go.Pie(
            labels=["Benign", "Malignant"], values=[357, 212],
            hole=0.55, marker_colors=["#00C875", "#E03C31"],
            textinfo="label+percent",
            textfont=dict(size=13, family="Inter"),
        ))
        fig_pie.update_layout(
            paper_bgcolor="rgba(0,0,0,0)", showlegend=False,
            height=260, margin=dict(t=10, b=10, l=10, r=10),
            annotations=[dict(text="569", x=0.5, y=0.5,
                               font=dict(size=20, color="#1A2B4A", family="Inter"),
                               showarrow=False)]
        )
        st.plotly_chart(fig_pie, use_container_width=True)

    with col_i:
        eda_imgs = [
            ("graphs/correlation_matrix.png",        "Korelasyon Matrisi"),
            ("graphs/target_correlation.png",         "Hedef Korelasyon"),
            ("graphs/top10_outliers_boxplot.png",     "Aykırı Değerler"),
            ("graphs/diagnosis_distribution.png",     "Dağılım"),
        ]
        sel = st.radio("Görsel Seç", [c for _, c in eda_imgs],
                       horizontal=True, label_visibility="collapsed")
        for path, cap in eda_imgs:
            if cap == sel and os.path.exists(path):
                st.image(path, caption=cap, use_container_width=True)

# ─── Footer ───────────────────────────────────────────────────────────────────
st.markdown("""
<hr style="border:none; border-top:1px solid #E8EDF4; margin:40px 0 16px 0;">
<div style="text-align:center; color:#9BAABB; font-size:0.78rem; padding-bottom:24px;">
    MedPredict AI · Wisconsin Breast Cancer Dataset ·
    Model: <b style="color:#0F4C81;">Logistic Regression</b> ·
    Yalnızca akademik kullanım içindir.
</div>
""", unsafe_allow_html=True)
