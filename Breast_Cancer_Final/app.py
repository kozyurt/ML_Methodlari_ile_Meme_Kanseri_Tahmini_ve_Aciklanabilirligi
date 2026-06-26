# ══════════════════════════════════════════════════════════════════════════════
# Breast Cancer Prediction — Streamlit Dashboard
# Proje mimarisi:
#   data.csv                      → ham veri (Kaggle WBCD)
#   breast_cancer_pipeline.pkl    → eğitilmiş pipeline (FE + Scaler + LogReg)
#   app.py                        → bu dosya
# ══════════════════════════════════════════════════════════════════════════════

import streamlit as st
import pandas as pd
import numpy as np
import pickle
import warnings
from pathlib import Path

# app.py nerede çalışırsa çalışsın dosyaları hep kendi klasöründe arar
BASE_DIR = Path(__file__).parent

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, roc_curve, confusion_matrix,
)
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier

import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

warnings.filterwarnings("ignore")

# ──────────────────────────────────────────────────────────────────────────────
# PAGE CONFIG
# ──────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Breast Cancer Prediction",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ──────────────────────────────────────────────────────────────────────────────
# THEME
# ──────────────────────────────────────────────────────────────────────────────
C = {
    "benign":    "#2ecc71",
    "malignant": "#e74c3c",
    "primary":   "#2980b9",
    "bg":        "#0d1117",
    "card":      "#161b22",
    "border":    "#30363d",
    "text":      "#e6edf3",
    "muted":     "#8b949e",
}

PLOT_BASE = dict(
    paper_bgcolor=C["bg"],
    plot_bgcolor=C["card"],
    font_color=C["text"],
    margin=dict(t=50, b=40, l=40, r=20),
)

st.markdown(f"""
<style>
    .main {{ background-color:{C['bg']}; color:{C['text']}; }}
    section[data-testid="stSidebar"] {{
        background-color:{C['card']};
        border-right:1px solid {C['border']};
    }}
    h1, h2, h3 {{ color:{C['text']} !important; }}

    .kpi-card {{
        background:{C['card']}; border:1px solid {C['border']};
        border-radius:10px; padding:18px 12px; text-align:center;
    }}
    .kpi-val  {{ font-size:2rem; font-weight:800; }}
    .kpi-lbl  {{ font-size:0.78rem; color:{C['muted']}; margin-top:4px; }}

    .sec-title {{
        font-size:1.2rem; font-weight:700; color:{C['text']};
        border-left:3px solid {C['primary']};
        padding-left:10px; margin:22px 0 14px;
    }}

    .result-box  {{ border-radius:14px; padding:28px; text-align:center; margin-top:8px; }}
    .benign-box  {{ background:rgba(46,204,113,0.12); border:2px solid {C['benign']}; }}
    .malign-box  {{ background:rgba(231,76,60,0.12);  border:2px solid {C['malignant']}; }}
    .result-label {{ font-size:2rem; font-weight:900; }}
    .result-sub   {{ font-size:1rem; color:{C['muted']}; margin-top:6px; }}

    .warn-box {{
        background:rgba(231,76,60,0.08); border:1px solid {C['malignant']};
        border-radius:8px; padding:12px; font-size:0.82rem; color:{C['muted']};
    }}
</style>
""", unsafe_allow_html=True)

# ──────────────────────────────────────────────────────────────────────────────
# FEATURE ENGINEERING  (notebook ile birebir aynı)
# ──────────────────────────────────────────────────────────────────────────────
class FeatureEngineeringTransformer(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.feature_names_ = None

    def fit(self, X, y=None):
        if isinstance(X, pd.DataFrame):
            self.feature_names_ = X.columns.tolist()
        return self

    def transform(self, X):
        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(
                X,
                columns=self.feature_names_ if self.feature_names_ else None,
            )
        X = X.copy()
        if "radius_mean" in X.columns and "area_mean" in X.columns:
            X["radius_area_ratio"] = X["radius_mean"] / (X["area_mean"] + 1e-6)
        if "perimeter_mean" in X.columns and "area_mean" in X.columns:
            X["perimeter_area_ratio"] = X["perimeter_mean"] / (X["area_mean"] + 1e-6)
        if "concavity_mean" in X.columns and "concave points_mean" in X.columns:
            X["concavity_points_product"] = (
                X["concavity_mean"] * X["concave points_mean"]
            )
        return X

# ──────────────────────────────────────────────────────────────────────────────
# DATA  — data.csv (zorunlu)
# ──────────────────────────────────────────────────────────────────────────────
@st.cache_data(show_spinner="Loading dataset…")
def load_data():
    df = pd.read_csv(BASE_DIR / "data.csv")
    df = df.drop(columns=[c for c in ["id", "Unnamed: 32"] if c in df.columns])
    df["diagnosis"] = LabelEncoder().fit_transform(df["diagnosis"])  # B=0, M=1
    X = df.drop("diagnosis", axis=1)
    y = df["diagnosis"]
    return X, y

# ──────────────────────────────────────────────────────────────────────────────
# SAVED PIPELINE  — breast_cancer_pipeline.pkl (zorunlu)
# ──────────────────────────────────────────────────────────────────────────────
@st.cache_resource(show_spinner="Loading saved pipeline…")
def load_pipeline():
    with open(BASE_DIR / "breast_cancer_pipeline.pkl", "rb") as f:
        return pickle.load(f)

# ──────────────────────────────────────────────────────────────────────────────
# COMPARISON MODELS  (sadece Model Comparison sayfası için)
# ──────────────────────────────────────────────────────────────────────────────
@st.cache_resource(show_spinner="Training comparison models…")
def train_comparison(_X, _y):
    X_tr, X_te, y_tr, y_te = train_test_split(
        _X, _y, test_size=0.2, random_state=42, stratify=_y
    )

    pre = Pipeline([
        ("fe",     FeatureEngineeringTransformer()),
        ("scaler", StandardScaler()),
    ])
    X_tr_s = pre.fit_transform(X_tr)
    X_te_s = pre.transform(X_te)

    feat_names = FeatureEngineeringTransformer().fit_transform(X_tr).columns.tolist()
    X_tr_s = pd.DataFrame(X_tr_s, columns=feat_names)
    X_te_s = pd.DataFrame(X_te_s, columns=feat_names)

    model_zoo = {
        "Logistic Regression": LogisticRegression(random_state=42, max_iter=10000, C=0.1),
        "Random Forest":       RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1),
        "Gradient Boosting":   GradientBoostingClassifier(n_estimators=200, random_state=42),
        "SVM":                 SVC(kernel="rbf", probability=True, random_state=42),
        "KNN":                 KNeighborsClassifier(n_neighbors=5),
        "Naive Bayes":         GaussianNB(),
        "Decision Tree":       DecisionTreeClassifier(random_state=42, max_depth=10),
    }

    results = {}
    for name, mdl in model_zoo.items():
        mdl.fit(X_tr_s, y_tr)
        yp   = mdl.predict(X_te_s)
        yprb = mdl.predict_proba(X_te_s)[:, 1]
        cv   = cross_val_score(mdl, X_tr_s, y_tr, cv=5, scoring="accuracy")
        results[name] = dict(
            Accuracy  = accuracy_score(y_te, yp),
            Precision = precision_score(y_te, yp, zero_division=0),
            Recall    = recall_score(y_te, yp, zero_division=0),
            F1        = f1_score(y_te, yp, zero_division=0),
            AUC       = roc_auc_score(y_te, yprb),
            CV_mean   = cv.mean(),
            CV_std    = cv.std(),
            y_pred    = yp,
            y_proba   = yprb,
        )

    # Feature importance via RF
    rf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
    rf.fit(X_tr_s, y_tr)
    imp = (
        pd.DataFrame({"feature": feat_names, "importance": rf.feature_importances_})
        .sort_values("importance", ascending=False)
        .reset_index(drop=True)
    )
    return results, X_te, y_te, imp

# ──────────────────────────────────────────────────────────────────────────────
# BOOTSTRAP
# ──────────────────────────────────────────────────────────────────────────────
try:
    X_raw, y_raw = load_data()
except FileNotFoundError:
    st.error("⚠️  `data.csv` bulunamadı. Dosyayı `app.py` ile aynı klasöre koy.")
    st.stop()

try:
    pipeline = load_pipeline()
except FileNotFoundError:
    st.error("⚠️  `breast_cancer_pipeline.pkl` bulunamadı. Dosyayı `app.py` ile aynı klasöre koy.")
    st.stop()

# pkl'den feature listesi al — kesinlikle doğru isimler bunlar
ORIG_FEATURES = pipeline.named_steps["feature_engineering"].feature_names_

# Saved pipeline metriklerini test seti üzerinden hesapla
X_tr_sp, X_te_sp, y_tr_sp, y_te_sp = train_test_split(
    X_raw, y_raw, test_size=0.2, random_state=42, stratify=y_raw
)
_yp   = pipeline.predict(X_te_sp)
_yprb = pipeline.predict_proba(X_te_sp)[:, 1]
PKL_METRICS = dict(
    Accuracy  = accuracy_score(y_te_sp, _yp),
    Precision = precision_score(y_te_sp, _yp, zero_division=0),
    Recall    = recall_score(y_te_sp, _yp, zero_division=0),
    F1        = f1_score(y_te_sp, _yp, zero_division=0),
    AUC       = roc_auc_score(y_te_sp, _yprb),
    y_pred    = _yp,
    y_proba   = _yprb,
)

comp_results, X_te_comp, y_te_comp, imp_df = train_comparison(X_raw, y_raw)

BEST_NAME = "Logistic Regression"

# ──────────────────────────────────────────────────────────────────────────────
# SIDEBAR
# ──────────────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🔬 Breast Cancer\nPrediction Dashboard")
    st.divider()
    page = st.radio(
        "nav",
        ["📊 Overview", "🔍 EDA", "🤖 Model Comparison", "🎯 Predict"],
        label_visibility="collapsed",
    )
    st.divider()
    st.caption(
        f"Best model: **{BEST_NAME}**  \n"
        f"Accuracy : **{PKL_METRICS['Accuracy']:.4f}**  \n"
        f"ROC-AUC  : **{PKL_METRICS['AUC']:.4f}**"
    )

# ──────────────────────────────────────────────────────────────────────────────
# HELPER: KPI card
# ──────────────────────────────────────────────────────────────────────────────
def kpi(col, val, lbl, color=None):
    col.markdown(f"""
    <div class="kpi-card">
        <div class="kpi-val" style="color:{color or C['primary']}">{val}</div>
        <div class="kpi-lbl">{lbl}</div>
    </div>""", unsafe_allow_html=True)

def sec(title):
    st.markdown(f'<div class="sec-title">{title}</div>', unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
# PAGE ❶  OVERVIEW
# ══════════════════════════════════════════════════════════════════════════════
if page == "📊 Overview":
    st.markdown("# 🔬 Breast Cancer Prediction")
    st.markdown(
        "Binary classification of tumor cells as **Benign** or **Malignant** "
        "using the Wisconsin Diagnostic Breast Cancer dataset."
    )
    st.divider()

    c1, c2, c3, c4, c5 = st.columns(5)
    kpi(c1, len(X_raw),            "Total Samples",              C["primary"])
    kpi(c2, int((y_raw==0).sum()), "Benign (0)",                 C["benign"])
    kpi(c3, int((y_raw==1).sum()), "Malignant (1)",              C["malignant"])
    kpi(c4, f"{PKL_METRICS['Accuracy']:.4f}", f"Accuracy\n({BEST_NAME})", C["primary"])
    kpi(c5, f"{PKL_METRICS['AUC']:.4f}",      "ROC-AUC",        C["primary"])

    st.divider()
    counts = y_raw.value_counts().sort_index()
    col_a, col_b = st.columns(2)

    with col_a:
        sec("Target Distribution")
        fig = go.Figure(go.Bar(
            x=["Benign (0)", "Malignant (1)"],
            y=[counts[0], counts[1]],
            marker_color=[C["benign"], C["malignant"]],
            text=[counts[0], counts[1]], textposition="outside",
        ))
        fig.update_layout(height=320, showlegend=False, **PLOT_BASE)
        st.plotly_chart(fig, use_container_width=True)

    with col_b:
        sec("Class Ratio")
        fig2 = go.Figure(go.Pie(
            labels=["Benign", "Malignant"],
            values=[counts[0], counts[1]],
            marker_colors=[C["benign"], C["malignant"]],
            hole=0.4, textinfo="label+percent",
        ))
        fig2.update_layout(height=320, **PLOT_BASE)
        st.plotly_chart(fig2, use_container_width=True)

    st.divider()
    sec("Key Findings")
    f1c, f2c, f3c = st.columns(3)
    with f1c:
        st.success(
            "**Best Model: Logistic Regression**\n\n"
            "F1-Score ~0.97 ve ROC-AUC ~0.996 — sınıfları neredeyse mükemmel ayırıyor."
        )
    with f2c:
        st.info(
            "**Top Feature: texture_worst**\n\n"
            "SHAP analizi: en kötü doku değeri ve radius_se en güçlü tahmin edici."
        )
    with f3c:
        st.warning(
            "**Sadece 3 Yanlış Sınıflandırma**\n\n"
            "Confusion matrix: 71 TN · 40 TP · 1 FP · 2 FN — son derece düşük FN oranı."
        )

# ══════════════════════════════════════════════════════════════════════════════
# PAGE ❷  EDA
# ══════════════════════════════════════════════════════════════════════════════
elif page == "🔍 EDA":
    st.markdown("# 🔍 Exploratory Data Analysis")
    st.divider()

    t1, t2, t3, t4 = st.tabs([
        "Feature Importance", "Outlier Analysis",
        "Correlation Heatmap", "Feature Distributions",
    ])

    # ── Feature Importance ────────────────────────────────────────────────────
    with t1:
        sec("Feature Importance  (Random Forest + Feature Engineering)")
        top_n = st.slider("Top N features", 10, len(imp_df), 20)
        df_imp = imp_df.head(top_n).sort_values("importance")
        colors = [C["malignant"] if i >= len(df_imp)-3 else C["primary"]
                  for i in range(len(df_imp))]
        fig = go.Figure(go.Bar(
            x=df_imp["importance"], y=df_imp["feature"],
            orientation="h", marker_color=colors,
            text=[f"{v:.4f}" for v in df_imp["importance"]], textposition="outside",
        ))
        fig.update_layout(
            height=max(400, top_n*22),
            xaxis_title="Importance Score",
            title="Top 3 kırmızı — concavity_points_product (engineered) çok yüksek",
            **PLOT_BASE,
        )
        st.plotly_chart(fig, use_container_width=True)

    # ── Outlier Analysis ──────────────────────────────────────────────────────
    with t2:
        sec("Top 10 Features with Most Outliers  (IQR Method)")
        def n_iqr(s):
            q1, q3 = s.quantile(.25), s.quantile(.75)
            iqr = q3 - q1
            return ((s < q1-1.5*iqr) | (s > q3+1.5*iqr)).sum()
        top10 = sorted(X_raw.columns, key=lambda c: n_iqr(X_raw[c]), reverse=True)[:10]
        fig = go.Figure()
        for col in top10:
            fig.add_trace(go.Box(
                y=X_raw[col], name=col, boxpoints="outliers",
                marker_color=C["primary"], line_color=C["primary"],
            ))
        fig.update_layout(height=480, showlegend=False,
                          title="Boxplot — Top 10 Features by Outlier Count", **PLOT_BASE)
        st.plotly_chart(fig, use_container_width=True)
        st.info(
            "ℹ️ Bu aykırı değerler hata değil — medikal veride uç değerler "
            "malign hücrelerin gerçek biyolojik varyansını yansıtır."
        )

    # ── Correlation Heatmap ───────────────────────────────────────────────────
    with t3:
        sec("Correlation Matrix")
        grp = st.selectbox("Feature group", ["_mean", "_se", "_worst", "All (first 15)"])
        cols_hm = (X_raw.columns[:15].tolist() if "All" in grp
                   else [c for c in X_raw.columns if grp in c])
        fig = px.imshow(
            X_raw[cols_hm].corr(),
            color_continuous_scale="RdBu_r", zmin=-1, zmax=1,
            text_auto=".2f",
            title=f"Correlation Matrix — {grp} features",
            template="plotly_dark",
        )
        fig.update_layout(height=500, paper_bgcolor=C["bg"], font_color=C["text"])
        st.plotly_chart(fig, use_container_width=True)

    # ── Feature Distributions ─────────────────────────────────────────────────
    with t4:
        sec("Feature Distribution by Diagnosis")
        feat = st.selectbox("Select feature", X_raw.columns.tolist())
        df_p = X_raw.assign(diagnosis=y_raw)
        fig = go.Figure()
        fig.add_trace(go.Histogram(
            x=df_p[df_p.diagnosis==0][feat], name="Benign",
            marker_color=C["benign"], opacity=0.7, nbinsx=40,
        ))
        fig.add_trace(go.Histogram(
            x=df_p[df_p.diagnosis==1][feat], name="Malignant",
            marker_color=C["malignant"], opacity=0.7, nbinsx=40,
        ))
        fig.update_layout(
            barmode="overlay", height=380,
            title=f"Distribution of  {feat}",
            xaxis_title=feat, yaxis_title="Count",
            **PLOT_BASE,
        )
        st.plotly_chart(fig, use_container_width=True)

# ══════════════════════════════════════════════════════════════════════════════
# PAGE ❸  MODEL COMPARISON
# ══════════════════════════════════════════════════════════════════════════════
elif page == "🤖 Model Comparison":
    st.markdown("# 🤖 Model Comparison")
    st.divider()

    comp_df = (
        pd.DataFrame([
            dict(Model=k, Accuracy=v["Accuracy"], Precision=v["Precision"],
                 Recall=v["Recall"], **{"F1-Score": v["F1"]},
                 **{"ROC-AUC": v["AUC"]}, CV_Mean=v["CV_mean"], CV_Std=v["CV_std"])
            for k, v in comp_results.items()
        ])
        .sort_values("F1-Score", ascending=False)
        .reset_index(drop=True)
    )

    sec(f"Best Model: {BEST_NAME}  (Saved Pipeline Metrics)")
    mc1, mc2, mc3, mc4, mc5 = st.columns(5)
    kpi(mc1, f"{PKL_METRICS['Accuracy']:.4f}",  "Accuracy",  C["primary"])
    kpi(mc2, f"{PKL_METRICS['Precision']:.4f}", "Precision", C["benign"])
    kpi(mc3, f"{PKL_METRICS['Recall']:.4f}",    "Recall",    C["malignant"])
    kpi(mc4, f"{PKL_METRICS['F1']:.4f}",        "F1-Score",  C["primary"])
    kpi(mc5, f"{PKL_METRICS['AUC']:.4f}",       "ROC-AUC",   C["primary"])

    st.divider()
    ta, tb, tc, td = st.tabs(["Accuracy & CV", "All Metrics", "ROC Curves", "Confusion Matrix"])

    # ── Accuracy & CV ─────────────────────────────────────────────────────────
    with ta:
        bar_c = [C["malignant"] if m == BEST_NAME else C["primary"] for m in comp_df["Model"]]
        fig = make_subplots(rows=1, cols=2,
                            subplot_titles=("Accuracy", "Cross-Validation Mean ± Std"))
        fig.add_trace(go.Bar(
            x=comp_df["Accuracy"], y=comp_df["Model"], orientation="h",
            marker_color=bar_c, text=[f"{v:.4f}" for v in comp_df["Accuracy"]],
            textposition="outside", name="Accuracy",
        ), row=1, col=1)
        fig.add_trace(go.Bar(
            x=comp_df["CV_Mean"], y=comp_df["Model"], orientation="h",
            error_x=dict(type="data", array=comp_df["CV_Std"]),
            marker_color=bar_c, name="CV Mean",
        ), row=1, col=2)
        fig.add_vline(x=PKL_METRICS["Accuracy"], line_dash="dash",
                      line_color=C["malignant"], row=1, col=1)
        fig.update_xaxes(range=[0.80, 1.0])
        fig.update_layout(height=380, showlegend=False, **PLOT_BASE)
        st.plotly_chart(fig, use_container_width=True)

    # ── All Metrics ───────────────────────────────────────────────────────────
    with tb:
        sel = st.multiselect(
            "Metrics",
            ["Accuracy", "Precision", "Recall", "F1-Score", "ROC-AUC"],
            default=["Accuracy", "Precision", "Recall", "F1-Score"],
        )
        mc = {"Accuracy": C["primary"], "Precision": "#f39c12",
              "Recall": C["benign"], "F1-Score": C["malignant"], "ROC-AUC": "#9b59b6"}
        fig = go.Figure()
        for m in sel:
            fig.add_trace(go.Bar(
                x=comp_df[m], y=comp_df["Model"], orientation="h",
                name=m, marker_color=mc.get(m, C["primary"]), opacity=0.85,
            ))
        fig.update_layout(barmode="group", height=420, xaxis_range=[0.80, 1.0], **PLOT_BASE)
        st.plotly_chart(fig, use_container_width=True)

    # ── ROC Curves ────────────────────────────────────────────────────────────
    with tc:
        pal = [C["primary"], C["benign"], C["malignant"],
               "#f39c12", "#9b59b6", "#1abc9c", "#e67e22"]
        fig = go.Figure()
        for i, (name, res) in enumerate(comp_results.items()):
            fpr, tpr, _ = roc_curve(y_te_comp, res["y_proba"])
            fig.add_trace(go.Scatter(
                x=fpr, y=tpr, mode="lines",
                name=f"{name}  (AUC={res['AUC']:.4f})",
                line=dict(color=pal[i % len(pal)],
                          width=3 if name == BEST_NAME else 1.5),
            ))
        fig.add_trace(go.Scatter(
            x=[0, 1], y=[0, 1], mode="lines", name="Random",
            line=dict(color=C["muted"], dash="dash", width=1),
        ))
        fig.update_layout(
            height=460,
            xaxis_title="False Positive Rate",
            yaxis_title="True Positive Rate",
            title="ROC Curves — All Models",
            **PLOT_BASE,
        )
        st.plotly_chart(fig, use_container_width=True)

    # ── Confusion Matrix ──────────────────────────────────────────────────────
    with td:
        mdl_sel = st.selectbox(
            "Select model", list(comp_results.keys()),
            index=list(comp_results.keys()).index(BEST_NAME),
        )
        cm = confusion_matrix(y_te_comp, comp_results[mdl_sel]["y_pred"])
        fig = px.imshow(
            cm, text_auto=True, color_continuous_scale="Blues",
            x=["Benign (0)", "Malignant (1)"],
            y=["Benign (0)", "Malignant (1)"],
            labels=dict(x="Predicted", y="Actual"),
            title=f"Confusion Matrix — {mdl_sel}",
        )
        fig.update_layout(height=380, paper_bgcolor=C["bg"], font_color=C["text"])
        st.plotly_chart(fig, use_container_width=True)

        tn, fp, fn, tp = cm.ravel()
        cc1, cc2, cc3, cc4 = st.columns(4)
        kpi(cc1, tn, "True Negative",              C["benign"])
        kpi(cc2, tp, "True Positive",              C["benign"])
        kpi(cc3, fp, "False Positive",             C["malignant"])
        kpi(cc4, fn, "False Negative (critical!)", C["malignant"])

# ══════════════════════════════════════════════════════════════════════════════
# PAGE ❹  PREDICT
# ══════════════════════════════════════════════════════════════════════════════
elif page == "🎯 Predict":
    st.markdown("# 🎯 Tumor Diagnosis Predictor")
    st.markdown(
        f"Aşağıdaki hücre ölçümlerini girin. "
        f"**{BEST_NAME}** pipeline tahmin üretecek."
    )
    st.divider()

    st.markdown(
        '<div class="warn-box">⚠️ <strong>Tıbbi Uyarı:</strong> '
        "Bu araç yalnızca eğitim ve araştırma amaçlıdır. "
        "Profesyonel tıbbi tanının yerini tutmaz. "
        "Lütfen her zaman nitelikli bir sağlık uzmanına danışın.</div>",
        unsafe_allow_html=True,
    )
    st.divider()

    # Stats for input bounds
    stats   = X_raw[ORIG_FEATURES].describe()
    X_b_med = X_raw[y_raw == 0][ORIG_FEATURES].median()
    X_m_med = X_raw[y_raw == 1][ORIG_FEATURES].median()
    X_med   = X_raw[ORIG_FEATURES].median()

    # session_state ile fill modunu kalici tut
    if "fill_mode" not in st.session_state:
        st.session_state.fill_mode = "overall"

    # Quick fill butonlari
    st.markdown("**Quick fill:**")
    qc1, qc2, qc3 = st.columns(3)
    if qc1.button("Fill Benign median",      use_container_width=True):
        st.session_state.fill_mode = "benign"
        for f in ORIG_FEATURES:
            st.session_state.pop(f"inp_{f}", None)
        st.rerun()
    if qc2.button("Fill Malignant median",   use_container_width=True):
        st.session_state.fill_mode = "malignant"
        for f in ORIG_FEATURES:
            st.session_state.pop(f"inp_{f}", None)
        st.rerun()
    if qc3.button("Reset to overall median", use_container_width=True):
        st.session_state.fill_mode = "overall"
        for f in ORIG_FEATURES:
            st.session_state.pop(f"inp_{f}", None)
        st.rerun()

    def default(col):
        mode = st.session_state.fill_mode
        if mode == "benign":    return float(X_b_med[col])
        if mode == "malignant": return float(X_m_med[col])
        return float(X_med[col])

        # Feature grupları
    mean_feats  = [c for c in ORIG_FEATURES if c.endswith("_mean")]
    se_feats    = [c for c in ORIG_FEATURES if c.endswith("_se")]
    worst_feats = [c for c in ORIG_FEATURES if c.endswith("_worst")]
    other_feats = [c for c in ORIG_FEATURES
                   if c not in mean_feats + se_feats + worst_feats]

    input_vals = {}
    groups = [
        ("📐 Mean Features",           mean_feats),
        ("📏 Standard Error Features", se_feats),
        ("⚠️ Worst Features",          worst_feats),
    ]
    if other_feats:
        groups.append(("📋 Other Features", other_feats))

    for grp_lbl, grp_feats in groups:
        sec(grp_lbl)
        cols = st.columns(5)
        for i, feat in enumerate(grp_feats):
            with cols[i % 5]:
                input_vals[feat] = st.number_input(
                    feat,
                    min_value=float(stats[feat]["min"]),
                    max_value=float(stats[feat]["max"]),
                    value=default(feat),
                    format="%.4f",
                    key=f"inp_{feat}",
                )

    st.divider()

    if st.button("🔬 Run Prediction", use_container_width=True, type="primary"):
        # DataFrame'i tam olarak ORIG_FEATURES sırasında oluştur
        input_df = pd.DataFrame([input_vals])[ORIG_FEATURES]

        pred   = pipeline.predict(input_df)[0]
        proba  = pipeline.predict_proba(input_df)[0]
        p_ben  = proba[0]
        p_mal  = proba[1]

        left, right = st.columns(2, gap="large")

        with left:
            if pred == 0:
                st.markdown(f"""
                <div class="result-box benign-box">
                    <div class="result-label" style="color:{C['benign']}">✅ BENIGN</div>
                    <div class="result-sub">
                        Model bu tümörün <strong>iyi huylu (benign)</strong> olduğunu tahmin etti.
                    </div>
                    <div style="font-size:1.5rem;margin-top:12px;color:{C['benign']}">
                        Güven: {p_ben*100:.1f}%
                    </div>
                </div>""", unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div class="result-box malign-box">
                    <div class="result-label" style="color:{C['malignant']}">⚠️ MALIGNANT</div>
                    <div class="result-sub">
                        Model bu tümörün <strong>kötü huylu (malignant)</strong> olduğunu tahmin etti.
                    </div>
                    <div style="font-size:1.5rem;margin-top:12px;color:{C['malignant']}">
                        Güven: {p_mal*100:.1f}%
                    </div>
                </div>""", unsafe_allow_html=True)

        with right:
            fig = go.Figure(go.Bar(
                x=["Benign", "Malignant"],
                y=[p_ben*100, p_mal*100],
                marker_color=[C["benign"], C["malignant"]],
                text=[f"{p_ben*100:.1f}%", f"{p_mal*100:.1f}%"],
                textposition="outside",
                width=0.4,
            ))
            fig.update_layout(
                height=300, yaxis_range=[0, 115],
                yaxis_title="Probability (%)",
                title="Prediction Probability",
                **PLOT_BASE,
            )
            st.plotly_chart(fig, use_container_width=True)

        # Engineered features (bilgi amaçlı)
        st.divider()
        sec("🧬 Engineered Features  (pipeline tarafından otomatik hesaplanır)")
        fe_t = FeatureEngineeringTransformer().fit(X_raw[ORIG_FEATURES])
        eng  = fe_t.transform(input_df)
        e1, e2, e3 = st.columns(3)
        e1.metric("radius_area_ratio",        f"{eng['radius_area_ratio'].values[0]:.6f}")
        e2.metric("perimeter_area_ratio",     f"{eng['perimeter_area_ratio'].values[0]:.6f}")
        e3.metric("concavity_points_product", f"{eng['concavity_points_product'].values[0]:.6f}")

        # Risk uyarısı
        st.divider()
        if pred == 1 and p_mal > 0.8:
            st.error(
                "🚨 **Yüksek güvenle malignant tahmini.** "
                "Lütfen derhal bir sağlık uzmanına başvurun."
            )
        elif pred == 1:
            st.warning(
                "⚠️ **Orta güvenle malignant tahmini.** "
                "İleri klinik değerlendirme önerilir."
            )
        else:
            st.success("✅ Benign tahmini. Düzenli kontroller önerilir.")
