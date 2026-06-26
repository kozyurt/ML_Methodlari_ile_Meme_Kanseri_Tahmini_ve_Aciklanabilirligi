import streamlit as st
import pandas as pd
import numpy as np
import warnings
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.pipeline import Pipeline
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, roc_curve, confusion_matrix
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

# ── Page config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Breast Cancer Prediction",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Colors ─────────────────────────────────────────────────────────────────────
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

# ── CSS ────────────────────────────────────────────────────────────────────────
st.markdown(f"""
<style>
    .main {{ background-color:{C['bg']}; color:{C['text']}; }}
    section[data-testid="stSidebar"] {{ background-color:{C['card']}; border-right:1px solid {C['border']}; }}
    h1,h2,h3 {{ color:{C['text']} !important; }}
    .kpi-card {{
        background:{C['card']}; border:1px solid {C['border']};
        border-radius:10px; padding:18px 12px; text-align:center;
    }}
    .kpi-val {{ font-size:2rem; font-weight:800; }}
    .kpi-lbl {{ font-size:0.78rem; color:{C['muted']}; margin-top:4px; }}
    .sec-title {{
        font-size:1.2rem; font-weight:700; color:{C['text']};
        border-left:3px solid {C['primary']}; padding-left:10px; margin:22px 0 14px;
    }}
    .result-box {{
        border-radius:14px; padding:28px; text-align:center; margin-top:8px;
    }}
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

# ══════════════════════════════════════════════════════════════════════════════
# FEATURE ENGINEERING (matches notebook exactly)
# ══════════════════════════════════════════════════════════════════════════════
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

# ══════════════════════════════════════════════════════════════════════════════
# DATA & MODEL LOADING
# ══════════════════════════════════════════════════════════════════════════════
@st.cache_data(show_spinner="Loading dataset…")
def load_data():
    # Try local file first, fall back to sklearn built-in
    try:
        df = pd.read_csv("data.csv")
        df = df.drop(columns=[c for c in ['id', 'Unnamed: 32'] if c in df.columns])
        encoder = LabelEncoder()
        df['diagnosis'] = encoder.fit_transform(df['diagnosis'])
        X = df.drop('diagnosis', axis=1)
        y = df['diagnosis']
    except FileNotFoundError:
        data = load_breast_cancer()
        X = pd.DataFrame(data.data, columns=data.feature_names)
        # Rename to match notebook feature names
        rename = {n: n.replace(' ', '_') for n in X.columns}
        # Keep original names for compatibility
        y = pd.Series(data.target)
    return X, y

@st.cache_resource(show_spinner="Training models…")
def train_models():
    X, y = load_data()
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    preprocessing = Pipeline([
        ('feature_engineering', FeatureEngineeringTransformer()),
        ('scaler', StandardScaler())
    ])
    X_train_s = preprocessing.fit_transform(X_train)
    X_test_s  = preprocessing.transform(X_test)

    fe = FeatureEngineeringTransformer().fit_transform(X_train)
    feature_names = fe.columns.tolist()
    X_train_s = pd.DataFrame(X_train_s, columns=feature_names)
    X_test_s  = pd.DataFrame(X_test_s,  columns=feature_names)

    models = {
        'Logistic Regression':  LogisticRegression(random_state=42, max_iter=10000, C=0.1, solver='lbfgs'),
        'Random Forest':        RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1),
        'Gradient Boosting':    GradientBoostingClassifier(n_estimators=200, random_state=42),
        'SVM':                  SVC(kernel='rbf', probability=True, random_state=42),
        'KNN':                  KNeighborsClassifier(n_neighbors=5),
        'Naive Bayes':          GaussianNB(),
        'Decision Tree':        DecisionTreeClassifier(random_state=42, max_depth=10),
    }

    results = {}
    for name, model in models.items():
        model.fit(X_train_s, y_train)
        y_pred  = model.predict(X_test_s)
        y_proba = model.predict_proba(X_test_s)[:, 1]
        cv      = cross_val_score(model, X_train_s, y_train, cv=5, scoring='accuracy')
        results[name] = {
            'model':     model,
            'Accuracy':  accuracy_score(y_test, y_pred),
            'Precision': precision_score(y_test, y_pred, zero_division=0),
            'Recall':    recall_score(y_test, y_pred, zero_division=0),
            'F1-Score':  f1_score(y_test, y_pred, zero_division=0),
            'ROC-AUC':   roc_auc_score(y_test, y_proba),
            'CV_Mean':   cv.mean(),
            'CV_Std':    cv.std(),
            'y_pred':    y_pred,
            'y_proba':   y_proba,
        }

    best_name = max(results, key=lambda k: results[k]['F1-Score'])

    # Build final pipeline
    final_pipeline = Pipeline([
        ('feature_engineering', FeatureEngineeringTransformer()),
        ('scaler', StandardScaler()),
        ('model', results[best_name]['model'])
    ])
    final_pipeline.fit(X_train, y_train)

    # Feature importance (RF)
    rf_imp = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
    rf_imp.fit(X_train_s, y_train)
    imp_df = pd.DataFrame({'feature': feature_names, 'importance': rf_imp.feature_importances_})\
               .sort_values('importance', ascending=False)

    return (results, best_name, X_test, y_test, X_train,
            preprocessing, final_pipeline, imp_df, feature_names, X, y)

# ── Load ───────────────────────────────────────────────────────────────────────
X_raw, y_raw = load_data()
(results, best_name, X_test, y_test, X_train,
 preprocessing, final_pipeline, imp_df, feature_names, X_all, y_all) = train_models()

ORIG_FEATURES = X_raw.columns.tolist()
best = results[best_name]

# ══════════════════════════════════════════════════════════════════════════════
# SIDEBAR
# ══════════════════════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown("## 🔬 Breast Cancer\nPrediction Dashboard")
    st.divider()
    page = st.radio(
        "Navigate",
        ["📊 Overview", "🔍 EDA", "🤖 Model Comparison", "🎯 Predict"],
        label_visibility="collapsed",
    )
    st.divider()
    st.caption(f"Best model: **{best_name}**  \nAccuracy: **{best['Accuracy']:.4f}**  \nAUC: **{best['ROC-AUC']:.4f}**")

# ══════════════════════════════════════════════════════════════════════════════
# HELPERS
# ══════════════════════════════════════════════════════════════════════════════
PLOT_LAYOUT = dict(
    paper_bgcolor=C['bg'], plot_bgcolor=C['card'],
    font_color=C['text'], margin=dict(t=50, b=40, l=40, r=20)
)

def kpi(col, value, label, color=None):
    color = color or C['primary']
    col.markdown(f"""
    <div class="kpi-card">
        <div class="kpi-val" style="color:{color}">{value}</div>
        <div class="kpi-lbl">{label}</div>
    </div>""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
# PAGE: OVERVIEW
# ══════════════════════════════════════════════════════════════════════════════
if page == "📊 Overview":
    st.markdown("# 🔬 Breast Cancer Prediction")
    st.markdown("Binary classification of tumor cells as **Benign** or **Malignant** using Wisconsin Diagnostic Breast Cancer data.")
    st.divider()

    c1,c2,c3,c4,c5 = st.columns(5)
    kpi(c1, len(X_all), "Total Samples", C['primary'])
    kpi(c2, f"{(y_all==0).sum()}", "Benign (0)", C['benign'])
    kpi(c3, f"{(y_all==1).sum()}", "Malignant (1)", C['malignant'])
    kpi(c4, f"{best['Accuracy']:.4f}", f"Best Accuracy\n({best_name})", C['primary'])
    kpi(c5, f"{best['ROC-AUC']:.4f}", "Best ROC-AUC", C['primary'])

    st.divider()

    col_a, col_b = st.columns(2)

    with col_a:
        st.markdown('<div class="sec-title">Target Distribution</div>', unsafe_allow_html=True)
        counts = y_all.value_counts().sort_index()
        fig = go.Figure(go.Bar(
            x=['Benign (0)', 'Malignant (1)'],
            y=[counts[0], counts[1]],
            marker_color=[C['benign'], C['malignant']],
            text=[counts[0], counts[1]], textposition='outside',
        ))
        fig.update_layout(height=320, showlegend=False, **PLOT_LAYOUT)
        st.plotly_chart(fig, use_container_width=True)

    with col_b:
        st.markdown('<div class="sec-title">Class Ratio</div>', unsafe_allow_html=True)
        fig2 = go.Figure(go.Pie(
            labels=['Benign', 'Malignant'], values=[counts[0], counts[1]],
            marker_colors=[C['benign'], C['malignant']],
            hole=0.4, textinfo='label+percent',
        ))
        fig2.update_layout(height=320, **PLOT_LAYOUT)
        st.plotly_chart(fig2, use_container_width=True)

    st.divider()
    st.markdown('<div class="sec-title">Key Findings</div>', unsafe_allow_html=True)
    f1c, f2c, f3c = st.columns(3)
    with f1c:
        st.success("**Best Model: Logistic Regression**\n\nAchieves F1-Score ~0.97 and ROC-AUC ~0.996 — near-perfect separation of classes.")
    with f2c:
        st.info("**Top Feature: texture_worst**\n\nSHAP analysis reveals worst-case texture and radius_se are the strongest predictors.")
    with f3c:
        st.warning("**Only 3 Misclassifications**\n\nConfusion matrix: 71 TN, 40 TP, 1 FP, 2 FN — extremely low false negative rate.")

# ══════════════════════════════════════════════════════════════════════════════
# PAGE: EDA
# ══════════════════════════════════════════════════════════════════════════════
elif page == "🔍 EDA":
    st.markdown("# 🔍 Exploratory Data Analysis")
    st.divider()

    tab1, tab2, tab3, tab4 = st.tabs(["Feature Importance", "Outlier Analysis", "Correlation Heatmap", "Feature Distributions"])

    # ── Tab 1: Feature Importance ──────────────────────────────────────────────
    with tab1:
        st.markdown('<div class="sec-title">Feature Importance (Random Forest + Feature Engineering)</div>', unsafe_allow_html=True)
        top_n = st.slider("Show top N features", 10, len(imp_df), 20, key="imp_n")
        df_imp = imp_df.head(top_n).sort_values('importance')
        bar_colors = [C['malignant'] if i >= len(df_imp)-3 else C['primary'] for i in range(len(df_imp))]
        fig = go.Figure(go.Bar(
            x=df_imp['importance'], y=df_imp['feature'],
            orientation='h', marker_color=bar_colors,
            text=[f"{v:.4f}" for v in df_imp['importance']], textposition='outside',
        ))
        fig.update_layout(height=max(400, top_n*22), xaxis_title="Importance Score",
                          title="Feature Importance — engineered features highlighted in red", **PLOT_LAYOUT)
        st.plotly_chart(fig, use_container_width=True)
        st.caption("🔴 **Red bars** = top 3 features. Note: `concavity_points_product` (engineered) ranks very high.")

    # ── Tab 2: Outlier Boxplot ─────────────────────────────────────────────────
    with tab2:
        st.markdown('<div class="sec-title">Top 10 Features with Most Outliers (IQR Method)</div>', unsafe_allow_html=True)

        def count_iqr_outliers(series):
            Q1, Q3 = series.quantile(0.25), series.quantile(0.75)
            IQR = Q3 - Q1
            return ((series < Q1 - 1.5*IQR) | (series > Q3 + 1.5*IQR)).sum()

        outlier_counts = {col: count_iqr_outliers(X_raw[col]) for col in X_raw.columns}
        top10_outlier = sorted(outlier_counts, key=outlier_counts.get, reverse=True)[:10]

        fig = go.Figure()
        for col in top10_outlier:
            fig.add_trace(go.Box(y=X_raw[col], name=col, boxpoints='outliers',
                                 marker_color=C['primary'], line_color=C['primary']))
        fig.update_layout(height=480, title="Boxplot — Top 10 Features by Outlier Count",
                          showlegend=False, **PLOT_LAYOUT)
        st.plotly_chart(fig, use_container_width=True)
        st.info("ℹ️ These outliers are **not errors** — extreme values in medical data often represent genuine biological variance in malignant cases.")

    # ── Tab 3: Correlation Heatmap ─────────────────────────────────────────────
    with tab3:
        st.markdown('<div class="sec-title">Correlation Matrix</div>', unsafe_allow_html=True)
        feature_group = st.selectbox("Feature group", ["_mean", "_se", "_worst", "All"])
        if feature_group == "All":
            cols_hm = X_raw.columns.tolist()[:15]
        else:
            cols_hm = [c for c in X_raw.columns if feature_group in c]

        corr = X_raw[cols_hm].corr()
        fig = px.imshow(corr, color_continuous_scale="RdBu_r", zmin=-1, zmax=1,
                        text_auto=".2f", title=f"Correlation Matrix — {feature_group} features",
                        template="plotly_dark")
        fig.update_layout(height=500, paper_bgcolor=C['bg'], font_color=C['text'])
        st.plotly_chart(fig, use_container_width=True)

    # ── Tab 4: Feature Distributions ──────────────────────────────────────────
    with tab4:
        st.markdown('<div class="sec-title">Feature Distribution by Diagnosis</div>', unsafe_allow_html=True)
        feat_sel = st.selectbox("Select feature", X_raw.columns.tolist())
        df_plot = X_raw.copy()
        df_plot['diagnosis'] = y_raw
        fig = go.Figure()
        fig.add_trace(go.Histogram(x=df_plot[df_plot['diagnosis']==0][feat_sel],
                                   name='Benign', marker_color=C['benign'], opacity=0.7, nbinsx=40))
        fig.add_trace(go.Histogram(x=df_plot[df_plot['diagnosis']==1][feat_sel],
                                   name='Malignant', marker_color=C['malignant'], opacity=0.7, nbinsx=40))
        fig.update_layout(barmode='overlay', height=380, title=f"Distribution of {feat_sel}",
                          xaxis_title=feat_sel, yaxis_title="Count", **PLOT_LAYOUT)
        st.plotly_chart(fig, use_container_width=True)

# ══════════════════════════════════════════════════════════════════════════════
# PAGE: MODEL COMPARISON
# ══════════════════════════════════════════════════════════════════════════════
elif page == "🤖 Model Comparison":
    st.markdown("# 🤖 Model Comparison")
    st.divider()

    comp_df = pd.DataFrame([{
        'Model': k, 'Accuracy': v['Accuracy'], 'Precision': v['Precision'],
        'Recall': v['Recall'], 'F1-Score': v['F1-Score'],
        'ROC-AUC': v['ROC-AUC'], 'CV_Mean': v['CV_Mean'], 'CV_Std': v['CV_Std']
    } for k, v in results.items()]).sort_values('F1-Score', ascending=False).reset_index(drop=True)

    # Metric cards for best model
    st.markdown(f'<div class="sec-title">Best Model: {best_name}</div>', unsafe_allow_html=True)
    mc1,mc2,mc3,mc4,mc5 = st.columns(5)
    kpi(mc1, f"{best['Accuracy']:.4f}",  "Accuracy",  C['primary'])
    kpi(mc2, f"{best['Precision']:.4f}", "Precision", C['benign'])
    kpi(mc3, f"{best['Recall']:.4f}",    "Recall",    C['malignant'])
    kpi(mc4, f"{best['F1-Score']:.4f}",  "F1-Score",  C['primary'])
    kpi(mc5, f"{best['ROC-AUC']:.4f}",   "ROC-AUC",   C['primary'])

    st.divider()

    tab_a, tab_b, tab_c, tab_d = st.tabs(["Accuracy", "All Metrics", "ROC Curves", "Confusion Matrix"])

    # ── Accuracy ───────────────────────────────────────────────────────────────
    with tab_a:
        fig = make_subplots(rows=1, cols=2, subplot_titles=("Accuracy", "Cross-Validation Mean ± Std"))
        colors_bar = [C['malignant'] if m == best_name else C['primary'] for m in comp_df['Model']]
        fig.add_trace(go.Bar(x=comp_df['Accuracy'], y=comp_df['Model'], orientation='h',
                             marker_color=colors_bar, text=[f"{v:.4f}" for v in comp_df['Accuracy']],
                             textposition='outside', name='Accuracy'), row=1, col=1)
        fig.add_trace(go.Bar(x=comp_df['CV_Mean'], y=comp_df['Model'], orientation='h',
                             error_x=dict(type='data', array=comp_df['CV_Std']),
                             marker_color=colors_bar, name='CV Mean'), row=1, col=2)
        fig.add_vline(x=best['Accuracy'], line_dash='dash', line_color=C['malignant'], row=1, col=1)
        fig.add_vline(x=best['CV_Mean'],  line_dash='dash', line_color=C['malignant'], row=1, col=2)
        fig.update_xaxes(range=[0.80, 1.0])
        fig.update_layout(height=380, showlegend=False, **PLOT_LAYOUT)
        st.plotly_chart(fig, use_container_width=True)

    # ── All Metrics ────────────────────────────────────────────────────────────
    with tab_b:
        metric_sel = st.multiselect("Metrics", ['Accuracy','Precision','Recall','F1-Score','ROC-AUC'],
                                     default=['Accuracy','Precision','Recall','F1-Score'])
        fig = go.Figure()
        metric_colors = {'Accuracy':C['primary'],'Precision':'#f39c12','Recall':C['benign'],'F1-Score':C['malignant'],'ROC-AUC':'#9b59b6'}
        for m in metric_sel:
            fig.add_trace(go.Bar(x=comp_df[m], y=comp_df['Model'], orientation='h',
                                 name=m, marker_color=metric_colors.get(m, C['primary']), opacity=0.85))
        fig.update_layout(barmode='group', height=420, xaxis_range=[0.80, 1.0], **PLOT_LAYOUT)
        st.plotly_chart(fig, use_container_width=True)

    # ── ROC Curves ─────────────────────────────────────────────────────────────
    with tab_c:
        fig = go.Figure()
        palette = [C['primary'], C['benign'], C['malignant'], '#f39c12', '#9b59b6', '#1abc9c', '#e67e22']
        for i, (name, res) in enumerate(results.items()):
            fpr, tpr, _ = roc_curve(y_test, res['y_proba'])
            lw = 3 if name == best_name else 1.5
            fig.add_trace(go.Scatter(x=fpr, y=tpr, mode='lines', name=f"{name} (AUC={res['ROC-AUC']:.4f})",
                                     line=dict(color=palette[i % len(palette)], width=lw)))
        fig.add_trace(go.Scatter(x=[0,1], y=[0,1], mode='lines', name='Random',
                                 line=dict(color=C['muted'], dash='dash', width=1)))
        fig.update_layout(height=460, xaxis_title='False Positive Rate',
                          yaxis_title='True Positive Rate', title='ROC Curves — All Models', **PLOT_LAYOUT)
        st.plotly_chart(fig, use_container_width=True)

    # ── Confusion Matrix ───────────────────────────────────────────────────────
    with tab_d:
        model_sel = st.selectbox("Select model", list(results.keys()),
                                  index=list(results.keys()).index(best_name))
        cm = confusion_matrix(y_test, results[model_sel]['y_pred'])
        fig = px.imshow(cm, text_auto=True, color_continuous_scale='Blues',
                        x=['Benign (0)', 'Malignant (1)'], y=['Benign (0)', 'Malignant (1)'],
                        labels=dict(x='Predicted', y='Actual'),
                        title=f"Confusion Matrix — {model_sel}")
        fig.update_layout(height=380, paper_bgcolor=C['bg'], font_color=C['text'])
        st.plotly_chart(fig, use_container_width=True)

        tn, fp, fn, tp = cm.ravel()
        cc1,cc2,cc3,cc4 = st.columns(4)
        kpi(cc1, tn, "True Negative",  C['benign'])
        kpi(cc2, tp, "True Positive",  C['benign'])
        kpi(cc3, fp, "False Positive", C['malignant'])
        kpi(cc4, fn, "False Negative (critical!)", C['malignant'])

# ══════════════════════════════════════════════════════════════════════════════
# PAGE: PREDICT
# ══════════════════════════════════════════════════════════════════════════════
elif page == "🎯 Predict":
    st.markdown("# 🎯 Tumor Diagnosis Predictor")
    st.markdown(f"Enter cell measurement values below. The **{best_name}** model will predict whether the tumor is **Benign** or **Malignant**.")
    st.divider()

    st.markdown('<div class="warn-box">⚠️ <strong>Medical Disclaimer:</strong> This tool is for educational and research purposes only. It is not a substitute for professional medical diagnosis. Always consult a qualified healthcare provider.</div>', unsafe_allow_html=True)
    st.divider()

    # Compute stats for smart defaults
    stats = X_raw.describe()

    # Group features into _mean, _se, _worst
    mean_feats  = [c for c in ORIG_FEATURES if c.endswith('_mean')]
    se_feats    = [c for c in ORIG_FEATURES if c.endswith('_se')]
    worst_feats = [c for c in ORIG_FEATURES if c.endswith('_worst')]

    st.markdown("**Quick fill:**")
    qc1, qc2, qc3 = st.columns(3)
    fill_benign   = qc1.button("Fill with Benign median values",   use_container_width=True)
    fill_malign   = qc2.button("Fill with Malignant median values", use_container_width=True)
    fill_reset    = qc3.button("Reset to overall median",           use_container_width=True)

    # Compute medians
    X_b_med = X_raw[y_raw == 0].median()
    X_m_med = X_raw[y_raw == 1].median()
    X_med   = X_raw.median()

    def get_default(col):
        if fill_benign:  return float(X_b_med[col])
        if fill_malign:  return float(X_m_med[col])
        return float(X_med[col])

    input_vals = {}

    st.markdown('<div class="sec-title">📐 Mean Features</div>', unsafe_allow_html=True)
    cols_m = st.columns(5)
    for i, feat in enumerate(mean_feats):
        with cols_m[i % 5]:
            input_vals[feat] = st.number_input(
                feat, min_value=float(stats[feat]['min']),
                max_value=float(stats[feat]['max']),
                value=get_default(feat),
                format="%.4f", key=f"inp_{feat}"
            )

    st.markdown('<div class="sec-title">📏 Standard Error Features</div>', unsafe_allow_html=True)
    cols_s = st.columns(5)
    for i, feat in enumerate(se_feats):
        with cols_s[i % 5]:
            input_vals[feat] = st.number_input(
                feat, min_value=float(stats[feat]['min']),
                max_value=float(stats[feat]['max']),
                value=get_default(feat),
                format="%.4f", key=f"inp_{feat}"
            )

    st.markdown('<div class="sec-title">⚠️ Worst Features</div>', unsafe_allow_html=True)
    cols_w = st.columns(5)
    for i, feat in enumerate(worst_feats):
        with cols_w[i % 5]:
            input_vals[feat] = st.number_input(
                feat, min_value=float(stats[feat]['min']),
                max_value=float(stats[feat]['max']),
                value=get_default(feat),
                format="%.4f", key=f"inp_{feat}"
            )

    st.divider()

    if st.button("🔬 Run Prediction", use_container_width=True, type="primary"):
        input_df = pd.DataFrame([input_vals])[ORIG_FEATURES]
        pred        = final_pipeline.predict(input_df)[0]
        proba       = final_pipeline.predict_proba(input_df)[0]
        prob_benign = proba[0]
        prob_malig  = proba[1]

        left, right = st.columns([1, 1], gap="large")

        with left:
            if pred == 0:
                st.markdown(f"""
                <div class="result-box benign-box">
                    <div class="result-label" style="color:{C['benign']}">✅ BENIGN</div>
                    <div class="result-sub">The model predicts this tumor is <strong>benign (non-cancerous)</strong>.</div>
                    <div style="font-size:1.5rem; margin-top:12px; color:{C['benign']}">
                        Confidence: {prob_benign*100:.1f}%
                    </div>
                </div>""", unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div class="result-box malign-box">
                    <div class="result-label" style="color:{C['malignant']}">⚠️ MALIGNANT</div>
                    <div class="result-sub">The model predicts this tumor is <strong>malignant (potentially cancerous)</strong>.</div>
                    <div style="font-size:1.5rem; margin-top:12px; color:{C['malignant']}">
                        Confidence: {prob_malig*100:.1f}%
                    </div>
                </div>""", unsafe_allow_html=True)

        with right:
            fig = go.Figure(go.Bar(
                x=['Benign', 'Malignant'],
                y=[prob_benign * 100, prob_malig * 100],
                marker_color=[C['benign'], C['malignant']],
                text=[f"{prob_benign*100:.1f}%", f"{prob_malig*100:.1f}%"],
                textposition='outside', width=0.4,
            ))
            fig.update_layout(
                height=300, yaxis_range=[0, 115],
                yaxis_title="Probability (%)",
                title="Prediction Probability",
                **PLOT_LAYOUT
            )
            st.plotly_chart(fig, use_container_width=True)

        # Engineered feature values
        st.divider()
        st.markdown('<div class="sec-title">🧬 Engineered Features (auto-calculated)</div>', unsafe_allow_html=True)
        fe_transformer = FeatureEngineeringTransformer()
        fe_transformer.fit(X_raw)
        eng = fe_transformer.transform(input_df)
        e1, e2, e3 = st.columns(3)
        e1.metric("radius_area_ratio",        f"{eng['radius_area_ratio'].values[0]:.4f}")
        e2.metric("perimeter_area_ratio",     f"{eng['perimeter_area_ratio'].values[0]:.4f}")
        e3.metric("concavity_points_product", f"{eng['concavity_points_product'].values[0]:.4f}")

        # Risk flag
        if pred == 1 and prob_malig > 0.8:
            st.error("🚨 **High-confidence malignant prediction.** Please consult a medical professional immediately.")
        elif pred == 1 and prob_malig <= 0.8:
            st.warning("⚠️ **Malignant prediction with moderate confidence.** Further clinical evaluation recommended.")
        else:
            st.success("✅ Benign prediction. Regular check-ups are still recommended.")
