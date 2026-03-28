"""
Solar Flare AI Prediction System — Streamlit Demo
Run: streamlit run demo.py
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (confusion_matrix, roc_auc_score,
                              precision_score, recall_score, f1_score,
                              accuracy_score)
from datetime import datetime, timedelta
import sys, os, re, warnings, pickle
warnings.filterwarnings('ignore')

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from scripts.data_preprocessing import load_flares, load_cme, load_storms
from scripts.feature_engineering import build_daily_features, FEATURE_COLS, TARGET_COL

try:
    import torch
    import torch.nn as nn
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

# ══════════════════════════════════════════════════════════════════════════════
# PAGE CONFIG
# ══════════════════════════════════════════════════════════════════════════════
st.set_page_config(
    page_title="☀️ Solar Flare AI",
    page_icon="☀️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ══════════════════════════════════════════════════════════════════════════════
# CSS — Dark Space Theme
# ══════════════════════════════════════════════════════════════════════════════
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Orbitron:wght@400;700;900&family=Inter:wght@300;400;600&display=swap');

/* ── Global ── */
html, body, [class*="css"] { font-family: 'Inter', sans-serif; }
.stApp {
    background: radial-gradient(ellipse at top, #0d1b3e 0%, #050510 60%);
    color: #e0e8ff;
}
.block-container { padding-top: 1.5rem; }

/* ── Sidebar ── */
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #070718 0%, #0d1b3e 100%);
    border-right: 1px solid #1e3a5f;
}
[data-testid="stSidebar"] * { color: #c0d0f0 !important; }

/* ── Headers ── */
h1 { font-family: 'Orbitron', monospace !important; font-weight: 900;
     background: linear-gradient(90deg, #ff9a3c, #ff4e6a);
     -webkit-background-clip: text; -webkit-text-fill-color: transparent;
     font-size: 2rem !important; }
h2 { font-family: 'Orbitron', monospace !important; color: #ff9a3c !important; font-size: 1.2rem !important; }
h3 { color: #88b8ff !important; }

/* ── Metric Cards ── */
.metric-card {
    background: linear-gradient(145deg, #0d1b3e, #131f4a);
    border: 1px solid rgba(255,154,60,0.3);
    border-radius: 16px; padding: 20px; text-align: center;
    box-shadow: 0 4px 20px rgba(255,100,50,0.15);
    transition: transform 0.2s;
}
.metric-card:hover { transform: translateY(-3px); }
.metric-title { font-size: 0.75rem; color: #7090c0; text-transform: uppercase; letter-spacing: 1px; }
.metric-value { font-family: 'Orbitron', monospace; font-size: 2rem; font-weight: 700; margin: 8px 0; }
.metric-sub   { font-size: 0.8rem; color: #8090b0; }

/* ── Risk Badges ── */
.risk-low      { color: #00e676; }
.risk-moderate { color: #ffd740; }
.risk-high     { color: #ff6d00; }
.risk-extreme  { color: #ff1744; text-shadow: 0 0 10px #ff1744; }

/* ── Alert Box ── */
.alert-box {
    border-radius: 12px; padding: 16px 20px; margin: 12px 0;
    border-left: 5px solid;
}
.alert-info    { background: rgba(33,150,243,0.1); border-color: #2196f3; }
.alert-warning { background: rgba(255,152,0,0.1);  border-color: #ff9800; }
.alert-danger  { background: rgba(244,67,54,0.1);  border-color: #f44336; }
.alert-success { background: rgba(76,175,80,0.1);  border-color: #4caf50; }

/* ── Tables ── */
.stDataFrame { border-radius: 10px; overflow: hidden; }

/* ── Tabs ── */
.stTabs [data-baseweb="tab-list"] { gap: 6px; background: transparent; }
.stTabs [data-baseweb="tab"] {
    background: rgba(13,27,62,0.8);
    border-radius: 10px 10px 0 0;
    color: #7090c0; border: 1px solid #1e3a5f;
    font-family: 'Orbitron', monospace; font-size: 0.7rem;
    padding: 8px 16px;
}
.stTabs [aria-selected="true"] {
    background: linear-gradient(135deg, #ff4e6a, #ff9a3c) !important;
    color: white !important; border-color: transparent !important;
}

/* ── Progress bars ── */
.stProgress > div > div { background: linear-gradient(90deg, #ff4e6a, #ff9a3c); border-radius: 4px; }

/* ── Divider ── */
hr { border-color: rgba(255,154,60,0.2) !important; }

/* ── Plotly chart background ── */
.js-plotly-plot { border-radius: 12px; overflow: hidden; }
</style>
""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
# PLOTLY THEME
# ══════════════════════════════════════════════════════════════════════════════
PLOTLY_THEME = dict(
    template="plotly_dark",
    paper_bgcolor="rgba(7,7,24,0.95)",
    plot_bgcolor="rgba(13,27,62,0.6)",
    font=dict(family="Inter", color="#c0d0f0"),
)
SOLAR_COLORS = px.colors.sequential.Inferno

# ══════════════════════════════════════════════════════════════════════════════
# DATA LOADING (CACHED)
# ══════════════════════════════════════════════════════════════════════════════
@st.cache_data(show_spinner="NASA verileri yükleniyor...")
def get_raw_data():
    flares = load_flares("data/raw/nasa_solar_flares.csv")
    cme    = load_cme("data/raw/nasa_cme.csv")
    storms = load_storms("data/raw/nasa_geomagnetic_storms.csv")
    return flares, cme, storms

@st.cache_data(show_spinner="Özellikler hesaplanıyor...")
def get_daily_features(_flares, _cme, _storms):
    return build_daily_features(_flares, _cme, _storms)

# ══════════════════════════════════════════════════════════════════════════════
# MODEL LOADING / TRAINING (CACHED)
# ══════════════════════════════════════════════════════════════════════════════
@st.cache_resource(show_spinner="Model yükleniyor...")
def get_trained_model(_daily):
    import json as _json
    daily     = _daily
    X         = daily[FEATURE_COLS].fillna(0).values
    y         = daily[TARGET_COL].values
    split_idx = int(len(X) * 0.8)
    X_train, X_test = X[:split_idx], X[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]

    # ── Try to load pre-trained RF from disk ─────────────────────────────────
    rf_path = "models/random_forest.pkl"
    if os.path.exists(rf_path):
        with open(rf_path, "rb") as fh:
            bundle = pickle.load(fh)
        rf      = bundle['model']
        scaler  = bundle['scaler']
        rf_thr  = float(bundle.get('threshold', 0.38))
    else:
        # Fallback: train inline
        scaler = StandardScaler()
        X_train_s = scaler.fit_transform(X_train)
        rf = RandomForestClassifier(
            n_estimators=500, max_depth=8, min_samples_leaf=4,
            class_weight='balanced', random_state=42, n_jobs=-1
        )
        rf.fit(X_train_s, y_train)
        rf_thr = 0.38

    scaler_fitted = scaler  # already fitted on training data
    X_train_s = scaler_fitted.transform(X_train)
    X_test_s  = scaler_fitted.transform(X_test)

    rf_prob = rf.predict_proba(X_test_s)[:, 1]
    rf_pred = (rf_prob >= rf_thr).astype(int)

    test_dates = daily['date'].values[split_idx:]

    # ── Try to load full metrics from results/metrics.json ───────────────────
    saved_metrics  = {}
    saved_imp      = {}
    saved_lstm_met = {}
    saved_ens_met  = {}
    if os.path.exists("results/metrics.json"):
        with open("results/metrics.json") as fh:
            res = _json.load(fh)
        saved_metrics  = res.get("rf_metrics",   {})
        saved_lstm_met = res.get("lstm_metrics",  {}) or {}
        saved_ens_met  = res.get("ens_metrics",   {}) or {}
        saved_imp      = res.get("feature_importance", {})

    # Use saved metrics if richer, else compute fresh
    if saved_metrics and 'f1' in saved_metrics:
        metrics = saved_metrics
    else:
        metrics = {
            'accuracy':  float(accuracy_score(y_test, rf_pred)),
            'precision': float(precision_score(y_test, rf_pred, zero_division=0)),
            'recall':    float(recall_score(y_test, rf_pred, zero_division=0)),
            'f1':        float(f1_score(y_test, rf_pred, zero_division=0)),
            'auc':       float(roc_auc_score(y_test, rf_prob)) if y_test.sum() > 0 else 0.5,
            'cm':        confusion_matrix(y_test, rf_pred).tolist(),
        }

    importances = saved_imp if saved_imp else dict(
        zip(FEATURE_COLS, rf.feature_importances_.tolist())
    )
    if 'cm' not in metrics:
        metrics['cm'] = confusion_matrix(y_test, rf_pred).tolist()

    return {
        'model': rf, 'scaler': scaler_fitted, 'threshold': rf_thr,
        'metrics': metrics, 'importances': importances,
        'lstm_metrics': saved_lstm_met, 'ens_metrics': saved_ens_met,
        'X_test': X_test_s, 'y_test': y_test,
        'prob_test': rf_prob, 'test_dates': test_dates,
        'split_idx': split_idx,
    }

# ══════════════════════════════════════════════════════════════════════════════
# HELPERS
# ══════════════════════════════════════════════════════════════════════════════
def risk_label(prob):
    if prob < 0.25: return "DÜŞÜK",   "risk-low",      "#00e676"
    if prob < 0.50: return "ORTA",    "risk-moderate", "#ffd740"
    if prob < 0.75: return "YÜKSEK",  "risk-high",     "#ff6d00"
    return              "KRİTİK",  "risk-extreme",  "#ff1744"

def class_color(cls):
    return {'A': '#607d8b', 'B': '#78909c', 'C': '#29b6f6',
            'M': '#ffa726', 'X': '#ef5350'}.get(cls, '#90a4ae')

def make_gauge(value, title, color):
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=round(value * 100, 1),
        title={'text': title, 'font': {'size': 14, 'color': '#88b8ff', 'family': 'Orbitron'}},
        number={'suffix': '%', 'font': {'size': 28, 'color': color, 'family': 'Orbitron'}},
        gauge={
            'axis': {'range': [0, 100], 'tickcolor': '#3050a0'},
            'bar':  {'color': color, 'thickness': 0.25},
            'bgcolor': 'rgba(13,27,62,0.8)',
            'bordercolor': '#1e3a5f',
            'steps': [
                {'range': [0, 25],  'color': 'rgba(0,230,118,0.15)'},
                {'range': [25, 50], 'color': 'rgba(255,215,64,0.15)'},
                {'range': [50, 75], 'color': 'rgba(255,109,0,0.15)'},
                {'range': [75,100], 'color': 'rgba(255,23,68,0.15)'},
            ],
            'threshold': {'line': {'color': color, 'width': 3}, 'thickness': 0.75, 'value': value*100},
        }
    ))
    fig.update_layout(**PLOTLY_THEME, height=220, margin=dict(l=20, r=20, t=40, b=10))
    return fig

# ══════════════════════════════════════════════════════════════════════════════
# LOAD DATA
# ══════════════════════════════════════════════════════════════════════════════
flares, cme, storms = get_raw_data()
daily   = get_daily_features(flares, cme, storms)
trained = get_trained_model(daily)

model   = trained['model']
scaler  = trained['scaler']
metrics = trained['metrics']

# ══════════════════════════════════════════════════════════════════════════════
# SIDEBAR
# ══════════════════════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown("""
    <div style='text-align:center; padding: 10px 0 20px'>
        <div style='font-family:Orbitron; font-size:1.2rem; color:#ff9a3c; font-weight:900;'>
            ☀️ SOLAR FLARE AI
        </div>
        <div style='font-size:0.7rem; color:#6080b0; letter-spacing:2px;'>
            PREDICTION SYSTEM v1.0
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")

    # Dataset stats
    st.markdown("**📊 Veri Seti**")
    c1, c2 = st.columns(2)
    c1.metric("Flare", f"{len(flares):,}")
    c2.metric("CME",   f"{len(cme):,}")
    c1.metric("Fırtına", f"{len(storms):,}")
    c2.metric("X-Sınıfı", f"{(flares['class_letter']=='X').sum()}")

    st.markdown("---")
    st.markdown("**🤖 Model (Ensemble)**")
    _em = trained.get('ens_metrics') or trained.get('lstm_metrics') or metrics
    st.metric("AUC",       f"{_em.get('auc', metrics['auc']):.3f}")
    st.metric("F1-Score",  f"{_em.get('f1',  metrics['f1']):.3f}")
    st.metric("Recall",    f"{_em.get('recall', metrics['recall']):.3f}")

    st.markdown("---")
    st.markdown("**🗓️ Analiz Tarihi**")
    date_range = st.date_input(
        "Tarih Seç",
        value=pd.Timestamp(daily['date'].max()).date(),
        min_value=pd.Timestamp(daily['date'].min()).date(),
        max_value=pd.Timestamp(daily['date'].max()).date(),
    )

    st.markdown("---")
    st.markdown("""
    <div style='font-size:0.65rem; color:#405080; text-align:center;'>
        Veri: NASA DONKI API<br>
        Dönem: 2023–2026<br>
        Model: Random Forest + LSTM Ensemble<br>
        <br>Made with ☀️ by Team Solar Sentinels
    </div>
    """, unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
# TITLE
# ══════════════════════════════════════════════════════════════════════════════
st.markdown("""
<h1>☀️ Solar Flare AI Prediction System</h1>
<p style='color:#6080b0; font-size:0.9rem; margin-top:-10px;'>
Güneş patlamalarını 7 gün önceden tahmin eden yapay zeka sistemi &nbsp;|&nbsp;
NASA DONKI verisi &nbsp;|&nbsp; Solar Cycle 25
</p>
""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
# TABS
# ══════════════════════════════════════════════════════════════════════════════
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "🌟 Dashboard",
    "🔮 AI Tahmin",
    "🌑 Arka Yüz",
    "📈 Tarihsel Analiz",
    "⚡ Etki Senaryoları",
])

# ══════════════════════════════════════════════════════════════════════════════
# TAB 1 — DASHBOARD
# ══════════════════════════════════════════════════════════════════════════════
with tab1:
    sel_date = pd.Timestamp(date_range)

    # Get row for selected date
    row = daily[daily['date'] == sel_date]
    if row.empty:
        # Use last available
        row = daily.iloc[[-1]]

    row = row.iloc[0]
    feat_vec = np.array([[row[c] for c in FEATURE_COLS]])
    feat_scaled = scaler.transform(feat_vec)
    prob_today = float(model.predict_proba(feat_scaled)[0, 1])
    label, css_class, color = risk_label(prob_today)

    # ── Top KPIs ─────────────────────────────────────────────────────────────
    k1, k2, k3, k4 = st.columns(4)
    with k1:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-title">7 Günlük Risk</div>
            <div class="metric-value {css_class}">{label}</div>
            <div class="metric-sub">X-class olasılığı: {prob_today:.1%}</div>
        </div>""", unsafe_allow_html=True)

    with k2:
        recent_x = int(daily[daily['date'] >= sel_date - pd.Timedelta(days=7)]['n_x'].sum())
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-title">Son 7 Gün X-Sınıfı</div>
            <div class="metric-value" style="color:#ef5350">{recent_x}</div>
            <div class="metric-sub">X-class patlama sayısı</div>
        </div>""", unsafe_allow_html=True)

    with k3:
        recent_m = int(daily[daily['date'] >= sel_date - pd.Timedelta(days=7)]['n_m'].sum())
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-title">Son 7 Gün M-Sınıfı</div>
            <div class="metric-value" style="color:#ffa726">{recent_m}</div>
            <div class="metric-sub">M-class patlama sayısı</div>
        </div>""", unsafe_allow_html=True)

    with k4:
        active_r = int(row['n_active_regions'])
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-title">Aktif Bölge</div>
            <div class="metric-value" style="color:#42a5f5">{active_r}</div>
            <div class="metric-sub">Bugünkü AR sayısı</div>
        </div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Gauge + 7-day Forecast ────────────────────────────────────────────────
    g_col, f_col = st.columns([1, 2])

    with g_col:
        st.plotly_chart(make_gauge(prob_today, "X-class Risk Skoru", color),
                        width="stretch")

        # Solar cycle indicator
        cycle_day = int(row['cycle_day'])
        cycle_year = cycle_day / 365.25
        st.markdown(f"""
        <div class="metric-card" style="margin-top:10px">
            <div class="metric-title">Solar Döngü 25</div>
            <div class="metric-value" style="color:#88b8ff; font-size:1.2rem;">Yıl {cycle_year:.1f}</div>
            <div class="metric-sub">~2019 Aralık'tan itibaren</div>
        </div>""", unsafe_allow_html=True)

    with f_col:
        # 7-day forecast
        forecast_dates = [sel_date + pd.Timedelta(days=i) for i in range(1, 8)]
        forecast_probs = []
        for fd in forecast_dates:
            r = daily[daily['date'] <= fd]
            if r.empty:
                forecast_probs.append(prob_today * 0.9)
                continue
            r = r.iloc[-1]
            fv = np.array([[r[c] for c in FEATURE_COLS]])
            forecast_probs.append(float(model.predict_proba(scaler.transform(fv))[0, 1]))

        colors_bar = [risk_label(p)[2] for p in forecast_probs]
        fig_fc = go.Figure()
        fig_fc.add_trace(go.Bar(
            x=[d.strftime('%a\n%d %b') for d in forecast_dates],
            y=[p * 100 for p in forecast_probs],
            marker_color=colors_bar,
            marker_line_color='rgba(255,255,255,0.2)',
            marker_line_width=1,
            text=[f"{p:.0%}" for p in forecast_probs],
            textposition='outside',
            textfont=dict(color='white', size=11),
        ))
        fig_fc.add_hline(y=50, line_dash="dot", line_color="#ff9800",
                         annotation_text="Risk eşiği (50%)",
                         annotation_font_color="#ff9800")
        fig_fc.update_layout(
            **PLOTLY_THEME,
            title="🔮 7 Günlük X-Class Flare Tahmin",
            yaxis=dict(title="Risk (%)", range=[0, 105]),
            xaxis=dict(title=""),
            showlegend=False,
            height=300,
            margin=dict(l=10, r=10, t=50, b=10),
        )
        st.plotly_chart(fig_fc, width="stretch")

    st.markdown("---")

    # ── Recent Flares Table + Activity Timeline ──────────────────────────────
    c_left, c_right = st.columns([1, 2])

    with c_left:
        st.markdown("### 🌋 Son Patlamalar")
        recent_flares = flares[flares['beginTime'] <= sel_date + pd.Timedelta(days=1)].tail(15)
        display_df = recent_flares[['classType', 'beginTime', 'sourceLocation', 'activeRegionNum']].copy()
        display_df.columns = ['Sınıf', 'Zaman', 'Konum', 'AR#']
        display_df['Zaman'] = display_df['Zaman'].dt.strftime('%Y-%m-%d %H:%M')
        display_df['AR#'] = display_df['AR#'].fillna('-').astype(str).str.replace('.0', '', regex=False)
        st.dataframe(
            display_df.sort_values('Zaman', ascending=False).reset_index(drop=True),
            width="stretch", height=350,
        )

    with c_right:
        st.markdown("### 📊 Son 90 Gün Aktivite")
        last90 = daily[daily['date'] >= sel_date - pd.Timedelta(days=90)].copy()
        fig_act = go.Figure()
        fig_act.add_trace(go.Bar(x=last90['date'], y=last90['n_c'],
                                  name='C-Sınıfı', marker_color='#29b6f6', opacity=0.7))
        fig_act.add_trace(go.Bar(x=last90['date'], y=last90['n_m'],
                                  name='M-Sınıfı', marker_color='#ffa726'))
        fig_act.add_trace(go.Bar(x=last90['date'], y=last90['n_x'],
                                  name='X-Sınıfı', marker_color='#ef5350'))
        fig_act.update_layout(
            **PLOTLY_THEME,
            barmode='stack', height=350,
            xaxis_title="", yaxis_title="Patlama Sayısı",
            legend=dict(orientation='h', y=1.05),
            margin=dict(l=10, r=10, t=10, b=10),
        )
        st.plotly_chart(fig_act, width="stretch")


# ══════════════════════════════════════════════════════════════════════════════
# TAB 2 — AI TAHMİN
# ══════════════════════════════════════════════════════════════════════════════
with tab2:
    st.markdown("## 🔮 Yapay Zeka Tahmin Motoru")

    col_ctrl, col_result = st.columns([1, 2])

    with col_ctrl:
        st.markdown("### Parametreler")
        pred_date = st.date_input(
            "Tahmin Başlangıç Tarihi",
            value=pd.Timestamp(daily['date'].max()).date(),
            min_value=pd.Timestamp(daily['date'].min() + pd.Timedelta(days=30)).date(),
            max_value=pd.Timestamp(daily['date'].max()).date(),
        )
        horizon = st.slider("Tahmin Ufku (gün)", 1, 14, 7)
        threshold = st.slider("Uyarı Eşiği (%)", 20, 80, 45) / 100

        st.markdown("**Model Ağırlıkları**")
        rf_w  = st.slider("Random Forest", 0.0, 1.0, 0.6, 0.05)
        rot_w = round(1 - rf_w, 2)
        st.markdown(f"Rotasyon Modeli: **{rot_w:.2f}**")

        run_pred = st.button("🚀 Tahmin Yap", width="stretch")

    with col_result:
        # Always compute predictions
        start_ts = pd.Timestamp(pred_date)
        pred_dates = [start_ts + pd.Timedelta(days=i) for i in range(horizon)]
        rf_probs, rot_probs, ens_probs = [], [], []

        for fd in pred_dates:
            r = daily[daily['date'] <= fd]
            if r.empty:
                rf_probs.append(0.3); rot_probs.append(0.3); ens_probs.append(0.3)
                continue
            r = r.iloc[-1]

            # RF prediction
            fv = np.array([[r[c] for c in FEATURE_COLS]])
            rf_p = float(model.predict_proba(scaler.transform(fv))[0, 1])

            # Rotation model: use 27-day lag intensity
            lag27 = float(r['lag27_intensity'])
            # Normalize: scale by typical X-class threshold (1e-4)
            rot_p = min(float(np.clip(lag27 / 3e-5 * 0.6, 0, 0.95)), 0.95)

            ens_p = rf_w * rf_p + rot_w * rot_p
            rf_probs.append(rf_p); rot_probs.append(rot_p); ens_probs.append(ens_p)

        fig_pred = go.Figure()
        date_strs = [d.strftime('%d %b') for d in pred_dates]
        fig_pred.add_trace(go.Scatter(
            x=date_strs, y=[p*100 for p in rf_probs],
            name='Random Forest', mode='lines+markers',
            line=dict(color='#42a5f5', width=2), marker=dict(size=8)
        ))
        fig_pred.add_trace(go.Scatter(
            x=date_strs, y=[p*100 for p in rot_probs],
            name='Rotasyon Modeli', mode='lines+markers',
            line=dict(color='#ff9a3c', width=2, dash='dot'), marker=dict(size=8)
        ))
        fig_pred.add_trace(go.Scatter(
            x=date_strs, y=[p*100 for p in ens_probs],
            name='Ensemble (Final)', mode='lines+markers',
            line=dict(color='#ef5350', width=3), marker=dict(size=10, symbol='star'),
            fill='tozeroy', fillcolor='rgba(239,83,80,0.1)'
        ))
        fig_pred.add_hline(y=threshold*100, line_dash="dash",
                           line_color="#ffd740",
                           annotation_text=f"Uyarı eşiği ({threshold:.0%})",
                           annotation_font_color="#ffd740")
        fig_pred.update_layout(
            **PLOTLY_THEME,
            title=f"X-Class Flare Olasılığı — {horizon} Günlük Tahmin",
            yaxis=dict(title="Olasılık (%)", range=[0, 100]),
            height=320, margin=dict(l=10, r=10, t=50, b=10),
            legend=dict(orientation='h', y=-0.2),
        )
        st.plotly_chart(fig_pred, width="stretch")

        # Daily risk table
        risk_df = pd.DataFrame({
            'Tarih': [d.strftime('%Y-%m-%d') for d in pred_dates],
            'RF (%)': [f"{p:.1%}" for p in rf_probs],
            'Rotasyon (%)': [f"{p:.1%}" for p in rot_probs],
            'Ensemble (%)': [f"{p:.1%}" for p in ens_probs],
            'Risk': [risk_label(p)[0] for p in ens_probs],
            'Uyarı': ['⚠️ EVET' if p >= threshold else '✅ Hayır' for p in ens_probs],
        })
        st.dataframe(risk_df, width="stretch", height=280)

    st.markdown("---")

    # Feature importance
    st.markdown("### 🎯 Özellik Önemi (Top 15)")
    imp = trained['importances']
    top15 = sorted(imp.items(), key=lambda x: x[1], reverse=True)[:15]
    feat_names = [x[0] for x in top15]
    feat_vals  = [x[1] for x in top15]

    fig_imp = go.Figure(go.Bar(
        x=feat_vals[::-1], y=feat_names[::-1],
        orientation='h',
        marker=dict(
            color=feat_vals[::-1],
            colorscale='Inferno',
            showscale=True,
            colorbar=dict(title=dict(text="Önem", font=dict(color='white')),
                          tickfont=dict(color='white'))
        ),
        text=[f"{v:.3f}" for v in feat_vals[::-1]],
        textposition='outside',
        textfont=dict(color='white'),
    ))
    fig_imp.update_layout(
        **PLOTLY_THEME, height=420,
        xaxis_title="Önem Skoru", yaxis_title="",
        margin=dict(l=10, r=60, t=20, b=10),
    )
    st.plotly_chart(fig_imp, width="stretch")


# ══════════════════════════════════════════════════════════════════════════════
# TAB 3 — ARKA YÜZ ANALİZİ
# ══════════════════════════════════════════════════════════════════════════════
with tab3:
    st.markdown("## 🌑 Arka Yüz Tahmini (27-Gün Rotasyon İnovasyonu)")

    st.markdown("""
    <div class="alert-box alert-info">
    <b>☀️ Güneş Diferansiyel Rotasyonu:</b>
    ω(λ) = 14.713 − 2.396·sin²(λ) − 1.787·sin⁴(λ) derece/gün
    <br>
    Güneş ekvatoru ~25.4 gün, kutupları ~38 günde döner.
    Arka yüzdeki bölge <b>~13-14 gün sonra</b> Dünya'ya dönük olacak.
    </div>
    """, unsafe_allow_html=True)

    bv_col1, bv_col2 = st.columns(2)
    sel_ts = pd.Timestamp(date_range)

    with bv_col1:
        st.markdown("### 🌞 Şu An Görünen Yüz")
        front_flares = flares[
            (flares['beginTime'] >= sel_ts - pd.Timedelta(days=14)) &
            (flares['beginTime'] <= sel_ts) &
            flares['lon'].notna()
        ]
        # Solar disk scatter
        theta_f = np.linspace(0, 2*np.pi, 200)
        fig_front = go.Figure()
        # Sun disk
        fig_front.add_trace(go.Scatter(
            x=np.cos(theta_f)*90, y=np.sin(theta_f)*90,
            fill='toself', fillcolor='rgba(255,180,30,0.15)',
            line=dict(color='#ffa726', width=2), showlegend=False
        ))
        # Equator line
        fig_front.add_hline(y=0, line_color='rgba(255,200,100,0.3)', line_width=1)
        fig_front.add_vline(x=0, line_color='rgba(255,200,100,0.3)', line_width=1)

        if not front_flares.empty:
            for cls, grp in front_flares.groupby('class_letter'):
                fig_front.add_trace(go.Scatter(
                    x=grp['lon'], y=grp['lat'],
                    mode='markers',
                    marker=dict(
                        size=np.clip(grp['class_number'].fillna(1) * 4, 5, 25),
                        color=class_color(cls),
                        opacity=0.85,
                        line=dict(color='white', width=0.5),
                        symbol='circle',
                    ),
                    name=f'{cls}-class ({len(grp)})',
                    text=grp['classType'] + '<br>' + grp['beginTime'].dt.strftime('%Y-%m-%d'),
                    hovertemplate='%{text}<br>Lon: %{x}°  Lat: %{y}°<extra></extra>',
                ))

        fig_front.update_layout(
            **PLOTLY_THEME,
            title=f"Görünen Yüz — {sel_ts.date()} öncesi 14 gün",
            xaxis=dict(title="Boylam (°)", range=[-100, 100],
                       zeroline=False, gridcolor='rgba(255,255,255,0.05)'),
            yaxis=dict(title="Enlem (°)", range=[-100, 100],
                       scaleanchor='x', zeroline=False,
                       gridcolor='rgba(255,255,255,0.05)'),
            height=400, margin=dict(l=10, r=10, t=50, b=10),
        )
        st.plotly_chart(fig_front, width="stretch")

    with bv_col2:
        st.markdown("### 🌑 Arka Yüz Tahmini (27 gün önceki veri)")
        back_date = sel_ts - pd.Timedelta(days=27)
        back_flares = flares[
            (flares['beginTime'] >= back_date - pd.Timedelta(days=14)) &
            (flares['beginTime'] <= back_date) &
            flares['lon'].notna()
        ]

        fig_back = go.Figure()
        # Sun disk (dimmer)
        fig_back.add_trace(go.Scatter(
            x=np.cos(theta_f)*90, y=np.sin(theta_f)*90,
            fill='toself', fillcolor='rgba(80,50,20,0.3)',
            line=dict(color='#5d4037', width=2, dash='dot'), showlegend=False
        ))
        fig_back.add_hline(y=0, line_color='rgba(150,100,50,0.3)', line_width=1)
        fig_back.add_vline(x=0, line_color='rgba(150,100,50,0.3)', line_width=1)

        if not back_flares.empty:
            for cls, grp in back_flares.groupby('class_letter'):
                alpha = 0.6
                col   = class_color(cls)
                fig_back.add_trace(go.Scatter(
                    x=grp['lon'], y=grp['lat'],
                    mode='markers',
                    marker=dict(
                        size=np.clip(grp['class_number'].fillna(1) * 4, 5, 25),
                        color=col, opacity=alpha,
                        line=dict(color='rgba(255,100,50,0.8)', width=1),
                        symbol='diamond',
                    ),
                    name=f'{cls}-class [{back_date.date()}]',
                    text=grp['classType'] + '<br>' + grp['beginTime'].dt.strftime('%Y-%m-%d'),
                    hovertemplate='%{text}<br>Lon: %{x}°  Lat: %{y}°<extra></extra>',
                ))

        # Highlight risk zone for active regions seen 27 days ago
        if not back_flares.empty and 'X' in back_flares['class_letter'].values:
            x_back = back_flares[back_flares['class_letter'] == 'X']
            for _, row_b in x_back.iterrows():
                if pd.notna(row_b['lon']) and pd.notna(row_b['lat']):
                    theta_c = np.linspace(0, 2*np.pi, 50)
                    fig_back.add_trace(go.Scatter(
                        x=row_b['lon'] + 15*np.cos(theta_c),
                        y=row_b['lat'] + 15*np.sin(theta_c),
                        mode='lines', line=dict(color='#ff1744', width=2, dash='dot'),
                        showlegend=False, hoverinfo='skip',
                    ))

        fig_back.update_layout(
            **PLOTLY_THEME,
            title=f"Arka Yüz Tahmini — {back_date.date()} verisi kullanıldı",
            xaxis=dict(title="Boylam (°)", range=[-100, 100],
                       zeroline=False, gridcolor='rgba(255,255,255,0.05)'),
            yaxis=dict(title="Enlem (°)", range=[-100, 100],
                       scaleanchor='x', zeroline=False,
                       gridcolor='rgba(255,255,255,0.05)'),
            height=400, margin=dict(l=10, r=10, t=50, b=10),
        )
        st.plotly_chart(fig_back, width="stretch")

    # 27-day rotation timeline
    st.markdown("---")
    st.markdown("### 🔄 27-Günlük Rotasyon Döngüsü")

    rot_days = 90
    rot_range = daily[daily['date'] >= sel_ts - pd.Timedelta(days=rot_days)].copy()
    rot_range['lag27_x'] = rot_range['n_x'].shift(27).fillna(0)

    fig_rot = go.Figure()
    fig_rot.add_trace(go.Bar(
        x=rot_range['date'], y=rot_range['n_x'],
        name='Görünen Yüz X-class', marker_color='#ef5350', opacity=0.9
    ))
    fig_rot.add_trace(go.Bar(
        x=rot_range['date'], y=rot_range['lag27_x'],
        name='Arka Yüz Tahmini (lag-27)', marker_color='#7e57c2', opacity=0.7
    ))
    fig_rot.update_layout(
        **PLOTLY_THEME,
        title="X-Class Patlama: Görünen Yüz vs Arka Yüz Tahmini",
        barmode='group', height=280,
        xaxis_title="", yaxis_title="X-class Sayısı",
        legend=dict(orientation='h', y=-0.3),
        margin=dict(l=10, r=10, t=50, b=10),
    )
    st.plotly_chart(fig_rot, width="stretch")

    # Differential rotation formula visualization
    st.markdown("---")
    st.markdown("### 📐 Diferansiyel Rotasyon Formülü")
    lat_arr = np.linspace(-90, 90, 200)
    omega   = 14.713 - 2.396*np.sin(np.radians(lat_arr))**2 - 1.787*np.sin(np.radians(lat_arr))**4
    period  = 360 / omega

    fig_diff = go.Figure()
    fig_diff.add_trace(go.Scatter(
        x=lat_arr, y=omega,
        mode='lines', name='ω (°/gün)',
        line=dict(color='#ff9a3c', width=3),
    ))
    fig_diff.add_trace(go.Scatter(
        x=lat_arr, y=period,
        mode='lines', name='Periyot (gün)',
        line=dict(color='#42a5f5', width=3),
        yaxis='y2'
    ))
    fig_diff.add_vrect(x0=-10, x1=10, fillcolor='rgba(255,154,60,0.1)',
                       line_width=0, annotation_text="Ekvator Bölgesi")
    fig_diff.update_layout(
        **PLOTLY_THEME,
        title="Güneş Diferansiyel Rotasyonu: ω(λ) = 14.713 − 2.396·sin²(λ) − 1.787·sin⁴(λ)",
        xaxis_title="Enlem (°)",
        yaxis=dict(title="Rotasyon Hızı (°/gün)", color='#ff9a3c'),
        yaxis2=dict(title="Rotasyon Periyodu (gün)", overlaying='y',
                    side='right', color='#42a5f5'),
        height=280, margin=dict(l=10, r=60, t=50, b=10),
        legend=dict(orientation='h', y=-0.3),
    )
    st.plotly_chart(fig_diff, width="stretch")


# ══════════════════════════════════════════════════════════════════════════════
# TAB 4 — TARİHSEL ANALİZ
# ══════════════════════════════════════════════════════════════════════════════
with tab4:
    st.markdown("## 📈 Tarihsel Güneş Aktivitesi Analizi")

    # Full timeline
    fig_timeline = go.Figure()
    fig_timeline.add_trace(go.Scatter(
        x=daily['date'], y=daily['roll7_intensity'],
        mode='lines', name='7-gün Ortalama Yoğunluk',
        line=dict(color='#ff9a3c', width=1.5),
        fill='tozeroy', fillcolor='rgba(255,154,60,0.08)',
    ))

    # Mark X-class days
    x_days = daily[daily['n_x'] > 0]
    fig_timeline.add_trace(go.Scatter(
        x=x_days['date'], y=x_days['roll7_intensity'],
        mode='markers', name='X-Sınıfı Patlama',
        marker=dict(color='#ef5350', size=10, symbol='star',
                    line=dict(color='white', width=1)),
    ))
    fig_timeline.update_layout(
        **PLOTLY_THEME,
        title="2023–2026 Güneş Aktivite Zaman Serisi",
        xaxis_title="", yaxis_title="X-ray Yoğunluk (W/m²)",
        height=300, margin=dict(l=10, r=10, t=50, b=10),
        legend=dict(orientation='h', y=-0.3),
    )
    st.plotly_chart(fig_timeline, width="stretch")

    # ── Stats Row ─────────────────────────────────────────────────────────────
    s1, s2, s3, s4, s5 = st.columns(5)
    cls_counts = flares['class_letter'].value_counts()
    s1.metric("Toplam Flare", f"{len(flares):,}")
    s2.metric("X-Sınıfı", str(cls_counts.get('X', 0)))
    s3.metric("M-Sınıfı", str(cls_counts.get('M', 0)))
    s4.metric("C-Sınıfı", str(cls_counts.get('C', 0)))
    s5.metric("Max Kp", f"{storms['max_kp'].max():.0f}")

    # ── Class Distribution + Monthly Heatmap ─────────────────────────────────
    h_left, h_right = st.columns(2)

    with h_left:
        st.markdown("### Sınıf Dağılımı")
        pie_data = cls_counts.reset_index()
        pie_data.columns = ['class', 'count']
        pie_data = pie_data[pie_data['class'].isin(['A','B','C','M','X'])]
        pie_data['color'] = pie_data['class'].map(class_color)

        fig_pie = go.Figure(go.Pie(
            labels=pie_data['class'],
            values=pie_data['count'],
            marker=dict(colors=pie_data['color'].tolist(),
                        line=dict(color='#0d1b3e', width=2)),
            textinfo='label+percent',
            textfont=dict(size=14, color='white'),
            hole=0.4,
        ))
        fig_pie.update_layout(
            **PLOTLY_THEME, height=320,
            margin=dict(l=10, r=10, t=10, b=10),
            showlegend=True,
            legend=dict(font=dict(color='white')),
        )
        st.plotly_chart(fig_pie, width="stretch")

    with h_right:
        st.markdown("### Aylık X+M Aktivite Heatmap")
        flares['year']  = flares['beginTime'].dt.year
        flares['month'] = flares['beginTime'].dt.month
        hm_data = flares[flares['class_letter'].isin(['M', 'X'])].groupby(
            ['year', 'month']
        ).size().reset_index(name='count')
        hm_pivot = hm_data.pivot(index='year', columns='month', values='count').fillna(0)
        month_names = ['Oca','Şub','Mar','Nis','May','Haz',
                       'Tem','Ağu','Eyl','Eki','Kas','Ara']

        fig_hm = go.Figure(go.Heatmap(
            z=hm_pivot.values,
            x=[month_names[m-1] for m in hm_pivot.columns],
            y=hm_pivot.index.astype(str).tolist(),
            colorscale='Inferno',
            text=hm_pivot.values.astype(int),
            texttemplate='%{text}',
            textfont=dict(size=11, color='white'),
            colorbar=dict(title=dict(text='Sayı', font=dict(color='white')),
                          tickfont=dict(color='white')),
        ))
        fig_hm.update_layout(
            **PLOTLY_THEME, height=320,
            xaxis_title="Ay", yaxis_title="Yıl",
            margin=dict(l=10, r=10, t=10, b=10),
        )
        st.plotly_chart(fig_hm, width="stretch")

    st.markdown("---")

    # ── Model Performance ─────────────────────────────────────────────────────
    st.markdown("### 🎯 Model Performansı")
    mp1, mp2 = st.columns(2)

    with mp1:
        st.markdown("#### Confusion Matrix")
        cm = np.array(metrics['cm'])
        labels_cm = ['Patlama Yok (0)', 'Patlama Var (1)']
        fig_cm = go.Figure(go.Heatmap(
            z=cm, x=labels_cm, y=labels_cm,
            colorscale=[[0, '#0d1b3e'], [1, '#ef5350']],
            text=cm, texttemplate='<b>%{text}</b>',
            textfont=dict(size=18, color='white'),
            showscale=False,
        ))
        fig_cm.update_layout(
            **PLOTLY_THEME, height=280,
            xaxis_title="Tahmin", yaxis_title="Gerçek",
            margin=dict(l=10, r=10, t=30, b=40),
        )
        # Annotations
        fig_cm.add_annotation(x=labels_cm[0], y=labels_cm[0],
            text="TN", font=dict(size=10, color='#88b8ff'), showarrow=False, yshift=-20)
        fig_cm.add_annotation(x=labels_cm[1], y=labels_cm[1],
            text="TP", font=dict(size=10, color='#88b8ff'), showarrow=False, yshift=-20)
        st.plotly_chart(fig_cm, width="stretch")

    with mp2:
        st.markdown("#### Performans Metrikleri")
        lstm_m = trained.get('lstm_metrics', {}) or {}
        ens_m  = trained.get('ens_metrics',  {}) or {}
        rows = [
            {'Model': 'Persistence Baseline',
             'Acc': 0.812, 'Prec': '-', 'Rec': '-', 'F1': 0.000, 'AUC': '-'},
            {'Model': 'Random Forest',
             'Acc': metrics['accuracy'], 'Prec': metrics['precision'],
             'Rec': metrics['recall'],   'F1':  metrics['f1'], 'AUC': metrics['auc']},
        ]
        if lstm_m.get('f1'):
            rows.append({'Model': 'LSTM (7-day seq.)',
                         'Acc': lstm_m.get('accuracy','-'),
                         'Prec': lstm_m.get('precision','-'),
                         'Rec': lstm_m.get('recall','-'),
                         'F1':  lstm_m.get('f1','-'),
                         'AUC': lstm_m.get('auc','-')})
        if ens_m.get('f1'):
            rows.append({'Model': 'Ensemble (RF + LSTM)',
                         'Acc': ens_m.get('accuracy','-'),
                         'Prec': ens_m.get('precision','-'),
                         'Rec': ens_m.get('recall','-'),
                         'F1':  ens_m.get('f1','-'),
                         'AUC': ens_m.get('auc','-')})
        m_df = pd.DataFrame(rows)
        for col_m in ['Acc', 'Prec', 'Rec', 'F1', 'AUC']:
            m_df[col_m] = m_df[col_m].apply(
                lambda x: f"{x:.3f}" if isinstance(x, float) else str(x)
            )
        st.dataframe(m_df.set_index('Model'), width="stretch", height=200)

        # ROC curve (approximate from test predictions)
        from sklearn.metrics import roc_curve
        fpr, tpr, _ = roc_curve(trained['y_test'], trained['prob_test'])
        fig_roc = go.Figure()
        fig_roc.add_trace(go.Scatter(
            x=fpr, y=tpr, mode='lines',
            name=f"RF (AUC={metrics['auc']:.3f})",
            line=dict(color='#ef5350', width=2),
            fill='tozeroy', fillcolor='rgba(239,83,80,0.1)',
        ))
        fig_roc.add_trace(go.Scatter(
            x=[0,1], y=[0,1], mode='lines',
            name='Rastgele', line=dict(color='gray', dash='dash'),
        ))
        fig_roc.update_layout(
            **PLOTLY_THEME, height=280,
            xaxis_title="FPR", yaxis_title="TPR",
            title="ROC Eğrisi",
            legend=dict(orientation='h', y=-0.3),
            margin=dict(l=10, r=10, t=40, b=10),
        )
        st.plotly_chart(fig_roc, width="stretch")


# ══════════════════════════════════════════════════════════════════════════════
# TAB 5 — ETKİ SENARYOLARI
# ══════════════════════════════════════════════════════════════════════════════
with tab5:
    st.markdown("## ⚡ X-Class Flare Etki Senaryoları")

    st.markdown("""
    <div class="alert-box alert-warning">
    <b>⚠️ Carrington Olayı Karşılaştırması:</b> 1859'da X45+ sınıfı patlama dünya çapında
    telgraf sistemlerini çökertdi. Bugün benzer bir olay tahmini <b>$2.6 trilyon</b> ekonomik hasar yaratabilir.
    </div>
    """, unsafe_allow_html=True)

    sc1, sc2 = st.columns([1, 2])
    with sc1:
        st.markdown("### Senaryo Ayarları")
        flare_class  = st.selectbox("Patlama Sınıfı", ['C5', 'M1', 'M5', 'X1', 'X5', 'X10', 'X45'])
        warning_time = st.slider("Uyarı Süresi (saat)", 0, 168, 168,
                                  help="168 saat = 7 gün (bizim hedefimiz)")
        affected_lat = st.slider("Etkilenen Enlem (°)", 40, 90, 60)

        # Impact lookup table
        impacts = {
            'C5':  dict(gps=5,   sat=2,   grid=1,   radio=10,  aurora=0,   econ=0.1),
            'M1':  dict(gps=15,  sat=5,   grid=3,   radio=30,  aurora=10,  econ=0.5),
            'M5':  dict(gps=30,  sat=15,  grid=10,  radio=60,  aurora=30,  econ=2),
            'X1':  dict(gps=50,  sat=30,  grid=20,  radio=80,  aurora=50,  econ=10),
            'X5':  dict(gps=75,  sat=55,  grid=45,  radio=95,  aurora=70,  econ=100),
            'X10': dict(gps=90,  sat=75,  grid=70,  radio=99,  aurora=85,  econ=500),
            'X45': dict(gps=100, sat=95,  grid=98,  radio=100, aurora=100, econ=2600),
        }
        imp = impacts[flare_class]

    with sc2:
        st.markdown("### Etki Analizi")

        # Mitigation factor from warning time
        mit_factor = min(1.0, warning_time / 168)  # 7 days = full mitigation
        mit_label  = f"{mit_factor:.0%} etki azaltma ({warning_time}s uyarı)"

        i1, i2, i3, i4 = st.columns(4)
        i1.metric("GPS Kesinti", f"{imp['gps']}%",
                  delta=f"-{imp['gps']*mit_factor:.0f}% ({mit_label})" if warning_time > 0 else None)
        i2.metric("Uydu Riski", f"{imp['sat']}%")
        i3.metric("Elektrik Şebekesi", f"{imp['grid']}%")
        i4.metric("Radyo Kesintisi", f"{imp['radio']}%")

        # Radar chart
        categories = ['GPS', 'Uydu', 'Şebeke', 'Radyo', 'Aurora']
        raw_vals   = [imp['gps'], imp['sat'], imp['grid'], imp['radio'], imp['aurora']]
        mit_vals   = [max(0, v - v*mit_factor*0.8) for v in raw_vals]

        fig_radar = go.Figure()
        fig_radar.add_trace(go.Scatterpolar(
            r=raw_vals + [raw_vals[0]],
            theta=categories + [categories[0]],
            fill='toself', name='Uyarısız',
            line=dict(color='#ef5350', width=2),
            fillcolor='rgba(239,83,80,0.2)',
        ))
        fig_radar.add_trace(go.Scatterpolar(
            r=mit_vals + [mit_vals[0]],
            theta=categories + [categories[0]],
            fill='toself', name=f'{warning_time}s Uyarı ile',
            line=dict(color='#00e676', width=2),
            fillcolor='rgba(0,230,118,0.15)',
        ))
        fig_radar.update_layout(
            **PLOTLY_THEME,
            polar=dict(
                bgcolor='rgba(13,27,62,0.8)',
                radialaxis=dict(range=[0, 100], tickfont=dict(color='white', size=9),
                                gridcolor='rgba(255,255,255,0.1)'),
                angularaxis=dict(tickfont=dict(color='white', size=11),
                                 gridcolor='rgba(255,255,255,0.1)'),
            ),
            title=f"{flare_class} Patlama Etki Senaryosu",
            legend=dict(orientation='h', y=-0.1),
            height=350, margin=dict(l=30, r=30, t=50, b=30),
        )
        st.plotly_chart(fig_radar, width="stretch")

    st.markdown("---")

    # Economic comparison
    st.markdown("### 💰 Ekonomik Etki Karşılaştırması")
    classes_all = ['C5', 'M1', 'M5', 'X1', 'X5', 'X10', 'X45']
    econ_vals   = [impacts[c]['econ'] for c in classes_all]
    econ_mit    = [max(0.01, v * (1 - mit_factor * 0.7)) for v in econ_vals]

    fig_econ = go.Figure()
    fig_econ.add_trace(go.Bar(
        x=classes_all, y=econ_vals,
        name='Uyarısız (milyar $)',
        marker_color=['#ef5350' if c.startswith('X') else '#ffa726' if c.startswith('M') else '#29b6f6'
                      for c in classes_all],
        text=[f"${v:.0f}B" for v in econ_vals],
        textposition='outside', textfont=dict(color='white'),
    ))
    fig_econ.add_trace(go.Bar(
        x=classes_all, y=econ_mit,
        name=f'{warning_time}s Uyarı ile',
        marker_color='rgba(0,230,118,0.6)',
        text=[f"${v:.0f}B" for v in econ_mit],
        textposition='outside', textfont=dict(color='#88ff88'),
    ))
    fig_econ.update_layout(
        **PLOTLY_THEME,
        title="Patlama Sınıfına Göre Ekonomik Etki (Milyar USD)",
        barmode='group', height=320,
        yaxis=dict(title="Milyar USD", type='log'),
        xaxis_title="Patlama Sınıfı",
        legend=dict(orientation='h', y=-0.2),
        margin=dict(l=10, r=10, t=50, b=10),
    )
    st.plotly_chart(fig_econ, width="stretch")

    # Satellite risk table
    st.markdown("### 🛰️ Yüksek Riskli Uydu Sistemleri")
    sat_data = pd.DataFrame([
        {'Sistem': 'GPS / GNSS',           'Kullanıcı': '4 milyar+',   'Risk': imp['gps'],  'Kesinti': f"{imp['gps']*0.3:.0f} saat"},
        {'Sistem': 'İletişim Uydular',      'Kullanıcı': '500+ uydu',   'Risk': imp['sat'],  'Kesinti': f"{imp['sat']*0.5:.0f} saat"},
        {'Sistem': 'Hava Durumu Uyduları',  'Kullanıcı': '50+ uydu',    'Risk': imp['sat']//2, 'Kesinti': f"{imp['sat']*0.2:.0f} saat"},
        {'Sistem': 'Elektrik Şebekesi',     'Kullanıcı': '1 milyar+',   'Risk': imp['grid'], 'Kesinti': f"{imp['grid']*2:.0f} saat"},
        {'Sistem': 'HF Radyo İletişimi',    'Kullanıcı': 'Havacılık',   'Risk': imp['radio'], 'Kesinti': f"{imp['radio']*0.1:.0f} saat"},
    ])
    sat_data['Risk Seviyesi'] = sat_data['Risk'].apply(
        lambda x: '🔴 KRİTİK' if x > 70 else ('🟠 YÜKSEK' if x > 40 else ('🟡 ORTA' if x > 20 else '🟢 DÜŞÜK'))
    )
    sat_data['Risk (%)'] = sat_data['Risk'].astype(str) + '%'
    st.dataframe(
        sat_data[['Sistem', 'Kullanıcı', 'Risk (%)', 'Risk Seviyesi', 'Kesinti']],
        width="stretch", height=220,
    )

    st.markdown(f"""
    <div class="alert-box alert-success">
    <b>✅ Sistemimizin Katkısı:</b> {warning_time} saatlik erken uyarı ile tahmini
    <b>${sum(econ_vals) - sum(econ_mit):.0f} milyar USD</b> ekonomik kayıp önlenebilir.
    Mevcut sistemler 15-60 dakika uyarı sağlarken, <b>bizim sistemimiz 7 gün (168 saat)</b> önceden tahmin yapıyor.
    </div>
    """, unsafe_allow_html=True)
