"""
Train RF + LSTM ensemble for Solar Flare Prediction.
Improvements over v1:
  - EMA features added (ema5/14_intensity, ema7_n_m, ema_trend)
  - LSTM trains on FULL train set (including val slice) → learns recent patterns
  - Threshold optimised on val slice for F-beta (beta=1.5, recall-weighted)
  - Ensemble threshold also optimised
Usage: python scripts/train_model.py
"""

import os, sys, json, warnings
warnings.filterwarnings('ignore')
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
import pickle
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (roc_auc_score, precision_score, recall_score,
                              f1_score, confusion_matrix, accuracy_score)

from scripts.data_preprocessing import load_flares, load_cme, load_storms
from scripts.feature_engineering import build_daily_features, FEATURE_COLS, TARGET_COL

try:
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader, TensorDataset
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    print("PyTorch not available – LSTM will be skipped.")


# ─── Helpers ──────────────────────────────────────────────────────────────────

def best_threshold(y_true, y_prob, beta=1.5):
    """Return threshold maximising F-beta on (y_true, y_prob)."""
    bt, bfb = 0.30, 0.0
    for t in np.arange(0.10, 0.75, 0.01):
        pred = (y_prob >= t).astype(int)
        if pred.sum() == 0:
            continue
        p = precision_score(y_true, pred, zero_division=0)
        r = recall_score(y_true, pred, zero_division=0)
        if p + r == 0:
            continue
        fb = (1 + beta**2) * p * r / (beta**2 * p + r)
        if fb > bfb:
            bfb, bt = fb, t
    return float(bt)


def evaluate(y_true, y_prob, thr, label=""):
    pred = (y_prob >= thr).astype(int)
    m = dict(
        accuracy  = float(accuracy_score(y_true, pred)),
        precision = float(precision_score(y_true, pred, zero_division=0)),
        recall    = float(recall_score(y_true, pred, zero_division=0)),
        f1        = float(f1_score(y_true, pred, zero_division=0)),
        auc       = float(roc_auc_score(y_true, y_prob)) if y_true.sum() > 0 else 0.5,
        threshold = float(thr),
        cm        = confusion_matrix(y_true, pred).tolist(),
    )
    if label:
        print(f"  {label:22s} | Acc={m['accuracy']:.3f} | Prec={m['precision']:.3f} | "
              f"Rec={m['recall']:.3f} | F1={m['f1']:.3f} | AUC={m['auc']:.3f} | thr={thr:.2f}")
    return m


# ─── LSTM ─────────────────────────────────────────────────────────────────────

class SolarLSTM(nn.Module):
    def __init__(self, input_size, hidden=64, layers=2, dropout=0.3):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden, layers,
                            batch_first=True, dropout=dropout)
        self.head = nn.Sequential(
            nn.Linear(hidden, 32), nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(32, 1), nn.Sigmoid()
        )

    def forward(self, x):
        out, _ = self.lstm(x)
        return self.head(out[:, -1, :]).squeeze(1)


def make_sequences(X, y, window=7):
    Xs, ys = [], []
    for i in range(window, len(X)):
        Xs.append(X[i-window:i])
        ys.append(y[i])
    return np.array(Xs, dtype=np.float32), np.array(ys, dtype=np.float32)


def train_lstm(X_train_s, y_train,
               val_frac=0.85, window=7, epochs=60, lr=1e-3, batch=32):
    """
    Train LSTM on X_train_s using sequences from the full training window.
    The last (1-val_frac) portion is used as internal validation for
    best-checkpoint selection and threshold tuning.
    """
    if not TORCH_AVAILABLE:
        return None, []

    val_idx    = int(len(X_train_s) * val_frac)
    X_val_l    = X_train_s[val_idx:]
    y_val_l    = y_train[val_idx:]

    # LSTM trains on ALL training sequences (includes val-period data)
    Xs_tr, ys_tr   = make_sequences(X_train_s, y_train, window)
    Xs_val, ys_val  = make_sequences(X_val_l,   y_val_l,  window)

    pos_w = float((ys_tr == 0).sum() / max((ys_tr == 1).sum(), 1))

    loader = DataLoader(
        TensorDataset(torch.FloatTensor(Xs_tr), torch.FloatTensor(ys_tr)),
        batch_size=batch, shuffle=True
    )
    model = SolarLSTM(input_size=X_train_s.shape[1])
    opt   = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    crit  = nn.BCELoss(reduction='none')

    best_auc, best_state = 0.0, None
    history = []

    for epoch in range(epochs):
        model.train()
        losses = []
        for xb, yb in loader:
            pred = model(xb)
            w    = torch.where(yb == 1,
                               torch.tensor(pos_w),
                               torch.tensor(1.0))
            loss = (crit(pred, yb) * w).mean()
            opt.zero_grad(); loss.backward(); opt.step()
            losses.append(loss.item())

        model.eval()
        with torch.no_grad():
            vp = model(torch.FloatTensor(Xs_val)).numpy()
        auc = roc_auc_score(ys_val, vp) if ys_val.sum() > 0 else 0.5
        history.append({'epoch': epoch+1, 'loss': float(np.mean(losses)), 'val_auc': float(auc)})

        if auc >= best_auc:
            best_auc  = auc
            best_state = {k: v.clone() for k, v in model.state_dict().items()}

        if (epoch + 1) % 10 == 0:
            print(f"  Epoch {epoch+1:3d} | loss={np.mean(losses):.4f} | "
                  f"val_auc={auc:.4f} | best={best_auc:.4f}")

    if best_state:
        model.load_state_dict(best_state)
    return model, history


# ─── Main ─────────────────────────────────────────────────────────────────────

def train_and_evaluate():
    print("Loading data...")
    flares = load_flares()
    cme    = load_cme()
    storms = load_storms()

    print("Engineering features...")
    daily = build_daily_features(flares, cme, storms)

    X     = daily[FEATURE_COLS].fillna(0).values
    y     = daily[TARGET_COL].values
    dates = daily['date'].values
    n     = len(X)

    test_split   = int(n * 0.80)
    X_train, y_train = X[:test_split], y[:test_split]
    X_test,  y_test  = X[test_split:], y[test_split:]

    scaler     = StandardScaler()
    X_train_s  = scaler.fit_transform(X_train)
    X_test_s   = scaler.transform(X_test)

    # Val slice (last 15% of train) for threshold tuning
    val_idx    = int(len(X_train_s) * 0.85)
    X_val_s    = X_train_s[val_idx:]
    y_val      = y_train[val_idx:]

    print(f"Train: {len(X_train)}  (val slice: {len(X_val_s)})  Test: {len(X_test)}")
    print(f"Positive rate — train: {y_train.mean():.2%}  "
          f"val-slice: {y_val.mean():.2%}  test: {y_test.mean():.2%}")

    os.makedirs("models",  exist_ok=True)
    os.makedirs("results", exist_ok=True)

    # ── Random Forest ─────────────────────────────────────────────────────────
    print("\nTraining Random Forest...")
    rf = RandomForestClassifier(
        n_estimators=500,
        max_depth=8,
        min_samples_leaf=4,
        max_features='sqrt',
        class_weight='balanced',
        random_state=42,
        n_jobs=-1,
    )
    rf.fit(X_train_s, y_train)

    rf_val_prob  = rf.predict_proba(X_val_s)[:, 1]
    rf_thr       = best_threshold(y_val, rf_val_prob, beta=1.5)
    rf_test_prob = rf.predict_proba(X_test_s)[:, 1]
    rf_metrics   = evaluate(y_test, rf_test_prob, rf_thr, "Random Forest")

    with open("models/random_forest.pkl", "wb") as fh:
        pickle.dump({'model': rf, 'scaler': scaler,
                     'feature_cols': FEATURE_COLS, 'threshold': rf_thr}, fh)

    # ── LSTM ──────────────────────────────────────────────────────────────────
    WINDOW = 7
    lstm_metrics   = None
    lstm_test_prob = None
    lstm_thr       = 0.5

    if TORCH_AVAILABLE:
        print("\nTraining LSTM (7-day window, 60 epochs)...")
        lstm, history = train_lstm(X_train_s, y_train,
                                   val_frac=0.85, window=WINDOW, epochs=60)
        if lstm:
            # Threshold on val-slice sequences
            Xs_val, ys_val_seq = make_sequences(X_val_s, y_val, WINDOW)
            lstm.eval()
            with torch.no_grad():
                lstm_val_prob = lstm(torch.FloatTensor(Xs_val)).numpy()
            lstm_thr = best_threshold(ys_val_seq, lstm_val_prob, beta=1.5)

            # Test
            Xs_te, ys_te_seq = make_sequences(X_test_s, y_test, WINDOW)
            with torch.no_grad():
                lstm_test_seq = lstm(torch.FloatTensor(Xs_te)).numpy()
            lstm_metrics = evaluate(ys_te_seq, lstm_test_seq, lstm_thr, "LSTM")

            lstm_test_prob = np.full(len(y_test), np.nan)
            lstm_test_prob[WINDOW:] = lstm_test_seq

            torch.save({
                'model_state': lstm.state_dict(),
                'input_size':  X_train_s.shape[1],
                'window':      WINDOW,
                'history':     history,
                'threshold':   lstm_thr,
            }, "models/lstm_model.pth")

    # ── Ensemble ──────────────────────────────────────────────────────────────
    print("\n--- Ensemble ---")
    ens_metrics = None
    ens_thr     = rf_thr

    if lstm_test_prob is not None:
        valid  = ~np.isnan(lstm_test_prob)
        ens_p  = 0.55 * rf_test_prob[valid] + 0.45 * lstm_test_prob[valid]
        y_ens  = y_test[valid]

        # Tune ensemble threshold on val sequences
        Xs_val, ys_val_seq = make_sequences(X_val_s, y_val, WINDOW)
        lstm.eval()
        with torch.no_grad():
            lstm_val_prob = lstm(torch.FloatTensor(Xs_val)).numpy()
        ens_val_p = 0.55 * rf_val_prob[WINDOW:] + 0.45 * lstm_val_prob
        ens_thr   = best_threshold(ys_val_seq, ens_val_p, beta=1.5)

        ens_metrics = evaluate(y_ens, ens_p, ens_thr, "Ensemble (RF + LSTM)")
    else:
        ens_metrics = rf_metrics
        print("  Ensemble = RF only")

    importances = dict(zip(FEATURE_COLS, rf.feature_importances_.tolist()))

    results = {
        'rf_metrics':         rf_metrics,
        'lstm_metrics':       lstm_metrics,
        'ens_metrics':        ens_metrics,
        'feature_importance': importances,
        'test_start_date':    str(pd.Timestamp(dates[test_split]).date()),
        'rf_threshold':       rf_thr,
        'lstm_threshold':     float(lstm_thr),
        'ens_threshold':      float(ens_thr),
    }
    with open("results/metrics.json", "w") as fh:
        json.dump(results, fh, indent=2)

    print(f"\n{'═'*62}")
    best = ens_metrics or rf_metrics
    print(f"Best | F1={best['f1']:.3f} | Recall={best['recall']:.3f} | "
          f"Precision={best['precision']:.3f} | AUC={best['auc']:.3f}")
    print("Saved → models/  results/metrics.json")
    return results


if __name__ == "__main__":
    train_and_evaluate()
