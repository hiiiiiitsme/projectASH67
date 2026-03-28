"""
Feature engineering for Solar Flare Prediction.
Creates daily time-series features from raw event data.
"""

import pandas as pd
import numpy as np
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from scripts.data_preprocessing import load_flares, load_cme, load_storms, get_date_range


def build_daily_features(flares, cme, storms):
    """
    Build a daily feature DataFrame covering the full date range.
    Each row = one day.  Target = any X-class flare in the NEXT 7 days.
    """
    dates = get_date_range(flares)
    df = pd.DataFrame({'date': dates})

    # ── Basic flare counts per class ──────────────────────────────────────────
    for cls in ['C', 'M', 'X', 'B']:
        sub  = flares[flares['class_letter'] == cls]
        daily = sub.groupby('date').size().rename(f'n_{cls.lower()}')
        df = df.merge(daily, on='date', how='left')

    # ── M5+ and M8+ (key X-class precursors) ─────────────────────────────────
    m5 = flares[(flares['class_letter'] == 'M') & (flares['class_number'] >= 5.0)]
    m8 = flares[(flares['class_letter'] == 'M') & (flares['class_number'] >= 8.0)]
    df = df.merge(m5.groupby('date').size().rename('n_m5plus'), on='date', how='left')
    df = df.merge(m8.groupby('date').size().rename('n_m8plus'), on='date', how='left')

    # ── Max class rank & max M-number per day ─────────────────────────────────
    df = df.merge(flares.groupby('date')['class_rank'].max().rename('max_class_rank'), on='date', how='left')
    df = df.merge(
        flares[flares['class_letter'] == 'M'].groupby('date')['class_number'].max().rename('max_m_num'),
        on='date', how='left'
    )

    # ── X-ray intensity ───────────────────────────────────────────────────────
    inten = flares.groupby('date')['intensity'].agg(
        max_intensity='max', sum_intensity='sum', mean_intensity='mean'
    )
    df = df.merge(inten, on='date', how='left')

    # ── Active regions ────────────────────────────────────────────────────────
    df = df.merge(
        flares.groupby('date')['activeRegionNum'].nunique().rename('n_active_regions'),
        on='date', how='left'
    )
    df = df.merge(flares.groupby('date')['lat'].mean().rename('mean_lat'), on='date', how='left')

    # ── CME ───────────────────────────────────────────────────────────────────
    cme_day = cme.groupby('date').agg(
        n_cme=('startTime', 'count'),
        max_cme_speed=('cme_speed', 'max'),
        mean_cme_speed=('cme_speed', 'mean'),
    )
    df = df.merge(cme_day, on='date', how='left')

    # ── Geomagnetic storms ────────────────────────────────────────────────────
    df = df.merge(storms.groupby('date')['max_kp'].max().rename('max_kp'), on='date', how='left')

    # ── Fill NaN → 0 ─────────────────────────────────────────────────────────
    zero_cols = ['n_c', 'n_m', 'n_x', 'n_b',
                 'n_m5plus', 'n_m8plus',
                 'max_class_rank', 'max_m_num',
                 'max_intensity', 'sum_intensity', 'mean_intensity',
                 'n_active_regions', 'n_cme', 'max_cme_speed', 'mean_cme_speed',
                 'max_kp']
    df[zero_cols] = df[zero_cols].fillna(0)
    df['mean_lat'] = df['mean_lat'].fillna(0)

    # ── Rolling windows: 3 / 7 / 14 days ────────────────────────────────────
    for w in [3, 7, 14]:
        df[f'roll{w}_n_x']       = df['n_x'].rolling(w, min_periods=1).sum()
        df[f'roll{w}_n_m']       = df['n_m'].rolling(w, min_periods=1).sum()
        df[f'roll{w}_n_m5plus']  = df['n_m5plus'].rolling(w, min_periods=1).sum()
        df[f'roll{w}_intensity'] = df['max_intensity'].rolling(w, min_periods=1).mean()
        df[f'roll{w}_n_cme']     = df['n_cme'].rolling(w, min_periods=1).sum()

    # ── EMA features (trend, robust to outliers) ─────────────────────────────
    df['ema5_intensity']  = df['max_intensity'].ewm(span=5, min_periods=1).mean()
    df['ema14_intensity'] = df['max_intensity'].ewm(span=14, min_periods=1).mean()
    df['ema7_n_m']        = df['n_m'].ewm(span=7, min_periods=1).mean()
    df['ema7_n_m5plus']   = df['n_m5plus'].ewm(span=7, min_periods=1).mean()

    # ── Intensity trend: short EMA / long EMA (is activity rising?) ──────────
    denom = df['ema14_intensity'].replace(0, np.nan)
    df['ema_trend'] = (df['ema5_intensity'] / denom).fillna(1.0).clip(0.1, 10.0)

    # ── 27-day lag (solar back-face estimation) ───────────────────────────────
    df['lag27_n_x']       = df['n_x'].shift(27).fillna(0)
    df['lag27_n_m']       = df['n_m'].shift(27).fillna(0)
    df['lag27_n_m5plus']  = df['n_m5plus'].shift(27).fillna(0)
    df['lag27_intensity'] = df['max_intensity'].shift(27).fillna(0)

    # ── Solar cycle phase ─────────────────────────────────────────────────────
    cycle_start   = pd.Timestamp('2019-12-01')
    df['cycle_day']   = (df['date'] - cycle_start).dt.days
    df['cycle_phase'] = np.sin(2 * np.pi * df['cycle_day'] / (11 * 365.25))
    df['day_of_year'] = pd.to_datetime(df['date']).dt.dayofyear

    # ── TARGET: any X-class flare in next 7 days (tomorrow .. +7) ────────────
    df['x_next_7d'] = (
        df['n_x'].rolling(7, min_periods=1).sum().shift(-7).fillna(0) > 0
    ).astype(int)

    return df.reset_index(drop=True)


FEATURE_COLS = [
    # Basic counts
    'n_c', 'n_m', 'n_x', 'n_b',
    # Intensity
    'max_intensity', 'sum_intensity', 'mean_intensity',
    # Active regions / position
    'n_active_regions', 'mean_lat',
    # CME
    'n_cme', 'max_cme_speed', 'mean_cme_speed',
    # Storm
    'max_kp',
    # Rolling windows (3 / 7 / 14 days)
    'roll3_n_x', 'roll3_n_m', 'roll3_intensity', 'roll3_n_cme',
    'roll7_n_x', 'roll7_n_m', 'roll7_intensity', 'roll7_n_cme',
    'roll14_n_x', 'roll14_n_m', 'roll14_intensity', 'roll14_n_cme',
    # EMA / trend (key improvement over v1)
    'ema5_intensity', 'ema14_intensity', 'ema7_n_m', 'ema_trend',
    # 27-day back-face lag
    'lag27_n_x', 'lag27_n_m', 'lag27_intensity',
    # Solar cycle
    'cycle_phase', 'day_of_year',
]

TARGET_COL = 'x_next_7d'


if __name__ == "__main__":
    flares = load_flares()
    cme    = load_cme()
    storms = load_storms()
    daily  = build_daily_features(flares, cme, storms)

    print(f"Shape: {daily.shape} | Features: {len(FEATURE_COLS)}")
    print(f"Date range: {daily['date'].min().date()} → {daily['date'].max().date()}")
    print(f"Target positive rate: {daily[TARGET_COL].mean():.1%}  ({daily[TARGET_COL].sum()} days)")

    corr = daily[FEATURE_COLS + [TARGET_COL]].corr()[TARGET_COL].drop(TARGET_COL)
    print("\nTop 10 correlations with target:")
    print(corr.abs().sort_values(ascending=False).head(10).to_string())

    os.makedirs("data/processed", exist_ok=True)
    daily.to_csv("data/processed/daily_features.csv", index=False)
    print("\nSaved → data/processed/daily_features.csv")
