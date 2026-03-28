"""
Data preprocessing for Solar Flare Prediction System.
Loads and cleans NASA DONKI CSV data.
"""

import pandas as pd
import numpy as np
import re
from pathlib import Path


def parse_location(loc):
    """Parse solar location string like 'S12E90' -> (lat, lon)"""
    if not isinstance(loc, str) or not loc.strip():
        return np.nan, np.nan
    m = re.match(r'([NS])(\d+)([EW])(\d+)', str(loc).strip())
    if m:
        lat = float(m.group(2)) * (1 if m.group(1) == 'N' else -1)
        lon = float(m.group(4)) * (1 if m.group(3) == 'E' else -1)
        return lat, lon
    return np.nan, np.nan


def parse_datetime(s):
    if not isinstance(s, str):
        return pd.NaT
    return pd.to_datetime(s.replace('Z', ''), errors='coerce')


def load_flares(path="data/raw/nasa_solar_flares.csv"):
    df = pd.read_csv(path)

    # Parse class type: "X1.2" -> letter="X", number=1.2
    df['class_letter'] = df['classType'].str.extract(r'^([A-Z])', expand=False)
    df['class_number'] = df['classType'].str.extract(r'^[A-Z]([\d.]+)', expand=False).astype(float)

    # Parse times
    df['beginTime'] = df['beginTime'].apply(parse_datetime)
    df['peakTime']  = df['peakTime'].apply(parse_datetime)
    df['endTime']   = df['endTime'].apply(parse_datetime)
    df['date']      = df['beginTime'].dt.normalize()

    # Numeric intensity (GOES W/m²): C=1e-6, M=1e-5, X=1e-4
    base = {'A': 1e-8, 'B': 1e-7, 'C': 1e-6, 'M': 1e-5, 'X': 1e-4}
    df['base_intensity'] = df['class_letter'].map(base).fillna(1e-7)
    df['intensity'] = df['base_intensity'] * df['class_number'].fillna(1.0)

    # Parse location
    locs = df['sourceLocation'].apply(parse_location)
    df['lat'] = locs.apply(lambda x: x[0])
    df['lon'] = locs.apply(lambda x: x[1])

    # Duration in minutes
    df['duration_min'] = (df['endTime'] - df['beginTime']).dt.total_seconds() / 60

    # Class rank: C=1, M=2, X=3
    df['class_rank'] = df['class_letter'].map({'A': 0, 'B': 0.5, 'C': 1, 'M': 2, 'X': 3}).fillna(0)

    return df.sort_values('beginTime').reset_index(drop=True)


def load_cme(path="data/raw/nasa_cme.csv"):
    df = pd.read_csv(path)
    df['startTime'] = df['startTime'].apply(parse_datetime)
    df['date'] = df['startTime'].dt.normalize()

    # Extract speed from cmeAnalyses JSON-like string
    def extract_speed(s):
        if not isinstance(s, str):
            return np.nan
        m = re.search(r"'speed':\s*([\d.]+)", s)
        return float(m.group(1)) if m else np.nan

    def extract_half_angle(s):
        if not isinstance(s, str):
            return np.nan
        m = re.search(r"'halfAngle':\s*([\d.]+)", s)
        return float(m.group(1)) if m else np.nan

    df['cme_speed']      = df['cmeAnalyses'].apply(extract_speed)
    df['cme_half_angle'] = df['cmeAnalyses'].apply(extract_half_angle)

    return df.sort_values('startTime').reset_index(drop=True)


def load_storms(path="data/raw/nasa_geomagnetic_storms.csv"):
    df = pd.read_csv(path)
    df['startTime'] = df['startTime'].apply(parse_datetime)
    df['date'] = df['startTime'].dt.normalize()

    # Extract max Kp index
    def extract_max_kp(s):
        if not isinstance(s, str):
            return np.nan
        vals = re.findall(r"'kpIndex':\s*([\d.]+)", s)
        return max([float(v) for v in vals]) if vals else np.nan

    df['max_kp'] = df['allKpIndex'].apply(extract_max_kp)
    return df.sort_values('startTime').reset_index(drop=True)


def get_date_range(flares):
    """Return full date range from first to last flare."""
    start = flares['date'].min()
    end   = flares['date'].max()
    return pd.date_range(start, end, freq='D')


if __name__ == "__main__":
    flares = load_flares()
    cme    = load_cme()
    storms = load_storms()

    print(f"Flares : {len(flares)} records  | {flares['class_letter'].value_counts().to_dict()}")
    print(f"CMEs   : {len(cme)} records")
    print(f"Storms : {len(storms)} records  | max Kp: {storms['max_kp'].max()}")
    print(f"Date range: {flares['date'].min().date()} -> {flares['date'].max().date()}")
