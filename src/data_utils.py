import numpy as np
import pandas as pd
import yfinance as yf
from sklearn.preprocessing import MinMaxScaler
import os, time

def fetch(tks, period='10y', out='data'):
    os.makedirs(out, exist_ok=True)
    paths = []
    for tk in tks:
        df = yf.download(tk, period=period, auto_adjust=True)
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        df = df[['Open','High','Low','Close','Volume']]
        df.ffill(inplace=True)
        df.dropna(inplace=True)
        df.index.name = 'Date'
        p = f'{out}/{tk}.csv'
        df.to_csv(p)
        paths.append(p)
        time.sleep(1)
    return paths

def load(path, dtype=np.float32):
    df = pd.read_csv(path, skiprows=[1, 2], index_col=0, parse_dates=True)
    cols = ['Open','High','Low','Close','Volume']
    # handle both flat and multi-header CSVs
    if not all(c in df.columns for c in cols):
        df = pd.read_csv(path, index_col=0, parse_dates=True)
    return df[['Open','High','Low','Close','Volume']].values.astype(dtype)

def fit_scaler(arr):
    s = MinMaxScaler()
    s.fit(arr)
    return s

def apply_scaler(arr, s):
    return s.transform(arr).astype(np.float32)

def save_scaler(s, path='models/scaler.npz'):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    np.savez(path, mn=s.min_.astype(np.float32), sc=s.scale_.astype(np.float32))

def load_scaler_np(path='models/scaler.npz'):
    d = np.load(path)
    return d['mn'], d['sc']

def apply_scaler_np(arr, mn, sc):
    return (arr * sc + mn).astype(np.float32)
