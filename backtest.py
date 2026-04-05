import numpy as np
import csv, sys
from src.data_utils import load, load_scaler_np, apply_scaler_np
from src.inference import load_model, get_action
from src.env import StockTradingEnv

def run(ticker='AAPL', model_path='models/agent.tflite',
        scaler_path='models/scaler.npz', oos_days=252):
    raw = load(f'data/{ticker}.csv')
    mn, sc = load_scaler_np(scaler_path)
    scaled = apply_scaler_np(raw, mn, sc)

    # last oos_days as out-of-sample
    split = max(0, len(raw) - oos_days)
    raw_oos = raw[split:]
    scl_oos = scaled[split:]

    env = StockTradingEnv(d=scl_oos, dr=raw_oos)
    interp = load_model(model_path)

    obs, _ = env.reset()
    log = []
    done = False

    while not done:
        a = get_action(interp, obs)
        obs, r, done, trunc, info = env.step(a)
        log.append({
            'timestamp': env.t,
            'action': info['action'],
            'price': float(env.dr[env.t, 3]),
            'balance': float(env.bal),
            'net_worth': float(info['nw'])
        })

    # write CSV log
    out_path = f'data/{ticker}_backtest.csv'
    with open(out_path, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=['timestamp','action','price','balance','net_worth'])
        w.writeheader()
        w.writerows(log)

    # buy & hold benchmark
    p0 = raw_oos[env.w, 3]  # first close after window
    pf = raw_oos[-1, 3]
    bh_ret = (pf - p0) / p0
    ag_ret = (log[-1]['net_worth'] - env.b0) / env.b0

    print(f'Ticker:           {ticker}')
    print(f'OOS period:       {len(raw_oos)} days')
    print(f'Agent return:     {ag_ret:.4f} ({ag_ret*100:.2f}%)')
    print(f'Buy&Hold return:  {bh_ret:.4f} ({bh_ret*100:.2f}%)')
    print(f'Log saved to:     {out_path}')
    return out_path

if __name__ == '__main__':
    tk = sys.argv[1] if len(sys.argv) > 1 else 'AAPL'
    run(ticker=tk)
