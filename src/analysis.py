import os, csv, json
import numpy as np
import requests
from dotenv import load_dotenv

load_dotenv()

PROMPT_TPL = """Analyze the following trading log from a DRL agent using a DQN with LSTM encoder.
The agent's reward function includes a Sharpe-ratio component: R_t = ln(NW_t/NW_{t-1}) + lambda * S_t.

Identify specific market conditions where the agent's Sharpe-ratio-aware reward led to
defensive behavior compared to a standard trend-follower.

Summary:
- Total trades: {n_trades}
- Buy actions: {n_buy}, Sell actions: {n_sell}, Hold actions: {n_hold}
- Initial net worth: {nw0:.2f}
- Final net worth: {nwf:.2f}
- Return: {ret:.4f} ({ret_pct:.2f}%)
- Max drawdown: {mdd:.4f} ({mdd_pct:.2f}%)
- Sharpe ratio (annualized): {sharpe:.4f}

Last 50 trades:
{trades}

Provide actionable insights for improving the agent's reward function or training."""

def _stats(rows):
    nw = np.array([float(r['net_worth']) for r in rows], dtype=np.float32)
    acts = [int(r['action']) for r in rows]
    n_buy = acts.count(1)
    n_sell = acts.count(2)
    n_hold = acts.count(0)
    peak = np.maximum.accumulate(nw)
    dd = (peak - nw) / (peak + 1e-8)
    mdd = float(dd.max())
    rets = np.diff(np.log(nw + 1e-8))
    mu = float(rets.mean()) * 252
    sd = float(rets.std()) * np.sqrt(252) + 1e-8
    sharpe = mu / sd
    ret = float((nw[-1] - nw[0]) / (nw[0] + 1e-8))
    return dict(n_trades=len(rows), n_buy=n_buy, n_sell=n_sell, n_hold=n_hold,
                nw0=float(nw[0]), nwf=float(nw[-1]), ret=ret, ret_pct=ret*100,
                mdd=mdd, mdd_pct=mdd*100, sharpe=sharpe)

def analyze(log_path):
    key = os.getenv('OPENROUTER_API_KEY')
    if not key or key == 'your_key_here':
        raise ValueError('Set OPENROUTER_API_KEY in .env')

    with open(log_path, 'r') as f:
        rows = list(csv.DictReader(f))

    st = _stats(rows)
    last50 = rows[-50:] if len(rows) > 50 else rows
    trades_str = '\n'.join(
        f"  {r['timestamp']}: act={r['action']} price={r['price']} bal={r['balance']} nw={r['net_worth']}"
        for r in last50
    )
    prompt = PROMPT_TPL.format(**st, trades=trades_str)

    resp = requests.post(
        'https://openrouter.ai/api/v1/chat/completions',
        headers={'Authorization': f'Bearer {key}', 'Content-Type': 'application/json'},
        json={'model': 'google/gemini-2.5-flash-lite',
              'messages': [{'role': 'user', 'content': prompt}]}
    )
    resp.raise_for_status()
    return resp.json()['choices'][0]['message']['content']

if __name__ == '__main__':
    import sys
    p = sys.argv[1] if len(sys.argv) > 1 else 'data/AAPL_backtest.csv'
    print(analyze(p))
