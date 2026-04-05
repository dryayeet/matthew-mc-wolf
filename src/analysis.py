import os, csv
import numpy as np
from google import genai
from dotenv import load_dotenv

load_dotenv()

PROMPT_TPL = """Be concise and accurate. No filler, no speculation. Use bullet points. Be very specific and to the point. Answer in short. 

Analyze this trading log from a DRL agent (DQN + LSTM encoder).
Reward function: R_t = ln(NW_t/NW_{{t-1}}) + lambda * S_t (Sharpe-ratio-aware).

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

Answer these three questions only, backed by data from the log:
1. Where did the Sharpe-aware reward cause defensive behavior vs a standard trend-follower?
2. What specific market conditions triggered buys, sells, and prolonged holds?
3. What are 2-3 concrete, actionable changes to improve the reward function or training?"""

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
    key = os.getenv('GEMINI_API_KEY')
    if not key or key == 'your_key_here':
        raise ValueError('Set GEMINI_API_KEY in .env')

    client = genai.Client(api_key=key)

    with open(log_path, 'r') as f:
        rows = list(csv.DictReader(f))

    st = _stats(rows)
    last50 = rows[-50:] if len(rows) > 50 else rows
    trades_str = '\n'.join(
        f"  {r['timestamp']}: act={r['action']} price={r['price']} bal={r['balance']} nw={r['net_worth']}"
        for r in last50
    )
    prompt = PROMPT_TPL.format(**st, trades=trades_str)

    resp = client.models.generate_content(
        model='gemini-2.5-flash-lite',
        contents=prompt
    )
    return resp.text

if __name__ == '__main__':
    import sys
    p = sys.argv[1] if len(sys.argv) > 1 else 'data/AAPL_backtest.csv'
    print(analyze(p))
