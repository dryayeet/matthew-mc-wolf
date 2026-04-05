import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import os, csv

from src.data_utils import load, fit_scaler, apply_scaler, save_scaler, load_scaler_np, apply_scaler_np
from src.inference import load_model, get_action
from src.env import StockTradingEnv
from src.analysis import analyze, _stats

st.set_page_config(page_title="DRL Trading Agent", layout="wide")
st.title("DRL Stock Trading Agent")

# --- cache model ---
@st.cache_resource
def cached_model():
    return load_model('models/trading_model.onnx')

# --- sidebar ---
with st.sidebar:
    st.header("Controls")
    ticker = st.selectbox("Ticker", ['AAPL', 'MSFT', 'BTC-USD', 'SPY'])
    threshold = st.slider("Buy/Sell Threshold", 0.001, 0.050, 0.005, 0.001, format="%.3f")
    run_bt = st.button("Run Backtest")
    run_llm = st.button("Analyze with Gemini", disabled='log' not in st.session_state)

# --- backtest ---
if run_bt:
    data_path = f'data/{ticker}.csv'
    if not os.path.exists(data_path):
        with st.spinner(f"Fetching {ticker} data..."):
            from src.data_utils import fetch
            fetch([ticker])

    with st.spinner("Running backtest..."):
        raw = load(data_path)
        train = raw[:len(raw) - 252]
        s = fit_scaler(train)
        save_scaler(s, 'models/scaler.npz')
        mn, sc = load_scaler_np('models/scaler.npz')
        scaled = apply_scaler_np(raw, mn, sc)

        split = max(0, len(raw) - 252)
        raw_oos = raw[split:]
        scl_oos = scaled[split:]

        env = StockTradingEnv(d=scl_oos, dr=raw_oos)
        sess = cached_model()

        obs, _ = env.reset()
        log = []
        done = False

        while not done:
            prices = scl_oos[env.t - env.w:env.t, 3]
            a = get_action(sess, prices, threshold=threshold)
            obs, r, done, trunc, info = env.step(a)
            log.append({
                'timestamp': int(env.t),
                'action': int(info['action']),
                'price': float(env.dr[env.t, 3]),
                'balance': float(env.bal),
                'net_worth': float(info['nw'])
            })

        # buy & hold curve
        p0 = raw_oos[env.w, 3]
        bh_curve = [env.b0 * raw_oos[env.w + i, 3] / p0 for i in range(len(log))]

        # save to CSV
        out_path = f'data/{ticker}_backtest.csv'
        with open(out_path, 'w', newline='') as f:
            w = csv.DictWriter(f, fieldnames=['timestamp', 'action', 'price', 'balance', 'net_worth'])
            w.writeheader()
            w.writerows(log)

        st.session_state['log'] = log
        st.session_state['bh_curve'] = bh_curve
        st.session_state['ticker'] = ticker
        st.session_state['csv_path'] = out_path
        st.session_state['b0'] = env.b0
        st.session_state['llm_result'] = None

# --- display ---
if 'log' in st.session_state:
    log = st.session_state['log']
    bh_curve = st.session_state['bh_curve']
    tk = st.session_state['ticker']
    b0 = st.session_state['b0']

    acts = [r['action'] for r in log]
    nw = [r['net_worth'] for r in log]
    prices = [r['price'] for r in log]
    n_buy = acts.count(1)
    n_sell = acts.count(2)
    n_hold = acts.count(0)

    ag_ret = (nw[-1] - b0) / b0
    bh_ret = (bh_curve[-1] - b0) / b0

    # stats
    nw_arr = np.array(nw, dtype=np.float32)
    peak = np.maximum.accumulate(nw_arr)
    mdd = float(((peak - nw_arr) / (peak + 1e-8)).max())
    rets = np.diff(np.log(nw_arr + 1e-8))
    sharpe = float(rets.mean() * 252 / (rets.std() * np.sqrt(252) + 1e-8))

    # --- metrics row ---
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Agent Return", f"{ag_ret*100:.2f}%")
    c2.metric("Buy & Hold", f"{bh_ret*100:.2f}%")
    c3.metric("Sharpe Ratio", f"{sharpe:.2f}")
    c4.metric("Max Drawdown", f"{mdd*100:.2f}%")
    c5.metric("Total Trades", f"{n_buy + n_sell}")

    # --- price chart with trade markers ---
    buy_idx = [i for i, a in enumerate(acts) if a == 1]
    sell_idx = [i for i, a in enumerate(acts) if a == 2]

    fig1 = go.Figure()
    fig1.add_trace(go.Scatter(y=prices, mode='lines', name='Close Price', line=dict(color='#636EFA')))
    if buy_idx:
        fig1.add_trace(go.Scatter(
            x=buy_idx, y=[prices[i] for i in buy_idx],
            mode='markers', name='Buy',
            marker=dict(symbol='triangle-up', size=14, color='#00CC96')
        ))
    if sell_idx:
        fig1.add_trace(go.Scatter(
            x=sell_idx, y=[prices[i] for i in sell_idx],
            mode='markers', name='Sell',
            marker=dict(symbol='triangle-down', size=14, color='#EF553B')
        ))
    fig1.update_layout(title=f"{tk} Close Price with Trade Signals", xaxis_title="Day", yaxis_title="Price ($)", height=450)
    st.plotly_chart(fig1, use_container_width=True)

    # --- net worth chart ---
    fig2 = go.Figure()
    fig2.add_trace(go.Scatter(y=nw, mode='lines', name='Agent', line=dict(color='#636EFA')))
    fig2.add_trace(go.Scatter(y=bh_curve, mode='lines', name='Buy & Hold', line=dict(color='#FFA15A', dash='dash')))
    fig2.update_layout(title="Portfolio Value: Agent vs Buy & Hold", xaxis_title="Day", yaxis_title="Net Worth ($)", height=400)
    st.plotly_chart(fig2, use_container_width=True)

    # --- action distribution ---
    fig3 = go.Figure(go.Bar(
        x=['Hold', 'Buy', 'Sell'],
        y=[n_hold, n_buy, n_sell],
        marker_color=['#636EFA', '#00CC96', '#EF553B']
    ))
    fig3.update_layout(title="Action Distribution", yaxis_title="Count", height=350)
    st.plotly_chart(fig3, use_container_width=True)

    # --- LLM analysis ---
    if run_llm:
        with st.spinner("Calling Gemini..."):
            try:
                result = analyze(st.session_state['csv_path'])
                st.session_state['llm_result'] = result
            except Exception as e:
                st.error(f"Gemini API error: {e}")

    if st.session_state.get('llm_result'):
        st.subheader("Gemini Analysis")
        st.markdown(st.session_state['llm_result'])

else:
    st.info("Select a ticker and click 'Run Backtest' in the sidebar to start.")
