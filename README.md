# DRL Stock Trading Agent

A deep reinforcement learning agent that learns to trade stocks from historical OHLCV data. Uses a DQN with an LSTM encoder to capture temporal patterns in price sequences. The reward function includes a Sharpe ratio component, so the agent learns to prefer stable returns over volatile high-risk strategies.

Training runs remotely on a T4 GPU (Google Colab). Inference and backtesting run locally on CPU with minimal memory usage (under 4GB).

## How It Works

The problem is framed as a Markov Decision Process:

- **State**: A sliding window of 100 days of normalized OHLCV data, plus the agent's current balance, shares held, and net worth.
- **Actions**: Hold (0), Buy (1), or Sell (2). Buys go all-in, sells liquidate entirely. A 0.1% transaction fee applies to every trade.
- **Reward**: `R = ln(NW_t / NW_{t-1}) + 0.1 * Sharpe_t`, where Sharpe is the rolling mean/std of log returns over 50 steps. This penalizes volatility and rewards risk-adjusted performance.
- **Inference**: The LSTM model predicts the next scaled close price. If the prediction is above the current by a threshold, it buys. Below, it sells. Otherwise holds.

## Project Structure

```
data/               Raw OHLCV CSVs (generated, gitignored)
models/             ONNX model and scaler files (gitignored)
src/
  data_utils.py     Data fetching (yfinance), loading, and MinMax scaling
  env.py            Gymnasium trading environment
  inference.py      ONNX Runtime inference wrapper
  analysis.py       Gemini API integration for LLM trade analysis
backtest.py         CLI backtest
app.py              Streamlit dashboard
convert.py          Model conversion script (run on Colab)
requirements.txt    Python dependencies
.env                API keys (gitignored)
```

## Setup

```bash
python -m venv stok
source stok/bin/activate   # on Windows: stok\Scripts\activate
pip install -r requirements.txt
```

Add your Gemini API key to `.env`:

```
GEMINI_API_KEY=your_key_here
```

## Usage

### 1. Fetch Data

```python
from src.data_utils import fetch
fetch(['AAPL', 'MSFT', 'BTC-USD', 'SPY'])
```

Downloads 10 years of daily data per ticker and saves CSVs to `data/`.

### 2. Backtest

```bash
python backtest.py AAPL
```

Runs the ONNX agent on the last 252 trading days (1 year). Outputs a CSV log to `data/AAPL_backtest.csv` and prints the agent's return vs. a buy-and-hold benchmark.

### 3. Dashboard

```bash
streamlit run app.py
```

Interactive frontend with price charts, trade markers, net worth comparison, and on-demand Gemini analysis. Use the sidebar to pick a ticker, adjust the buy/sell threshold, and run backtests.

### 4. LLM Analysis (CLI, optional)

```bash
python -c "from src.analysis import analyze; print(analyze('data/AAPL_backtest.csv'))"
```

Sends the backtest results to Gemini for analysis. Also available via the dashboard button.

## Memory Optimizations

- All arrays use float32 (half the memory of float64)
- Sharpe ratio uses a fixed-size ring buffer, not a growing list
- Scaler params saved as .npz so inference needs only NumPy, not sklearn
- ONNX Runtime for lightweight model inference
- Observations are flat vectors, no dictionary overhead

## Dependencies

- yfinance: historical market data
- gymnasium: RL environment interface
- numpy, pandas, scikit-learn: data processing
- onnxruntime: model inference
- python-dotenv: environment variable management
- google-genai: Gemini API for trade analysis
- streamlit: interactive dashboard
- plotly: charts
