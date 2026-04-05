# DRL Stock Trading Agent

A deep reinforcement learning agent that learns to trade stocks from historical OHLCV data. Uses a DQN with an LSTM encoder to capture temporal patterns in price sequences. The reward function includes a Sharpe ratio component, so the agent learns to prefer stable returns over volatile high-risk strategies.

Training runs remotely on a T4 GPU (Google Colab). Inference and backtesting run locally on CPU with minimal memory usage (under 4GB).

## How It Works

The problem is framed as a Markov Decision Process:

- **State**: A sliding window of 30 days of normalized OHLCV data, plus the agent's current balance, shares held, and net worth. Flattened to a 153-element vector.
- **Actions**: Hold (0), Buy (1), or Sell (2). Buys go all-in, sells liquidate entirely. A 0.1% transaction fee applies to every trade.
- **Reward**: `R = ln(NW_t / NW_{t-1}) + 0.1 * Sharpe_t`, where Sharpe is the rolling mean/std of log returns over 50 steps. This penalizes volatility and rewards risk-adjusted performance.

## Project Structure

```
data/               Raw OHLCV CSVs (generated, gitignored)
models/             TFLite model and scaler files (gitignored)
src/
  data_utils.py     Data fetching (yfinance), loading, and MinMax scaling
  env.py            Gymnasium trading environment
  inference.py      TFLite runtime wrapper
  analysis.py       Gemini API integration for LLM trade analysis
backtest.py         Run the trained agent on out-of-sample data
convert.py          Convert SavedModel to quantized TFLite (run on Colab)
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

This downloads 10 years of daily data per ticker and saves CSVs to `data/`.

### 2. Train (on Google Colab)

Upload `src/env.py` and `src/data_utils.py` to a Colab notebook. These files work without modification on Colab. Train your DQN/LSTM agent there using the Gymnasium environment, then export a SavedModel.

### 3. Convert Model

Run this on Colab after training:

```bash
python convert.py models/saved_model
```

This produces a quantized `models/agent.tflite` file (float16, roughly half the size of the original). Download it to your local `models/` folder.

You also need the scaler file. After fitting the scaler on your training data:

```python
from src.data_utils import fit_scaler, save_scaler, load
data = load('data/AAPL.csv')
train_data = data[:int(len(data)*0.8)]
s = fit_scaler(train_data)
save_scaler(s, 'models/scaler.npz')
```

Download `models/scaler.npz` to your local machine as well.

### 4. Backtest

```bash
python backtest.py AAPL
```

Runs the TFLite agent on the last 252 trading days (1 year). Outputs a CSV log to `data/AAPL_backtest.csv` and prints the agent's return vs. a buy-and-hold benchmark.

### 5. LLM Analysis (optional)

```bash
python -c "from src.analysis import analyze; print(analyze('data/AAPL_backtest.csv'))"
```

Sends the backtest results to Gemini Flash via OpenRouter. The LLM analyzes when the Sharpe-aware reward caused the agent to play defensively compared to a simple trend-follower.

## Memory Optimizations

- All arrays use float32 (half the memory of float64)
- Sharpe ratio uses a fixed-size ring buffer, not a growing list
- Scaler params saved as .npz so inference needs only NumPy, not sklearn
- TFLite runtime is used instead of full TensorFlow (5MB vs 500MB)
- Observations are flat vectors, no dictionary overhead

## Dependencies

- yfinance: historical market data
- gymnasium: RL environment interface
- numpy, pandas, scikit-learn: data processing
- tflite-runtime: lightweight model inference
- python-dotenv: environment variable management
- requests: OpenRouter API calls
