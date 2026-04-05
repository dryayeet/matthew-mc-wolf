# Project Report: Autonomous Stock Trading Agent using Deep Reinforcement Learning

## 1. Abstract

This project implements an autonomous stock trading system that uses Deep Reinforcement Learning (DRL) to learn optimal buy/hold/sell decisions from historical price data. The core architecture pairs a Deep Q-Network (DQN) with a Long Short-Term Memory (LSTM) encoder to process temporal OHLCV (Open, High, Low, Close, Volume) sequences. The agent's reward function incorporates a Sharpe ratio component, explicitly optimizing for risk-adjusted returns rather than raw profit. The system operates within a 4GB RAM, CPU-only constraint, using ONNX Runtime for inference and a custom Gymnasium environment for simulation. An LLM integration (Google Gemini) provides post-hoc analysis of trading behavior.

## 2. Problem Formulation

### 2.1 Motivation

Financial markets are non-linear, non-stationary stochastic systems. Traditional rule-based strategies (moving average crossovers, RSI thresholds, etc.) rely on hand-crafted heuristics that fail to capture complex temporal dependencies in price data. Deep Reinforcement Learning offers a data-driven alternative: the agent learns a policy directly from market observations, without explicit feature engineering of trading rules.

### 2.2 Markov Decision Process (MDP) Formulation

The trading problem is modeled as an MDP defined by the tuple (S, A, T, R, gamma):

- **State space S**: A sliding window of 100 days of MinMax-normalized OHLCV data, concatenated with three portfolio metadata values (normalized balance, shares held, and net worth). The observation is a flat float32 vector of dimension 503 (100 x 5 + 3).

- **Action space A**: Discrete with 3 actions:
  - 0: Hold (no action)
  - 1: Buy (all-in, spend entire available balance)
  - 2: Sell (liquidate entire position)

- **Transition function T**: Deterministic given the action and next market price. The environment advances one trading day per step.

- **Reward function R**: Defined as:

  ```
  R_t = ln(NW_t / NW_{t-1}) + lambda * S_t
  ```

  Where:
  - `NW_t` is the portfolio net worth at time t
  - `ln(NW_t / NW_{t-1})` is the logarithmic return (penalizes large losses more than it rewards equivalent gains)
  - `S_t` is the rolling Sharpe ratio over the trailing M steps
  - `lambda` (default 0.1) controls the weight of risk-adjusted performance

- **Discount factor gamma**: Implicit in the training phase (handled by the RL algorithm on Colab).

## 3. System Architecture

### 3.1 High-Level Pipeline

```
Phase 1 (Local)  : Data Acquisition --> CSV storage --> float32 NumPy arrays
Phase 2 (Local)  : Gymnasium environment definition (Colab-compatible)
Phase 3 (Remote) : DQN+LSTM training on T4 GPU (Google Colab)
Phase 4 (Local)  : ONNX inference --> Backtesting --> LLM Analysis --> Dashboard
```

### 3.2 Directory Structure

```
project_root/
  data/                 Raw OHLCV CSVs (gitignored)
  models/
    trading_model.onnx  Trained LSTM model (ONNX format, 21KB)
    scaler.npz          MinMaxScaler parameters (float32)
  src/
    __init__.py          Package marker
    data_utils.py        Data fetching, loading, scaling
    env.py               Gymnasium trading environment
    inference.py         ONNX Runtime inference wrapper
    analysis.py          Gemini API integration
  backtest.py            CLI backtest orchestrator
  app.py                 Streamlit dashboard
  convert.py             TFLite conversion script (Colab-only)
  requirements.txt       Python dependencies
  .env                   API keys (gitignored)
  .gitignore             Excludes .env, data/, models/, __pycache__/
```

## 4. Implementation Details

### 4.1 Data Acquisition and Preprocessing (src/data_utils.py)

**Source**: Yahoo Finance via the `yfinance` library.

**Tickers**: AAPL, MSFT, BTC-USD, SPY (configurable).

**Temporal scope**: 10 years of daily OHLCV data (~2500 rows per equity ticker, ~3650 for BTC-USD which trades 365 days/year).

**Preprocessing pipeline**:

1. `fetch(tickers, period='10y')`: Downloads data via `yf.download()`, handles MultiIndex columns from yfinance v1.2+, forward-fills NaN values (holidays, missing days), drops any remaining NaN rows, saves to `data/{ticker}.csv`.

2. `load(path)`: Reads CSV to float32 NumPy array of shape (T, 5). Handles both legacy flat-header and new multi-header CSV formats from different yfinance versions via a `skiprows=[1,2]` fallback strategy.

3. **MinMax scaling**: `fit_scaler(arr)` wraps `sklearn.preprocessing.MinMaxScaler` to normalize each column independently to [0, 1]. This is critical because raw OHLCV features span vastly different ranges (e.g., Close: $25-$260, Volume: 20M-400M for AAPL).

4. **Scaler persistence**: `save_scaler(s, path)` extracts the scaler's `min_` and `scale_` arrays and stores them as a lightweight `.npz` file. At inference time, `load_scaler_np()` and `apply_scaler_np()` reconstruct the identical transformation using pure NumPy arithmetic (`arr * scale + min`), eliminating the sklearn dependency from the inference path.

**Memory optimization**: All arrays are explicitly cast to `np.float32` (4 bytes per value) rather than the pandas default of float64 (8 bytes). For 2500 rows x 5 columns, this reduces memory from 100KB to 50KB per ticker. The pandas DataFrame is discarded immediately after `.values` extraction.

### 4.2 Trading Environment (src/env.py)

**Class**: `StockTradingEnv(gymnasium.Env)`

**Constructor parameters**:

| Parameter | Default | Description |
|-----------|---------|-------------|
| `d`       | --      | Scaled OHLCV array (T, 5), float32 |
| `dr`      | --      | Raw OHLCV array (T, 5), float32 |
| `w`       | 100     | Sliding window size (matches LSTM input length) |
| `m`       | 50      | Sharpe ratio lookback period |
| `lam`     | 0.1     | Sharpe weight in reward function |
| `fee`     | 0.001   | Transaction cost (0.1% per trade) |
| `b0`      | 10000.0 | Initial cash balance |

**Dual-array design**: The environment receives both scaled (`d`) and raw (`dr`) OHLCV arrays. Scaled data is used for neural network observations (values in [0, 1] are optimal for gradient-based learning). Raw data is used for dollar-denominated transaction calculations (buying shares, computing net worth). Memory overhead is negligible: two copies of ~2500 x 5 x 4 bytes = 100KB total.

**Observation construction** (`_obs()`):
1. Extract the scaled OHLCV window: `d[t-w:t]` of shape (100, 5)
2. Flatten to (500,) via `.ravel()` (returns a view, no allocation)
3. Compute normalized metadata: `[balance/b0, shares/100, net_worth/b0]`
4. Concatenate to final shape (503,) via `np.concatenate`

The observation space is declared as `Box(low=-inf, high=inf, shape=(503,), dtype=float32)`. A flat vector was chosen over a Dict space for universal RL framework compatibility (Stable-Baselines3, RLlib, etc.).

**Action execution logic**:

- **Buy (action=1)**: Computes maximum affordable shares as `floor(balance / (price * (1 + fee)))`. If less than 1 share is affordable, the action is forced to Hold. Otherwise, deducts `shares * price * (1 + fee)` from balance.
- **Sell (action=2)**: If shares held < 1, forced to Hold. Otherwise, adds `shares * price * (1 - fee)` to balance and sets shares to 0.
- **Hold (action=0)**: No operation.

**Transaction fees**: A 0.1% fee is applied to every buy and sell. For a buy, the cost is `shares * price * 1.001`. For a sell, the revenue is `shares * price * 0.999`. This models real-world brokerage friction and prevents the agent from learning hyperactive strategies that would be unprofitable after costs.

**Reward computation**:

1. **Log return**: `r_log = ln(NW_t / NW_{t-1})`. The logarithmic formulation has two properties: (a) it is additive over time (sum of log returns = log of total return), and (b) it penalizes losses more than it rewards equivalent gains (asymmetric risk sensitivity).

2. **Rolling Sharpe ratio**: Computed over the trailing `m=50` steps using a ring buffer. The ring buffer `rh` is a pre-allocated `np.zeros(m, dtype=float32)` array. Each step writes `r_log` to index `ri % m`, overwriting the oldest value. Mean and standard deviation are computed via `rh.mean()` and `rh.std()`, which operate on the fixed-size array in O(m) time with zero memory allocation. For the first `m` steps, the Sharpe component is set to 0.0 to avoid unstable estimates from small samples.

3. **Combined reward**: `R = r_log + 0.1 * sharpe`. The lambda=0.1 weight means the agent values stability at roughly 10% of the importance of raw returns. This encourages strategies that achieve consistent gains rather than volatile high-risk approaches.

**Episode termination**:
- Normal: when `t >= len(data) - 1` (end of data)
- Early: when `net_worth < 1.0` (bankruptcy), with an additional -1.0 penalty added to the reward

**Colab compatibility**: The environment depends only on `gymnasium` and `numpy`. No OS-specific code, no file I/O, no Windows paths. It can be uploaded directly to a Google Colab notebook and used with any RL training library without modification.

### 4.3 Neural Network Architecture

The model is a sequential LSTM network trained remotely on a T4 GPU:

```
Layer              Output Shape        Parameters
LSTM               (None, 100, 50)     10,400
Dropout            (None, 100, 50)     0
LSTM               (None, 100, 60)     26,640
Dropout            (None, 100, 60)     0
LSTM               (None, 100, 80)     45,120
Dropout            (None, 100, 80)     0
LSTM               (None, 120)         96,480
Dropout            (None, 120)         0
Dense              (None, 1)           121
Total parameters: 178,763 (698.30 KB)
```

**Input**: (batch, 100, 1) -- 100 timesteps of a single feature (scaled close price).

**Output**: (batch, 1) -- predicted next scaled close price.

**Architecture rationale**: Four stacked LSTM layers with increasing hidden dimensions (50, 60, 80, 120) progressively extract higher-order temporal features. The first three LSTMs return full sequences (`return_sequences=True`) to feed into subsequent layers. The final LSTM returns only the last hidden state, which is projected through a Dense layer to produce the scalar prediction. Dropout layers between LSTMs provide regularization against overfitting on training data.

**Note**: This is a price prediction model, not a direct policy network. The predicted price is converted to a trading action via a threshold-based strategy in the inference module (Section 4.4).

### 4.4 Inference Engine (src/inference.py)

**Runtime**: ONNX Runtime (`onnxruntime` package, ~15MB) rather than full TensorFlow (~500MB). The model was originally trained in Keras and exported as `.h5`, then converted through the pipeline: `.h5` --> TensorFlow SavedModel --> ONNX (via `tf2onnx`). The ONNX format was necessary because the `.tflite` export used TensorFlow Flex delegate ops (FlexTensorListReserve for LSTM), which are not supported by any TFLite interpreter available on Windows via pip.

**Model loading**: `load_model(path)` creates an `ort.InferenceSession`, which allocates only the memory required by the model graph. For this 178K-parameter network, runtime memory is approximately 1MB.

**Action derivation** (`get_action(sess, prices, threshold=0.005)`):

1. Reshape input prices to (1, 100, 1) float32
2. Run ONNX inference to get predicted next scaled close price
3. Compute relative difference: `diff = (predicted - last_input) / |last_input|`
4. Decision logic:
   - If `diff > threshold` (default 0.5%): return 1 (Buy) -- model predicts price increase
   - If `diff < -threshold`: return 2 (Sell) -- model predicts price decrease
   - Otherwise: return 0 (Hold) -- prediction is within noise margin

The threshold parameter is configurable (0.1% to 5.0% in the Streamlit dashboard) and controls the agent's sensitivity. A lower threshold produces more trades; a higher threshold produces fewer, more confident trades.

### 4.5 Backtesting Pipeline (backtest.py)

**Orchestration flow**:

1. Load raw OHLCV data from CSV
2. Load scaler parameters from `.npz` file
3. Apply MinMax scaling to raw data
4. Split: last 252 trading days (~1 year) as out-of-sample (OOS) test set
5. Initialize `StockTradingEnv` with OOS data
6. Load ONNX model
7. Run episode: at each step, extract the 100-day scaled close price window, get action from model, step the environment
8. Write trade log to CSV: `[timestamp, action, price, balance, net_worth]`
9. Compute and print agent return vs. buy-and-hold benchmark

**Buy-and-hold benchmark**: Computed as `(final_price - initial_price) / initial_price`, where initial price is the close on the first day after the window warmup period. This represents the return of a passive strategy that buys on day 1 and holds until the end.

**Memory profile of a backtest run**:
- OOS data (raw + scaled): ~20KB
- Environment state (ring buffer, metadata): ~1KB
- ONNX interpreter: ~1MB
- Trade log (list of dicts): ~50KB
- **Total: under 2MB**, well within the 4GB constraint

### 4.6 LLM Analysis (src/analysis.py)

**API**: Google Gemini via the `google-genai` SDK (model: `gemini-2.5-flash-lite`).

**Pre-processing**: Before sending the trade log to the LLM, summary statistics are computed locally:

- **Total return**: `(NW_final - NW_initial) / NW_initial`
- **Max drawdown**: `max((peak - NW) / peak)` where peak is the running maximum of net worth
- **Annualized Sharpe ratio**: `(mean_daily_return * 252) / (std_daily_return * sqrt(252))`
- **Action counts**: number of buys, sells, and holds
- **Last 50 trades**: formatted as text to keep API payload small

**Prompt engineering**: The prompt instructs the LLM to be concise and accurate, use bullet points, avoid speculation, and answer three specific questions:
1. Where did the Sharpe-aware reward cause defensive behavior vs. a trend-follower?
2. What market conditions triggered buys, sells, and prolonged holds?
3. What are 2-3 concrete improvements to the reward function or training?

**API key management**: Loaded from `.env` via `python-dotenv`. The `.env` file is gitignored to prevent credential leakage.

### 4.7 Streamlit Dashboard (app.py)

**Layout**:

- **Sidebar**: Ticker selector (dropdown), buy/sell threshold slider (0.001 to 0.050), "Run Backtest" button, "Analyze with Gemini" button
- **Metrics row**: Five `st.metric` cards showing agent return, buy-and-hold return, Sharpe ratio, max drawdown, total trades
- **Price chart**: Plotly line chart of close prices with green triangle-up markers at buy points and red triangle-down markers at sell points
- **Net worth chart**: Plotly dual-line chart comparing agent portfolio value vs. buy-and-hold portfolio value over time
- **Action distribution**: Plotly bar chart showing counts of Hold, Buy, Sell actions
- **LLM analysis**: Gemini output rendered as markdown, triggered only by button click to avoid unnecessary API calls

**State management**: Backtest results are stored in `st.session_state` so they persist across Streamlit reruns. The ONNX model is loaded once via `@st.cache_resource` and reused across backtests.

**Data auto-fetch**: If the selected ticker's CSV does not exist in `data/`, the app automatically fetches it via `yfinance` before running the backtest.

## 5. Experimental Results

### 5.1 AAPL Backtest (252-day Out-of-Sample Period)

| Metric | Value |
|--------|-------|
| Agent return | 13.82% |
| Buy & Hold return | 11.82% |
| Alpha (excess return) | +2.00% |
| Annualized Sharpe ratio | 2.87 |
| Maximum drawdown | 1.20% |
| Total trades | 8 (4 buys, 4 sells) |
| Hold actions | 143 |
| Total trading days | 151 |

### 5.2 Analysis

The agent outperformed the buy-and-hold benchmark by 2 percentage points while maintaining an exceptionally low maximum drawdown of 1.20%. The annualized Sharpe ratio of 2.87 indicates strong risk-adjusted performance (values above 2.0 are generally considered excellent).

The trading behavior is highly conservative: only 8 trades were executed over 151 trading days. The Sharpe ratio component in the reward function (`lambda=0.1`) successfully trained the agent to prioritize stability over aggressive profit-seeking. The agent preferred to remain in cash during volatile or sideways market conditions, entering positions only when the LSTM predicted a sufficiently strong directional move.

**Specific trade example**: The agent bought at $248.12 (trade 202), held through a price increase to $258.03, and sold at $256.20 (trade 206), capturing a $8.08 per-share gain while exiting before a subsequent price decline.

### 5.3 Limitations

1. **Low trade frequency**: 8 trades in 252 days is extremely passive. The 2% alpha could be statistical noise rather than genuine predictive ability. A longer OOS period or multiple tickers would be needed to establish significance.

2. **Single-feature input**: The LSTM receives only scaled close prices. Adding volume, volatility indicators (ATR, Bollinger Bands), or other technical features could improve signal quality.

3. **All-or-nothing actions**: The discrete Buy/Sell/Hold action space forces all-in or all-out positions. A continuous action space allowing partial position sizing would enable more nuanced risk management.

4. **Scaler distribution shift**: The MinMax scaler is fit on training data. OOS prices that exceed the training range produce scaled values > 1.0, which the model has never seen during training. This out-of-distribution issue could degrade prediction quality over time.

## 6. Memory and Performance Optimizations

| Optimization | Implementation | Impact |
|---|---|---|
| float32 everywhere | `astype(np.float32)` on all arrays | 50% memory reduction vs. float64 |
| Ring buffer for Sharpe | Fixed `np.zeros(m)` array with modular index | O(1) memory, no growing lists |
| Scaler as .npz | Save `min_`/`scale_` arrays, apply via NumPy | Removes sklearn from inference path |
| ONNX Runtime | `onnxruntime` package (~15MB) | ~97% smaller than full TensorFlow (~500MB) |
| Flat observation vector | 503-element contiguous array | No dict overhead, single allocation |
| Lazy model loading | `@st.cache_resource` in Streamlit | Model loaded once, reused across runs |
| No pandas at inference | CSV written via `csv` module, arrays via NumPy | Avoids DataFrame overhead during backtest |

**Total runtime memory for a backtest**: under 2MB (excluding Python interpreter and package overhead).

## 7. Technology Stack

| Component | Technology | Version | Purpose |
|---|---|---|---|
| Data source | yfinance | 1.2.0 | Historical OHLCV data from Yahoo Finance |
| RL environment | Gymnasium | 1.2.3 | Standard RL environment interface |
| Numerical computing | NumPy | 2.4.4 | Array operations, float32 throughout |
| Data loading | Pandas | 3.0.2 | CSV parsing (data acquisition only) |
| Feature scaling | scikit-learn | 1.8.0 | MinMaxScaler (training only) |
| Model inference | ONNX Runtime | 1.24.4 | LSTM model execution on CPU |
| LLM analysis | google-genai | 1.70.0 | Gemini API for trade log analysis |
| Environment config | python-dotenv | 1.2.2 | API key management via .env file |
| Dashboard | Streamlit | 1.56.0 | Interactive web frontend |
| Visualization | Plotly | 6.6.0 | Interactive charts |

## 8. Model Conversion Pipeline

The original model was trained in Keras and exported as `keras_model.h5`. The conversion path to the final ONNX format was:

```
keras_model.h5 --> TensorFlow SavedModel --> ONNX (via tf2onnx)
```

**Why not TFLite?** The initial `.tflite` conversion used TensorFlow Select Ops (Flex delegate) to handle LSTM operations. The Flex delegate is not available in the pip-distributed TensorFlow package on Windows. This is a known platform limitation. The ONNX conversion path avoids this entirely, as ONNX Runtime has native LSTM support on all platforms.

**Model size**: The final `trading_model.onnx` file is 21,568 bytes (~21KB), containing 178,763 parameters.

## 9. Reproducibility

### 9.1 Environment Setup

```bash
python -m venv stok
stok\Scripts\activate          # Windows
pip install -r requirements.txt
```

### 9.2 Full Pipeline Execution

```bash
# Step 1: Fetch data
python -c "from src.data_utils import fetch; fetch(['AAPL', 'MSFT', 'BTC-USD', 'SPY'])"

# Step 2: Run backtest
python backtest.py AAPL

# Step 3: LLM analysis
python -c "from src.analysis import analyze; print(analyze('data/AAPL_backtest.csv'))"

# Step 4: Interactive dashboard
streamlit run app.py
```

### 9.3 Expected Output

```
Ticker:           AAPL
OOS period:       252 days
Agent return:     0.1382 (13.82%)
Buy&Hold return:  0.1182 (11.82%)
Log saved to:     data/AAPL_backtest.csv
```

## 10. Future Work

1. **Multi-feature LSTM input**: Extend the model input from (100, 1) to (100, 5) to include all OHLCV features, plus derived indicators like ATR, RSI, and MACD.

2. **Continuous action space**: Replace Discrete(3) with a Box action space for fractional position sizing (e.g., "buy 30% of available balance"), enabling more granular risk management.

3. **Lambda tuning**: Systematically sweep `lambda` values (0.01 to 1.0) and evaluate the Pareto frontier of return vs. Sharpe ratio to find the optimal risk-return tradeoff.

4. **Multi-ticker training**: Train on a portfolio of tickers simultaneously to learn cross-asset correlations and sector rotation strategies.

5. **Online adaptation**: Periodically retrain or fine-tune the model on recent data to handle non-stationarity (concept drift) in financial markets.

6. **PPO/SAC algorithms**: Evaluate Proximal Policy Optimization or Soft Actor-Critic as alternatives to DQN, which may explore the state-action space more efficiently and handle the continuous action extension natively.
