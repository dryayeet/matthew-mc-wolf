import numpy as np
import gymnasium as gym
from gymnasium import spaces

class StockTradingEnv(gym.Env):
    metadata = {'render_modes': []}

    def __init__(self, d, dr, w=100, m=50, lam=0.1, fee=0.001, b0=10000.0):
        super().__init__()
        self.d = d          # scaled OHLCV (T,5) float32
        self.dr = dr        # raw OHLCV (T,5) float32
        self.w = w          # window size
        self.m = m          # sharpe lookback
        self.lam = lam      # sharpe weight
        self.fee = fee      # transaction cost
        self.b0 = b0        # initial balance
        self.action_space = spaces.Discrete(3)
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf,
            shape=(w * 5 + 3,), dtype=np.float32
        )

    def _obs(self):
        wnd = self.d[self.t - self.w:self.t].ravel()
        meta = np.array([self.bal / self.b0, self.sh / 100.0, self.nw / self.b0], dtype=np.float32)
        return np.concatenate([wnd, meta])

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.t = self.w
        self.bal = self.b0
        self.sh = 0.0
        self.nw = self.b0
        self.rh = np.zeros(self.m, dtype=np.float32)
        self.ri = 0
        self.steps = 0
        return self._obs(), {}

    def step(self, a):
        prev = self.nw
        p = self.dr[self.t, 3]  # raw close price

        if a == 1:  # Buy
            mx = int(self.bal // (p * (1 + self.fee)))
            if mx < 1:
                a = 0
            else:
                self.bal -= mx * p * (1 + self.fee)
                self.sh += mx
        elif a == 2:  # Sell
            if self.sh < 1:
                a = 0
            else:
                self.bal += self.sh * p * (1 - self.fee)
                self.sh = 0.0

        self.t += 1
        done = self.t >= len(self.d) - 1
        if done:
            self.t = len(self.d) - 1

        self.nw = self.bal + self.sh * self.dr[self.t, 3]

        # log return
        r_log = float(np.log(self.nw / (prev + 1e-8) + 1e-8))

        # ring buffer update
        self.rh[self.ri % self.m] = r_log
        self.ri += 1
        self.steps += 1

        # sharpe component
        if self.steps >= self.m:
            mu = self.rh.mean()
            sd = self.rh.std() + 1e-8
            sharpe = float(mu / sd)
        else:
            sharpe = 0.0

        reward = r_log + self.lam * sharpe

        # early termination if bankrupt
        if self.nw < 1.0:
            done = True
            reward -= 1.0

        return self._obs(), float(reward), done, False, {"nw": self.nw, "action": a}
