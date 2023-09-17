import numpy as np
import matplotlib.pyplot as plt


class Strategy:

    def __init__(self, name, returns):
        self.name = name
        self.returns = returns
        self.pnl = 10000*np.c_[np.array([100.]), 100.*np.cumprod(1.+returns.to_numpy()).reshape(1,-1)].flatten()

    def volatility(self):
        return np.std(self.returns)*np.sqrt(252)

    def sharpe_ratio(self):
        return np.mean(self.returns)*np.sqrt(252)/np.std(self.returns)

    def max_dd(self):
        return 1. - np.min(np.flip(np.minimum.accumulate(np.flip(self.pnl)))/self.pnl)

    def max_dd_2(self):
        return np.max(1. - self.pnl/np.maximum.accumulate(self.pnl))

