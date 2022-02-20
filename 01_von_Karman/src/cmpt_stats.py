"""
********************************************************************************
computes statistics
********************************************************************************
"""

import numpy as np

def fwd_stats(u, u_, n):
    mse = np.mean(np.square(u - u_))
    sem = np.std (np.square(u - u_), ddof = 1) / np.sqrt(n)
    return mse, sem

def inv_stats(theta_, window = 100):
    # theta_: list
    theta_ = theta_[-window:]
    mean = np.mean(theta_)
    std  = np.std (theta_, ddof = 1)
    return mean, std