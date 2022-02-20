"""
********************************************************************************
figs
********************************************************************************
"""

import numpy as np
from scipy.interpolate import griddata
import matplotlib.pyplot as plt

def plot_sol0(X_star, phi1):
    lb = X_star.min(0); ub = X_star.max(0); nn = 200
    x, y = np.linspace(lb[0], ub[0], nn), np.linspace(lb[1], ub[1], nn); x, y = np.meshgrid(x, y)
    PHI_I = griddata(X_star, phi1.flatten(), (x, y), method = "linear")

    plt.figure(figsize = (8, 4))
    plt.xlabel("t"); plt.ylabel("x")
    plt.xticks(np.arange(-30, 30, 1)); plt.yticks(np.arange(-30, 30, 1))
    plt.pcolor(x, y, PHI_I, cmap = "coolwarm", shading = "auto")
    plt.colorbar()
    plt.show()
    
def plot_sol1(X_star, phi1, v0, v1, ticks):
    lb = X_star.min(0); ub = X_star.max(0); nn = 200
    x, y = np.linspace(lb[0], ub[0], nn), np.linspace(lb[1], ub[1], nn); x, y = np.meshgrid(x, y)
    PHI_I = griddata(X_star, phi1.flatten(), (x, y), method = "linear")

    plt.figure(figsize = (8, 4))
    plt.xlabel("t"); plt.ylabel("x")
    plt.xticks(np.arange(-30, 30, 1)); plt.yticks(np.arange(-30, 30, 1))
    plt.pcolor(x, y, PHI_I, cmap = "coolwarm", shading = "auto", vmin = v0, vmax = v1)
    plt.colorbar(ticks = np.arange(-30, 30, ticks))
    plt.show()

