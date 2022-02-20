"""
********************************************************************************
plots solutions
********************************************************************************
"""

import numpy as np
from scipy.interpolate import griddata
import matplotlib.pyplot as plt

def plot_sol0(X_star, phi1):
    lb = X_star.min(0)
    ub = X_star.max(0)
    nn = 200
    x, y = np.linspace(lb[0], ub[0], nn), np.linspace(lb[1], ub[1], nn); x, y = np.meshgrid(x, y)
    PHI_I = griddata(X_star, phi1.flatten(), (x, y), method = "linear")

    plt.figure(figsize = (5, 4))
    plt.xlabel("x", fontstyle = "italic")
    plt.ylabel("y", fontstyle = "italic")
    plt.title("$ \phi $")
    plt.xticks(np.arange(-10, 10, 1)); plt.yticks(np.arange(-10, 10, 1))
    plt.pcolor(x, y, PHI_I, cmap = "coolwarm", shading = "auto")
    plt.colorbar()
    plt.show()
    
def plot_sol1(X_star, phi1, v0, v1, ticks):
    lb = X_star.min(0)
    ub = X_star.max(0)
    nn = 200
    x, y = np.linspace(lb[0], ub[0], nn), np.linspace(lb[1], ub[1], nn); x, y = np.meshgrid(x, y)
    PHI_I = griddata(X_star, phi1.flatten(), (x, y), method = "linear")

    plt.figure(figsize = (5, 4))
    plt.xlabel("x", fontstyle = "italic")
    plt.ylabel("y", fontstyle = "italic")
    plt.xticks(np.arange(-10, 10, 1)); plt.yticks(np.arange(-10, 10, 1))
    plt.pcolor(x, y, PHI_I, cmap = "coolwarm", shading = "auto", vmin = v0, vmax = v1)
    plt.colorbar(ticks = np.arange(-10, 10, ticks))
    plt.show()

def plot_sol2(X_star, phi1, phi2, v0, v1, ticks):
    lb = X_star.min(0)
    ub = X_star.max(0)
    nn = 200
    x, y = np.linspace(lb[0], ub[0], nn), np.linspace(lb[1], ub[1], nn); x, y = np.meshgrid(x, y)
    PHI_I   = griddata(X_star, phi1.flatten(), (x, y), method = "linear")
    PHI_II  = griddata(X_star, phi2.flatten(), (x, y), method = "linear")

    plt.figure(figsize = (16, 3))

    plt.subplot(1, 3, 1)
    plt.xlabel("x")
    plt.ylabel("y")
    plt.title("$ \phi_1 $")
    plt.xticks(np.arange(-10, 10, 1))
    plt.yticks(np.arange(-10, 10, 1))
    plt.pcolor(x, y, PHI_I, cmap = "coolwarm", shading = "auto", vmin = v0, vmax = v1)
    plt.colorbar(ticks = np.arange(-10, 10, ticks))
    plt.subplot(1, 3, 2)
    plt.xlabel("x")
    plt.ylabel("y")
    plt.title("$ \phi_2 $")
    plt.xticks(np.arange(-10, 10, 1))
    plt.yticks(np.arange(-10, 10, 1))
    plt.pcolor(x, y, PHI_II, cmap = "coolwarm", shading = "auto", vmin = v0, vmax = v1)
    plt.colorbar(ticks = np.arange(-10, 10, ticks))
    plt.show()

def plot_diff(X_star, phi1, phi2, v0, v1, vt, d0, d1, dt):
    lb = X_star.min(0)
    ub = X_star.max(0)
    nn = 200
    x, y = np.linspace(lb[0], ub[0], nn), np.linspace(lb[1], ub[1], nn); x, y = np.meshgrid(x, y)
    PHI_I   = griddata(X_star, phi1.flatten(), (x, y), method = "linear")
    PHI_II  = griddata(X_star, phi2.flatten(), (x, y), method = "linear")
    PHI_III = griddata(X_star, (phi2 - phi1).flatten(), (x, y), method = "linear")

    plt.figure(figsize = (16, 2))

    plt.subplot(1, 3, 1)
    plt.xlabel("x", fontstyle = "italic")
    plt.ylabel("y", fontstyle = "italic")
    plt.title("$ \phi_1 $")
    plt.xticks(np.arange(-10, 10, 1))
    plt.yticks(np.arange(-10, 10, 1))
    plt.pcolor(x, y, PHI_I, cmap = "coolwarm", shading = "auto", vmin = v0, vmax = v1)
    plt.colorbar(ticks = np.arange(-3, 3, vt))

    plt.subplot(1, 3, 2)
    plt.xlabel("x", fontstyle = "italic")
    plt.ylabel("y", fontstyle = "italic")
    plt.title("$ \phi_2 $")
    plt.xticks(np.arange(-10, 10, 1))
    plt.yticks(np.arange(-10, 10, 1))
    plt.pcolor(x, y, PHI_II, cmap = "coolwarm", shading = "auto", vmin = v0, vmax = v1)
    plt.colorbar(ticks = np.arange(-3, 3, vt))

    plt.subplot(1, 3, 3)
    plt.xlabel("x", fontstyle = "italic")
    plt.ylabel("y", fontstyle = "italic")
    plt.title("$ \phi_2 - \phi_1 $")
    plt.xticks(np.arange(-10, 10, 1))
    plt.yticks(np.arange(-10, 10, 1))
    plt.pcolor(x, y, PHI_III, cmap = "coolwarm", shading = "auto", vmin = d0, vmax = d1)
    plt.colorbar(ticks = np.arange(-3, 3, dt))
    plt.show()
