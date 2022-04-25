"""
********************************************************************************
make figures
********************************************************************************
"""

import numpy as np
from scipy.interpolate import griddata
import matplotlib.pyplot as plt

def plt_sol0(XY, u, width, height, cmap):
    lb = XY.min(0)
    ub = XY.max(0)
    nx = 200
    x = np.linspace(lb[0], ub[0], nx)
    y = np.linspace(lb[1], ub[1], nx)
    X, Y = np.meshgrid(x, y)
    U = griddata(XY, u.flatten(), (X, Y), method = "linear")
    plt.figure(figsize = (width, height))
    plt.pcolor(X, Y, U, cmap = cmap, shading = "auto")
    plt.colorbar()
    plt.xlabel("x")
    plt.ylabel("y")
    plt.title(r"$ \phi $")
    plt.show()

def plt_sol1(XY, u, width, height, cmap, v0, v1, vt):
    lb = XY.min(0)
    ub = XY.max(0)
    nx = 200
    x = np.linspace(lb[0], ub[0], nx)
    y = np.linspace(lb[1], ub[1], nx)
    X, Y = np.meshgrid(x, y)
    U = griddata(XY, u.flatten(), (X, Y), method = "cubic")
    plt.figure(figsize = (width, height))
    plt.pcolor(X, Y, U, cmap = cmap, shading = "auto", vmin = v0, vmax = v1)
    plt.colorbar(ticks = np.arange(v0, v1 + .001, vt))
    plt.xlabel("x")
    plt.ylabel("y")
    plt.title(r"$ \phi $")
    plt.show()

def plt_diff(x, u1, u2, height, width, cmap, v0, v1, vt):
    lb = x.min(0)
    ub = x.max(0)
    nx = 200
    x = np.linspace(lb[0], ub[0], nx)
    y = np.linspace(lb[1], ub[1], nx)
    X, Y = np.meshgrid(x, y)
    U1 = griddata(x, u1.flatten(), (X, Y), method = "cubic")
    U2 = griddata(x, u2.flatten(), (X, Y), method = "cubic")
    U3 = griddata(x, (u1 - u2).flatten(), (X, Y), method = "cubic")
    plt.figure(figsize = (height, width))
    plt.subplot(1, 3, 1)
    plt.pcolor(X, Y, U1, cmap = cmap, shading = "auto", vmin = v0, vmax = v1)
    plt.colorbar(ticks = np.arange(v0, v1 + .001, vt))
    plt.xlabel("x")
    plt.ylabel("y")
    plt.title(r"$ \phi_1 $")
    plt.subplot(1, 3, 2)
    plt.pcolor(X, Y, U2, cmap = cmap, shading = "auto", vmin = v0, vmax = v1)
    plt.colorbar(ticks = np.arange(v0, v1 + .001, vt))
    plt.xlabel("x")
    plt.ylabel("y")
    plt.title(r"$ \phi_2 $")
    plt.subplot(1, 3, 3)
    plt.pcolor(X, Y, U3, cmap = cmap, shading = "auto", vmin = v0, vmax = v1)
    plt.colorbar(ticks = np.arange(v0 / 10, (v1 + .001) / 10, vt / 10))
    plt.xlabel("x")
    plt.ylabel("y")
    plt.title(r"$ \phi_1 - \phi_2 $")
