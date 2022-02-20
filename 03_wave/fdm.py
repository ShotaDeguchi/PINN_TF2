"""
********************************************************************************
FDM for 2D wave equation
********************************************************************************
"""

import numpy as np
import matplotlib.pylab as plt

def FDM(xmin, xmax, nx, dx, 
        ymin, ymax, ny, dy, 
        nt, dt, 
        x, y, u, c, BC):
    # FDM for reference
    u[0, :, :] = np.exp(-(x - 3) ** 2) * np.exp(-(y - 3) ** 2)
    u[1, :, :] = u[0, :, :]
    for n in range(1, nt - 1):
        if n % int(1e2) == 0:
            print(">>>>> FDM computing... n =", n)
        for i in range(1, nx - 1):
            for j in range(1, ny - 1):
                u[n + 1, i, j] = 2 * u[n, i, j] - u[n - 1, i, j] \
                                + (c * dt / dx) ** 2 * (u[n, i + 1, j] - 2 * u[n, i, j] + u[n, i - 1, j]) \
                                + (c * dt / dy) ** 2 * (u[n, i, j + 1] - 2 * u[n, i, j] + u[n, i, j - 1])
        if BC == "Dir":
            for i in range(1, nx - 1):
                u[n + 1, i,  0] = 0.
                u[n + 1, i, -1] = 0.
            for j in range(1, ny - 1):
                u[n + 1,  0, j] = 0.
                u[n + 1, -1, j] = 0.
            u[n,  0,  0] = 0.
            u[n, -1,  0] = 0.
            u[n,  0, -1] = 0.
            u[n, -1, -1] = 0.
        elif BC == "Neu":
            for i in range(1, nx - 1):
                u[n + 1, i,  0] = u[n + 1, i,  1]
                u[n + 1, i, -1] = u[n + 1, i, -2]
            for j in range(1, ny - 1):
                u[n + 1,  0, j] = u[n + 1,  1, j]
                u[n + 1, -1, j] = u[n + 1, -2, j]
            u[n,  0,  0] = (u[n,  1,  0] + u[n,  0,  1]) / 2
            u[n, -1,  0] = (u[n, -2,  0] + u[n, -1,  1]) / 2
            u[n,  0, -1] = (u[n,  1, -1] + u[n,  0, -2]) / 2
            u[n, -1, -1] = (u[n, -2, -1] + u[n, -1, -2]) / 2

    # plot FDM solutions
    fig  = plt.figure(figsize = (16, 4))
    # snapshot 1
    ax   = fig.add_subplot(1, 3, 1, projection = "3d")
    surf = ax.plot_surface(x, y, u[0, :, :], cmap = "coolwarm", linewidth = 0, vmin = -.5, vmax = .5)
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)
    ax.set_zlim(-1, 1)
    ax.set_xlabel("x", fontstyle = "italic")
    ax.set_ylabel("y", fontstyle = "italic")
    ax.set_zlabel("u (x, y)", fontstyle = "italic")
    # snapshot 2
    ax   = fig.add_subplot(1, 3, 2, projection = "3d")
    surf = ax.plot_surface(x, y, u[100, :, :], cmap = "coolwarm", linewidth = 0, vmin = -.5, vmax = .5)
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)
    ax.set_zlim(-1, 1)
    ax.set_xlabel("x", fontstyle = "italic")
    ax.set_ylabel("y", fontstyle = "italic")
    ax.set_zlabel("u (x, y)", fontstyle = "italic")
    # snapshot 3
    ax   = fig.add_subplot(1, 3, 3, projection = "3d")
    surf = ax.plot_surface(x, y, u[-1, :, :], cmap = "coolwarm", linewidth = 0, vmin = -.5, vmax = .5)
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)
    ax.set_zlim(-1, 1)
    ax.set_xlabel("x", fontstyle = "italic")
    ax.set_ylabel("y", fontstyle = "italic")
    ax.set_zlabel("u (x, y)", fontstyle = "italic")
    plt.show()

    return u
