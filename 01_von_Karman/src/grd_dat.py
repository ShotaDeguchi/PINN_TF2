"""
********************************************************************************
make grid
********************************************************************************
"""

import numpy as np

def grd_dat(xmin, xmax, nx, 
            ymin, ymax, ny):
    x, y = np.linspace(xmin, xmax, nx), np.linspace(ymin, ymax, ny)
    x, y = np.meshgrid(x, y)

    dx = (xmax - xmin) / (nx - 1)
    dy = (ymax - ymin) / (ny - 1)

    print("xmin: %.3e, xmax: %.3e, nx: %d, dx: %.3e" \
            % (xmin, xmax, nx, dx))
    print("ymin: %.3e, ymax: %.3e, ny: %d, dy: %.3e" \
            % (ymin, ymax, ny, dy))

    return x, y


