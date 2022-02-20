"""
POD: Proper Orthogonal Decomposition
"""

import numpy as np
import pandas as pd
from scipy.interpolate import griddata
import matplotlib.pyplot as plt
from figs import *

path = "../input/"
name = "data_"
tail = ".csv"

t = np.array([])
x = np.array([])
y = np.array([])
u = np.array([])
v = np.array([])
w = np.array([])
p = np.array([])

T = 250
for k in range(T):
    if k == 0:
        data = pd.read_csv(path + name + str(k) + tail, header = 0)
        data = data.values
        t = np.append(t, data[:, 0])
        x = np.append(x, data[:, 5])
        y = np.append(y, data[:, 6])
        u = np.append(u, data[:, 8])
        v = np.append(v, data[:, 9])
        w = np.append(w, data[:, 4])
        p = np.append(p, data[:, 1])
    else:
        data = pd.read_csv(path + name + str(k) + tail, header = 0)
        data = data.values
        t = np.c_[t, data[:, 0]]
        x = np.c_[x, data[:, 5]]
        y = np.c_[y, data[:, 6]]
        u = np.c_[u, data[:, 8]]
        v = np.c_[v, data[:, 9]]
        w = np.c_[w, data[:, 4]]
        p = np.c_[p, data[:, 1]]
        if k % 20 == 0:
            print("loading data at", k)

# convert to real time
t = t / 10.

# snip out
XX = x[:,0:200]
YY = y[:,0:200]
TT = t[:,0:200]
UU = u[:,0:200]
VV = v[:,0:200]
WW = w[:,0:200]
PP = p[:,0:200]
res_space = XX.shape[0]
res_time  = XX.shape[1]

# random sampling
XY = np.c_[XX[:,0], YY[:,0]]
N_sample = 300
print("N_sample", N_sample)
x_rand = np.random.choice(XY[:,0], N_sample)
y_rand = np.random.choice(XY[:,1], N_sample)

lb = XY.min(0)
ub = XY.max(0)
nx = 200
x = np.linspace(lb[0], ub[0], nx)
y = np.linspace(lb[1], ub[1], nx)
X, Y = np.meshgrid(x, y)
phi = griddata(XY, UU[:,10].flatten(), (X, Y), method = "cubic")

plt.figure(figsize = (7, 4))
plt.pcolor(X, Y, phi, cmap = "coolwarm", shading = "auto", vmin = -.3, vmax = 1.5)
plt.colorbar(ticks = np.arange(-.3, 1.5 + .001, .3))
plt.scatter(x_rand, y_rand, marker = "x", alpha = .7, c = "k")
plt.xticks(np.arange(-10, 10, 1))
plt.yticks(np.arange(-10, 10, 1))
plt.xlim(1, 8)
plt.ylim(-2, 2)
plt.xlabel("x")
plt.ylabel("y")
plt.grid(alpha = .5)
plt.show()

# POD
u_left, u_sing, u_right = np.linalg.svd(UU, full_matrices=False, compute_uv=True)   # horizontal vel
v_left, v_sing, v_right = np.linalg.svd(VV, full_matrices=False, compute_uv=True)   # vertical vel
w_left, w_sing, w_right = np.linalg.svd(WW, full_matrices=False, compute_uv=True)   # vorticity

# for m in range(3):
#     lb = XY.min(0)
#     ub = XY.max(0)
#     nx = 200
#     x = np.linspace(lb[0], ub[0], nx)
#     y = np.linspace(lb[1], ub[1], nx)
#     X, Y = np.meshgrid(x, y)
#     phi = griddata(XY, u_left[:,m].flatten(), (X, Y), method = "cubic")
#     plt.figure(figsize = (8, 4))
#     plt.pcolor(X, Y, phi, cmap = "coolwarm", shading = "auto", vmin = -.03, vmax = .03)
#     plt.colorbar(ticks = np.arange(-.03, .03 + .001, .015))
#     plt.xlabel("x")
#     plt.ylabel("y")
#     plt.title(r"$ \phi $")
#     plt.show()

crit = .01
idx_pod = []
for i in range(res_space):
    if crit < abs(u_left[i,1]):
        idx_pod.append(i)

# re-sampling to match N_sample
idx_pod = np.random.choice(idx_pod, N_sample)
XY_pod = XY.copy()
x_pod = XY_pod[idx_pod,0]
y_pod = XY_pod[idx_pod,1]
plt.figure(figsize = (7, 4))
plt.scatter(x_pod, y_pod, marker = ".", alpha = .7)
plt.grid(alpha = .5)
plt.show()

plt.figure(figsize = (7, 4))
plt.plot(np.arange(0, len(u_right[0,:]), 1), u_right[6,:])
# plt.plot(np.arange(0, len(u_right[0,:]), 1), u_right[0,:], label = "mode0")
# plt.plot(np.arange(0, len(u_right[0,:]), 1), u_right[1,:], label = "mode1")
# plt.plot(np.arange(0, len(u_right[0,:]), 1), u_right[2,:], label = "mode2")
# plt.legend(loc = "upper right")
plt.ylim(-.15, .15)
plt.show()

crit = .01
m_max = 3
for m in range(m_max):
    idx_pod = []
    for i in range(res_space):
        if crit < abs(u_left[i,m]):
            idx_pod.append(i)
    idx_pod = np.random.choice(idx_pod, int(N_sample / m_max))
    print("len(idx_pod)", len(idx_pod))
    XY_pod = XY.copy()
    x_pod = XY_pod[idx_pod,0]
    y_pod = XY_pod[idx_pod,1]

    lb = XY.min(0)
    ub = XY.max(0)
    nx = 200

    x = np.linspace(lb[0], ub[0], nx)
    y = np.linspace(lb[1], ub[1], nx)
    X, Y = np.meshgrid(x, y)
    phi = griddata(XY, u_left[:,m].flatten(), (X, Y), method = "cubic")

    plt.figure(figsize = (7, 4))
    plt.pcolor(X, Y, phi, cmap = "coolwarm", shading = "auto", vmin = -.03, vmax = .03)
    plt.colorbar(ticks = np.arange(-.03, .03 + .001, .015))
    plt.scatter(x_pod, y_pod, marker = "x", alpha = .7, c = "k")
    plt.xticks(np.arange(-10, 10, 1))
    plt.yticks(np.arange(-10, 10, 1))
    plt.xlim(1, 8)
    plt.ylim(-2, 2)
    plt.xlabel("x")
    plt.ylabel("y")
    plt.grid(alpha = .5)
    plt.show()

