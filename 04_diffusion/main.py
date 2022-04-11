"""
********************************************************************************
main file to execute your program
********************************************************************************
"""

import time
import numpy as np
import tensorflow as tf

from pinn import PINN
from config_gpu import config_gpu
from prp_dat import func_u0, func_ub, prp_grd, prp_dataset
from params import params
from make_fig import plot_sol0, plot_sol1
from plot_hist import *

def main():
    config_gpu(gpu_flg = 1)

    tmin, tmax, nt =  0., 1., int(2e3) + 1
    xmin, xmax, nx = -1., 1., int(2e2) + 1
    t, x, TX = prp_grd(tmin, tmax, nt, 
                       xmin, xmax, nx)

    in_dim, out_dim, width, depth, \
    w_init, b_init, act, \
    lr, opt, \
    w_0, w_b, w_r, \
    f_mntr, r_seed, \
    n_epch, n_btch, c_tol, \
    N_0, N_b, N_r = params()

    t_0, x_0, t_b, x_b, t_r, x_r = prp_dataset(tmin, tmax, xmin, xmax, N_0, N_b, N_r)
    u_0 = func_u0(x_0)
    u_b = func_ub(x_b)

    pinn = PINN(t_0, x_0, u_0, 
                t_b, x_b, u_b, 
                t_r, x_r, 
                Rm = in_dim, Rn = out_dim, Rl = width, depth = depth, activ = act, 
                w_init = w_init, b_init = b_init, 
                lr = lr, opt = opt, w_0 = w_0, w_b = w_b, w_r = w_r,
                f_mntr = f_mntr, r_seed = r_seed)

    with tf.device("/device:GPU:0"):
        pinn.train(n_epch, n_btch, c_tol)

    plt.figure(figsize=(8, 4))
    plt.plot(pinn.ep_log, pinn.loss_log, alpha=.7)
    plt.yscale("log")
    plt.xlabel("epoch")
    plt.ylabel("loss")
    plt.grid(alpha=.5)
    plt.show()

    # PINN inference
    t0 = time.time()
    u_hat, gv_hat = pinn.infer(t, x)
    t1 = time.time()
    elps = t1 - t0
    print("elapsed time for inference (sec):", elps)

    # FDM 
    t_num = np.linspace(tmin, tmax, nt, dtype = "float32")
    x_num = np.linspace(xmin, xmax, nx, dtype = "float32")
    dt_num = t_num[1] - t_num[0]
    dx_num = x_num[1] - x_num[0]
    D_num = pinn.D
    u_num = np.empty([nx, nt], dtype="float32")
    # initial condition
    for i in range(1, nx):
        u_num[i, 0] = .5 * np.sin(     np.pi * x_num[i]) \
                    + .3 * np.sin( 4 * np.pi * x_num[i]) \
                    + .1 * np.sin(16 * np.pi * x_num[i])
    # boundary condition
    u_num[ 0,:] = 0.
    u_num[-1,:] = 0.


    # FDM simulation
    t0 = time.time()
    for n in range(nt - 1):
        for i in range(1, nx - 1):
            u_num[i, n+1] =   u_num[i, n] \
                            + D_num * dt_num / dx_num ** 2 * (u_num[i+1, n] - 2 * u_num[i, n] + u_num[i-1, n])
    t1 = time.time()
    elps = t1 - t0
    print("elapsed time for FDM (sec):", elps)
   

if __name__ == "__main__":
    main()
