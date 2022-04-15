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

    tmin, tmax =  0., 1.
    xmin, xmax = -1., 1.

    in_dim, out_dim, width, depth, \
        w_init, b_init, act, \
        lr, opt, \
        f_scl, laaf, \
        rho, nu, \
        w_dat, w_pde, \
        f_mntr, r_seed, \
        n_epch, n_btch, c_tol, \
        N_0, N_b, N_r = params()

    t_0, x_0, t_b, x_b, t_r, x_r = prp_dataset(tmin, tmax, xmin, xmax, N_0, N_b, N_r)
    u_0 = func_u0(x_0)
    u_b = func_ub(x_b)

    pinn = PINN(t_0, x_0, u_0, 
                t_b, x_b, u_b, 
                t_r, x_r, 
                Rm = in_dim, Rn = out_dim, Rl = width, depth = depth, activ = "tanh", BN = False, 
                w_init = "glorot_normal", b_init = "zeros", 
                lr = lr, opt = opt, w_0 = 1., w_b = 1., w_r = 1.,
                f_mntr = 10, r_seed = 1234)
    with tf.device("/device:GPU:0"):
        pinn.train(n_epch, n_btch, c_tol)
    plot_loss(pinn.ep_log, pinn.loss_log)

    # PINN inference
    nt = int(1e3) + 1
    nx = int(1e2) + 1
    t, x, TX = prp_grd(
        tmin, tmax, nt, 
        xmin, xmax, nx
    )
    t0 = time.time()
    u_hat, gv_hat = pinn.predict(t, x)
    t1 = time.time()
    elps = t1 - t0
    print("elapsed time for PINN inference (sec):", elps)
    print("elapsed time for PINN inference (min):", elps / 60.)
    plot_sol1(TX, u_hat .numpy(), -1, 1, .25)
    plot_sol1(TX, gv_hat.numpy(), -1, 1, .25)

    # FDM approximation
    factor = 20
    nt = int(factor * (nt - 1)) + 1
    nx = int(factor * (nx - 1)) + 1
    t, x, TX = prp_grd(
        tmin, tmax, nt, 
        xmin, xmax, nx
    )
    t, x = np.linspace(tmin, tmax, nt), np.linspace(xmin, xmax, nx)
    dt, dx = t[1] - t[0], x[1] - x[0]
    nu = pinn.nu.numpy()
    u = np.zeros([nx, nt])
    # impose IC
    for i in range(nx):
        # u[:,0] = - np.sin(np.pi * x)
        u[:,0] = - np.sin(2* np.pi * x)
    # explicit time integration
    t0 = time.time()
    for n in range(nt - 1):
        for i in range(1, nx - 1):
            u[i, n + 1] = u[i, n] \
                - dt / dx * u[i, n] * (u[i, n] - u[i - 1, n]) \
                + nu * dt / dx ** 2 * (u[i + 1, n] - 2 * u[i, n] + u[i - 1, n])
    t1 = time.time()
    elps = t1 - t0
    print("elapsed time for FDM simulation (sec):", elps)
    print("elapsed time for FDM simulation (sec):", elps / 60.)
    plot_sol1(TX, u.reshape(-1, 1), -1, 1, .25)

if __name__ == "__main__":
    main()
