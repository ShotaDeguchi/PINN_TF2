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

    # with tf.device("/device:CPU:0"):
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
    print("elapsed time for PINN inference (sec):", elps)
    plt.figure(figsize=(16, 3))
    plt.subplot(1,2,1)
    plt.imshow(u_hat.numpy().reshape(nx, nt), cmap="turbo", aspect=5, interpolation="bilinear", vmin=-1, vmax=1)
    plt.colorbar()
    plt.subplot(1,2,2)
    plt.imshow(gv_hat.numpy().reshape(nx, nt), cmap="coolwarm", aspect=5, interpolation="bilinear", vmin=-.1, vmax=.1)
    plt.colorbar()
    plt.show()

    # FDM 
    t_num = np.linspace(tmin, tmax, nt, dtype = "float32")
    x_num = np.linspace(xmin, xmax, nx, dtype = "float32")
    dt_num = t_num[1] - t_num[0]
    dx_num = x_num[1] - x_num[0]
    D_num = pinn.D.numpy()
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
    print("elapsed time for FDM simulation (sec):", elps)

    # comparison
    u_hat_  = u_hat .numpy().reshape(nx, nt)
    gv_hat_ = gv_hat.numpy().reshape(nx, nt)
    for n in range(nt):
        if n % int(nt/5) == 0:
            norm_u = np.linalg.norm(u_num[:,n] - u_hat_[:,n], 2)
            mse_u  = np.mean(np.square(u_num[:,n] - u_hat_[:,n]))
            sem_u  = np.std (np.square(u_num[:,n] - u_hat_[:,n]), ddof = 1) / np.sqrt(u_hat_[:,n].shape[0])
            norm_gv= np.linalg.norm(gv_hat_[:,n], 2)
            mse_gv = np.mean(np.square(gv_hat_[:,n]))
            sem_gv = np.std (np.square(gv_hat_[:,n]), ddof = 1) / np.sqrt(gv_hat_[:,n].shape[0])
            print("t: %.3f, norm_u: %.6e, mse_u: %.6e, sem_u: %.6e, norm_gv: %.6e, mse_gv: %.6e, sem_gv: %.6e" 
                % (n / nt, norm_u, mse_u, sem_u, norm_gv, mse_gv, sem_gv))
            
            plt.figure(figsize=(4, 4))
            plt.plot(x_num, u_num [:,n], label = "FDM",  color = "k", linestyle="-",  linewidth=3) 
            plt.plot(x_num, u_hat_[:,n], label = "PINN", color = "r", linestyle="--", linewidth=3)
            plt.xlim(-1.2, 1.2)
            plt.ylim(-1.2, 1.2)
            plt.xlabel("x", fontstyle = "italic")
            plt.ylabel("u", fontstyle = "italic")
            plt.grid(alpha = .5)
            plt.legend(loc = "lower right")
            plt.show()

if __name__ == "__main__":
    main()
