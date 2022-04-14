"""
********************************************************************************
main file to execute
********************************************************************************
"""

import time
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

from pinn import PINN
from config_gpu import config_gpu
from params import params
from prp_dat import prp_dat
from plot_sol import *
from fdm import FDM

def main():
    # gpu confiuration
    config_gpu(gpu_flg = 1)

    # params
    f_in, f_out, width, depth, \
    w_init, b_init, act, \
    lr, opt, \
    f_scl, laaf, c, \
    w_ini, w_bnd, w_pde, BC, \
    f_mntr, r_seed, \
    n_epch, n_btch, c_tol = params()

    # domain
    tmin = 0.; tmax = 10.; nt = int(5e2) + 1
    xmin = 0.; xmax =  5.; nx = int(1e2) + 1
    ymin = 0.; ymax =  5.; ny = int(1e2) + 1
    t_ = np.linspace(tmin, tmax, nt)
    x_ = np.linspace(xmin, xmax, nx)
    y_ = np.linspace(ymin, ymax, ny)
    dt = t_[1] - t_[0]
    dx = x_[1] - x_[0]
    dy = y_[1] - y_[0]
    cfl = c * dt / dx
    print("CFL number:", cfl)

    x, y = np.meshgrid(x_, y_)
    u    = np.empty((nt, nx, ny))
    print("tmin: %.3f, tmax: %.3f, nt: %d, dt: %.3e" % (tmin, tmax, nt, dt))
    print("xmin: %.3f, xmax: %.3f, nx: %d, dx: %.3e" % (xmin, xmax, nx, dx))
    print("ymin: %.3f, ymax: %.3f, ny: %d, dy: %.3e" % (ymin, ymax, ny, dy))

    # FDM simulation
    u_FDM = FDM(xmin, xmax, nx, dx, 
                ymin, ymax, ny, dy, 
                nt, dt, 
                x, y, u, c, BC)

    # prep data
    TX, lb, ub, \
    t_ini, x_ini, y_ini, u_ini, \
    t_bnd, x_bnd, y_bnd, \
    t_pde, x_pde, y_pde = prp_dat(t_, x_, y_, 
                                    N_ini = int(5e3), N_bnd = int(1e4), N_pde = int(3e4))

    pinn = PINN(t_ini, x_ini, y_ini, u_ini, 
                t_bnd, x_bnd, y_bnd, 
                t_pde, x_pde, y_pde, 
                f_in, f_out, width, depth, 
                w_init, b_init, act, 
                lr, opt, 
                f_scl, laaf, c, 
                w_ini, w_bnd, w_pde, BC, 
                f_mntr, r_seed)

    t0 = time.time()
    with tf.device("/device:GPU:0"):
        pinn.train(epoch = n_epch, batch = n_btch, tol = c_tol)
    t1 = time.time()
    elps = t1 - t0
    print(">>>>> elapse time for training (sec):", elps)
    print(">>>>> elapse time for training (min):", elps / 60.)

    # inference
    x_inf, y_inf = np.meshgrid(x_, y_)
    x_inf, y_inf = x_inf.reshape(-1, 1), y_inf.reshape(-1, 1)
    elps = 0
    for t in t_:
        t_inf = np.ones_like(x_inf) * t
        t0 = time.time()
        u_, gv_ = pinn.infer(t_inf, x_inf, y_inf)
        t1 = time.time()
        temp = t1 - t0
        elps += temp
    print(">>>>> elapse time for inference (sec):", elps)
    print(">>>>> elapse time for inference (min):", elps / 60.)

    # x_inf = np.unique(TX[:,1:2])
    # y_inf = np.unique(TX[:,2:3])
    # x_inf, y_inf = np.meshgrid(x_inf, y_inf)
    # x_inf, y_inf = x_inf.reshape(-1, 1), y_inf.reshape(-1, 1)
    # elps = 0.
    # for n in range(nt):
    #     if n % 100 == 0:
    #         print("currently", n)
    #     t = n * dt   # convert to real time
    #     u_fdm = u_FDM[n,:,:]
    #     n = np.array([n])
    #     t_inf = np.unique(TX[:,0:1])
    #     t_inf = np.tile(t_inf.reshape(-1, 1), (1, x_inf.shape[0])).T[:,n]
    #     t0 = time.time()
    #     u_, gv_ = pinn.infer(t_inf, x_inf, y_inf)
    #     t1 = time.time()
    #     temp = t1 - t0
    #     elps += temp
    # print(">>>>> elapse time for inference (sec):", elps)
    # print(">>>>> elapse time for inference (min):", elps / 60.)

    plt.figure(figsize = (8, 4))
    plt.plot(pinn.ep_log, pinn.loss_log,     alpha = .7, linestyle = "-",  label = "loss", c = "k")
    plt.plot(pinn.ep_log, pinn.loss_ini_log, alpha = .5, linestyle = "--", label = "loss_ini")
    plt.plot(pinn.ep_log, pinn.loss_bnd_log, alpha = .5, linestyle = "--", label = "loss_bnd")
    plt.plot(pinn.ep_log, pinn.loss_pde_log, alpha = .5, linestyle = "--", label = "loss_pde")
    plt.yscale("log")
    plt.xlabel("epoch")
    plt.ylabel("loss")
    plt.legend(loc = "upper right")
    plt.grid(alpha = .5)
    plt.show()

    for n in range(nt):
        if n % (int(nt / 20)) == 0:
            t = n * dt   # convert to real time
            u_fdm = u_FDM[n,:,:]
            n = np.array([n])
            t_inf = np.unique(TX[:,0:1])
            x_inf = np.unique(TX[:,1:2])
            y_inf = np.unique(TX[:,2:3])
            x_inf, y_inf = np.meshgrid(x_inf, y_inf)
            x_inf, y_inf = x_inf.reshape(-1, 1), y_inf.reshape(-1, 1)
            t_inf = np.tile(t_inf.reshape(-1, 1), (1, x_inf.shape[0])).T[:,n]
            u_, gv_ = pinn.infer(t_inf, x_inf, y_inf)

            fig = plt.figure(figsize=(16, 4))
            ax = fig.add_subplot(1, 1, 1, projection = "3d")
            ax.plot_surface(x, y, u_fdm, vmin = -1., vmax = 1.)
            ax.set_xlim(xmin, xmax)
            ax.set_ylim(ymin, ymax)
            ax.set_zlim(-1., 1.)

            ax = fig.add_subplot(1, 2, 2, projection = "3d")
            ax.plot_surface(x, y, u_.numpy().reshape(nx, ny), vmin = -1., vmax = 1.)
            ax.set_xlim(xmin, xmax)
            ax.set_ylim(ymin, ymax)
            ax.set_zlim(-1., 1.)

            u_diff = u_fdm - u_.numpy().reshape(nx, ny)
            u_l2  = np.linalg.norm(u_diff, ord=2) / np.linalg.norm(u_fdm, ord=2)
            u_mse = np.mean(np.square(u_diff)) / np.sqrt(nx * ny)
            u_sem = np.std (np.square(u_diff), ddof = 1) / np.sqrt(nx * ny)
            print("t: %.3f, l2: %.3e, mse: %.3e, sem: %.3e" % (t, u_l2, u_mse, u_sem))

if __name__ == "__main__":
    main()
