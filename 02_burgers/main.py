"""
********************************************************************************
main file to execute your program
********************************************************************************
"""

import tensorflow as tf

from pinn import PINN
from config_gpu import config_gpu
from prp_dat import func_u0, func_ub, prp_grd, prp_dataset
from params import params
from make_fig import plot_sol0, plot_sol1
from plot_hist import *

def main():
    config_gpu(gpu_flg = 1)

    tmin, tmax, nt =  0., 1., int(1e3) + 1
    xmin, xmax, nx = -1., 1., int(5e2) + 1
    t, x, TX = prp_grd(tmin, tmax, nt, 
                       xmin, xmax, nx)

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
                Rm = 2, Rn = 1, Rl = 100, depth = 5, activ = "tanh", BN = False, 
                w_init = "glorot_normal", b_init = "zeros", 
                lr = lr, opt = opt, w_0 = 1., w_b = 1., w_r = 1.,
                f_mntr = 10, r_seed = 1234)

    with tf.device("/device:GPU:0"):
        pinn.train(n_epch, n_btch, c_tol)

    plot_loss(pinn.ep_log, pinn.loss_log)

    u_hat, gv_hat = pinn.predict(t, x)
    plot_sol1(TX, u_hat.numpy(), -1, 1, .25)
    plot_sol1(TX, gv_hat.numpy(), -1, 1, .25)

if __name__ == "__main__":
    main()
