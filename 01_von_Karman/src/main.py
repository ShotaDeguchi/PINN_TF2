"""
********************************************************************************
main file to execute your program
********************************************************************************
"""

import tensorflow as tf

from pinn import PINN
from config_gpu import config_gpu
from prp_dat import prp_dat

from prp_dat import nth_dat, sth_dat, est_dat, wst_dat, pde_dat

from grd_dat import grd_dat
from params import params
from plot_hist import *
from figs import *
from cmpt_stats import *

def main():
    config_gpu(gpu_flg = 1)

    in_dim, out_dim, width, depth, \
            w_init, b_init, act, \
            lr, opt, \
            f_scl, laaf, inv, \
            rho, nu, \
            w_nth, w_sth, w_est, w_wst, w_pde, \
            f_mntr, r_seed, \
            n_epch, n_btch, c_tol, \
            N_trn, N_val, N_inf = params()
    
    x_trn, y_trn, t_trn, u_trn, v_trn, p_trn, \
    x_val, y_val, t_val, u_val, v_val, p_val, \
    x_inf, y_inf, t_inf, u_inf, v_inf, p_inf  \
        = prp_dat(N_trn, N_val, N_inf)

    xmin, xmax, nx = 0, 40, 401
    ymin, ymax, ny = 0, 10, 101
    tmin, tmax, nt = 0, 150, 151

    x_nth, y_nth, t_nth, u_nth, v_nth, p_nth = nth_dat(xmin, xmax, ymin, ymax, tmin, tmax, N_nth = int(1e4))
    x_sth, y_sth, t_sth, u_sth, v_sth, p_sth = sth_dat(xmin, xmax, ymin, ymax, tmin, tmax, N_sth = int(1e4))
    x_est, y_est, t_est, u_est, v_est, p_est = est_dat(xmin, xmax, ymin, ymax, tmin, tmax, N_est = int(1e4))
    x_wst, y_wst, t_wst, u_wst, v_wst, p_wst = wst_dat(xmin, xmax, ymin, ymax, tmin, tmax, N_wst = int(1e4))
    x_pde, y_pde, t_pde = pde_dat(xmin, xmax, ymin, ymax, tmin, tmax, N_pde = int(5e4))

    # plt.figure(figsize = (4, 4))
    # plt.scatter(x_nth, y_nth, label = "north bound")
    # plt.scatter(x_sth, y_sth, label = "south bound")
    # plt.scatter(x_est, y_est, label = "east  bound")
    # plt.scatter(x_wst, y_wst, label = "west  bound")
    # plt.legend()
    # plt.show()

    pinn = PINN(x_nth, y_nth, t_nth, u_nth, v_nth, p_nth, 
                x_sth, y_sth, t_sth, u_sth, v_sth, p_sth, 
                x_est, y_est, t_est, u_est, v_est, p_est, 
                x_wst, y_wst, t_wst, u_wst, v_wst, p_wst, 
                x_pde, y_pde, t_pde, 
                in_dim, out_dim, width, depth, 
                w_init, b_init, act, 
                lr, opt, 
                f_scl, laaf, inv, 
                rho, nu, 
                w_nth, w_sth, w_est, w_wst, w_pde, 
                f_mntr, r_seed
                )

    with tf.device("/device:GPU:0"):
        pinn.train(epoch = n_epch, batch = n_btch, tol = c_tol)
    
    # plot_loss(pinn.ep_log, pinn.loss_trn_log, pinn.loss_val_log)
    plt.figure(figsize = (8, 4))
    plt.plot(pinn.ep_log, pinn.loss_glb_log, alpha = .7, linestyle = "-", color = "k", label = "loss_glb")
    plt.plot(pinn.ep_log, pinn.loss_bnd_log, alpha = .7, linestyle = ":", color = "c", label = "loss_bnd")
    plt.plot(pinn.ep_log, pinn.loss_pde_log, alpha = .7, linestyle = ":", color = "m", label = "loss_pde")
    plt.yscale("log")
    plt.grid(alpha = .5)
    plt.legend(loc = "upper right")
    plt.show()

    x_star, y_star = np.linspace(xmin, xmax, nx), np.linspace(ymin, ymax, ny)
    x_star, y_star = np.meshgrid(x_star, y_star)
    x_star, y_star = x_star.flatten()[:,None], y_star.flatten()[:,None]
    t = 100
    t_star = np.ones_like(x_star) * t
    u_, v_, p_, gv_c_, gv_x_, gv_y_ = pinn.inference(x_star, y_star, t_star)

    XY = np.c_[x_star, y_star]
    plt_sol0(XY, u_.numpy(), width = 8, height = 4, cmap = "coolwarm")
    plt_sol0(XY, v_.numpy(), width = 8, height = 4, cmap = "coolwarm")
    plt_sol0(XY, p_.numpy(), width = 8, height = 4, cmap = "coolwarm")

    u_, v_, p_, gv_c_, gv_x_, gv_y_ = pinn.inference(x_inf, y_inf, t_inf)
    print("mse_u: %.6e, sem_u: %.6e" % fwd_stats(u_inf, u_, N_inf))
    print("mse_v: %.6e, sem_v: %.6e" % fwd_stats(v_inf, v_, N_inf))
    print("mse_p: %.6e, sem_p: %.6e" % fwd_stats(p_inf, p_, N_inf))
    print("mse_gv_c: %.6e, sem_gv_c: %.6e" % fwd_stats(np.zeros_like(gv_c_), gv_c_, N_inf))
    print("mse_gv_x: %.6e, sem_gv_x: %.6e" % fwd_stats(np.zeros_like(gv_x_), gv_x_, N_inf))
    print("mse_gv_y: %.6e, sem_gv_y: %.6e" % fwd_stats(np.zeros_like(gv_y_), gv_y_, N_inf))

    if inv == True:
        plot_inv(pinn.ep_log, lambda_ = 1., lambda_log = pinn.rho_log, name = "rho")
        plot_inv(pinn.ep_log, lambda_ = .01, lambda_log = pinn.nu_log, name = "nu")

        window = int(n_epch / 10)
        print("statistics of the last %d epochs:" % window)
        print("mean_rho: %.6e, std_rho: %.6e" % inv_stats(pinn.rho_log, window = window))
        print("mean_nu : %.6e, std_nu : %.6e" % inv_stats(pinn.nu_log,  window = window))
    else:
        pass

if __name__ == "__main__":
    main()