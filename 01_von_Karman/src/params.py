"""
********************************************************************************
all your parameters
********************************************************************************
"""

import sys
import os
import numpy as np
import tensorflow as tf

# network structure
in_dim  = 3
out_dim = 3
width = 2 ** 8   # 2 ** 6 = 64, 2 ** 8 = 256
depth = 5

# training setting
n_epch = int(5e4)
n_btch = 0
c_tol  = 1e-8

# dataset prep
N_trn = int(1e4)
N_val = int(5e3)
N_inf = int(1e3)

# optimization
w_init = "Glorot"
b_init = "zeros"
act = "tanh"
lr0 = 1e-2
gam = 1e-2
lrd_exp = tf.keras.optimizers.schedules.ExponentialDecay(
    initial_learning_rate = lr0, 
    decay_steps = n_epch, 
    decay_rate = gam, 
    staircase = False
    )
lrd_cos = tf.keras.optimizers.schedules.CosineDecay(
    initial_learning_rate = lr0, 
    decay_steps = n_epch, 
    alpha = gam
    )
lr = 1e-3   # 1e-3 / lrd_exp / lrd_cos
opt = "Adam"
f_scl = "minmax"
laaf = True
inv = False

# system param
rho = 1.
nu  = .01

# weight
w_nth = 1.
w_sth = 1. 
w_est = 1.
w_wst = 1. 
w_pde = 1.

# rarely change
f_mntr = 10
r_seed = 1234

def params():
    print("python    :", sys.version)
    print("tensorflow:", tf.__version__)
    print("rand seed :", r_seed)
    os.environ["PYTHONHASHSEED"] = str(r_seed)
    np.random.seed(r_seed)
    tf.random.set_seed(r_seed)

    return in_dim, out_dim, width, depth, \
        w_init, b_init, act, \
        lr, opt, \
        f_scl, laaf, inv, \
        rho, nu, \
        w_nth, w_sth, w_est, w_wst, w_pde, \
        f_mntr, r_seed, \
        n_epch, n_btch, c_tol, \
        N_trn, N_val, N_inf
