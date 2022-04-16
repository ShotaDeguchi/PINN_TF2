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
in_dim  = 2
out_dim = 1
width = 2 ** 6   # 2 ** 6 = 64, 2 ** 8 = 256
depth = 4

# training setting
n_epch = int(5e4)
n_btch = int(0)
c_tol  = 1e-8

# dataset prep
N_0 = int(1e3)   # evaluates initial condition
N_b = int(1e3)   # evaluates boundary condition
N_r = int(1e4)   # evaluates PDE residual within the domain


N_0 = int(2e4)   # evaluates initial condition
N_b = int(3e4)   # evaluates boundary condition
N_r = int(1e5)   # evaluates PDE residual within the domain


# optimization
w_init = "glorot_normal"
b_init = "zeros"
act = "tanh"
lr0 = 5e-3
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
lr = lrd_cos   # 1e-3 / 5e-4 / lrd_exp / lrd_cos
opt = "Adam"

# weight
w_0 = 1.
w_b = 1.
w_r = 1.

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

    return \
        in_dim, out_dim, width, depth, \
        w_init, b_init, act, \
        lr, opt, \
        w_0, w_b, w_r, \
        f_mntr, r_seed, \
        n_epch, n_btch, c_tol, \
        N_0, N_b, N_r
