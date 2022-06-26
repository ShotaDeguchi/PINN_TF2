"""
********************************************************************************
all your params
********************************************************************************
"""

import tensorflow as tf

# network structure
f_in  = 3
f_out = 1
width = 2 ** 8   # 2 ** 6 = 64, 2 ** 8 = 256
depth = 5

# training setting
n_epch = int(5e4)
n_btch = 0
c_tol  = 1e-8

# initializers
w_init = "Glorot"
b_init = "zeros"
act = "tanh"

# optimization
lr0 = 5e-3
gam = 1e-2
lrd_exp = tf.keras.optimizers.schedules.ExponentialDecay(
    initial_learning_rate = lr0, decay_steps = n_epch, decay_rate = gam, staircase=False)
lrd_cos = tf.keras.optimizers.schedules.CosineDecay(
    initial_learning_rate = lr0, decay_steps = n_epch, alpha = gam)
lr  = 5e-4   # constant (1e-3 / 5e-4 / etc.) / lrd_exp / lrd_cos
opt = "Adam"
f_scl = "minmax"   # "minmax" / "mean"
laaf = False

# system params
c = 1.

# weights
w_ini = 1.
w_bnd = 1.
w_pde = 1.

# boundary condition 
BC = "Neu"   # "Dir" for Dirichlet, "Neu" for Neumann

# rarely changed params
f_mntr = 10
r_seed = 1234

def params():
    return \
                f_in, f_out, width, depth, \
                w_init, b_init, act, \
                lr, opt, \
                f_scl, laaf, c, \
                w_ini, w_bnd, w_pde, BC, \
                f_mntr, r_seed, \
                n_epch, n_btch, c_tol, \

