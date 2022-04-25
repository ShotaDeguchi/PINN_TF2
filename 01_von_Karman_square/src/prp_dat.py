"""
********************************************************************************
prepares data for training, validation, and test
********************************************************************************
"""

import numpy as np
import tensorflow as tf
import pandas as pd

N_trn = int(1e4)
N_val = int(5e3)
N_inf = int(1e3)

def prp_dat(N_trn, N_val, N_inf):

    T = 250
    T = 30
    path = "../input/"
    name = "data_"
    tail = ".csv"

    t = np.array([]); x = np.array([]); y = np.array([])
    u = np.array([]); v = np.array([]); w = np.array([]); p = np.array([])

    for k in range(T):
        if k == 0:
            data = pd.read_csv(path + name + str(k) + tail, header = 0).values
            t = np.append(t, data[:, 0]); x = np.append(x, data[:, 5]); y = np.append(y, data[:, 6])
            u = np.append(u, data[:, 8]); v = np.append(v, data[:, 9]); w = np.append(w, data[:, 4]); p = np.append(p, data[:, 1])
            
        else:
            data = pd.read_csv(path + name + str(k) + tail, header = 0).values
            t = np.c_[t, data[:, 0]]; x = np.c_[x, data[:, 5]]; y = np.c_[y, data[:, 6]]
            u = np.c_[u, data[:, 8]]; v = np.c_[v, data[:, 9]]; w = np.c_[w, data[:, 4]]; p = np.c_[p, data[:, 1]]
            if k % 10 == 0:
                print("loading data at", k)

    t = t / 10.   # convert to real time

    XX = x[:,0:200]; YY = y[:,0:200]; TT = t[:,0:200]
    UU = u[:,0:200]; VV = v[:,0:200]; WW = w[:,0:200]; PP = p[:,0:200]

    x_flt = XX.flatten()[:,None]; y_flt = YY.flatten()[:,None]; t_flt = TT.flatten()[:,None]
    u_flt = UU.flatten()[:,None]; v_flt = VV.flatten()[:,None]; w_flt = WW.flatten()[:,None]; p_flt = PP.flatten()[:,None]

    X_star = np.c_[XX[:,0], YY[:,0]]
    t_star = t[0, :].reshape(-1, 1)

    N = XX.shape[0]
    T = XX.shape[1]

    N_all = N_trn + N_val + N_inf

    idx_all = np.random.choice(int(N * T), int(N_all), replace = False)
    idx_trn = idx_all[0 : int(N_trn)]
    idx_val = idx_all[int(N_trn) : int(N_trn + N_val)]
    idx_inf = idx_all[int(N_trn + N_val) : int(N_all)]

    x_trn = x_flt[idx_trn,:]; y_trn = y_flt[idx_trn,:]; t_trn = t_flt[idx_trn,:]
    u_trn = u_flt[idx_trn,:]; v_trn = v_flt[idx_trn,:]; p_trn = p_flt[idx_trn,:]
    x_val = x_flt[idx_val,:]; y_val = y_flt[idx_val,:]; t_val = t_flt[idx_val,:]
    u_val = u_flt[idx_val,:]; v_val = v_flt[idx_val,:]; p_val = p_flt[idx_val,:]
    x_inf = x_flt[idx_inf,:]; y_inf = y_flt[idx_inf,:]; t_inf = t_flt[idx_inf,:]
    u_inf = u_flt[idx_inf,:]; v_inf = v_flt[idx_inf,:]; p_inf = p_flt[idx_inf,:]

    return x_trn, y_trn, t_trn, u_trn, v_trn, p_trn, \
           x_val, y_val, t_val, u_val, v_val, p_val, \
           x_inf, y_inf, t_inf, u_inf, v_inf, p_inf

def nth_dat(xmin, xmax, 
            ymin, ymax, 
            tmin, tmax, 
            N_nth):
    x = tf.random.uniform((N_nth, 1), xmin, xmax, dtype = tf.float32)
    y = ymax * tf.ones((N_nth, 1), dtype = tf.float32)
    t = tf.random.uniform((N_nth, 1), tmin, tmax, dtype = tf.float32)
    u = tf.zeros((N_nth, 1), dtype = tf.float32)
    v = tf.zeros((N_nth, 1), dtype = tf.float32)
    p = tf.zeros((N_nth, 1), dtype = tf.float32)
    return x, y, t, u, v, p

def sth_dat(xmin, xmax, 
            ymin, ymax, 
            tmin, tmax, 
            N_sth):
    x = tf.random.uniform((N_sth, 1), xmin, xmax, dtype = tf.float32)
    y = ymin * tf.ones((N_sth, 1), dtype = tf.float32)
    t = tf.random.uniform((N_sth, 1), tmin, tmax, dtype = tf.float32)
    u = tf.zeros((N_sth, 1), dtype = tf.float32)
    v = tf.zeros((N_sth, 1), dtype = tf.float32)
    p = tf.zeros((N_sth, 1), dtype = tf.float32)
    return x, y, t, u, v, p

def est_dat(xmin, xmax, 
            ymin, ymax, 
            tmin, tmax, 
            N_est):
    x = xmax * tf.ones((N_est, 1), dtype = tf.float32)
    y = tf.random.uniform((N_est, 1), ymin, ymax, dtype = tf.float32)
    t = tf.random.uniform((N_est, 1), tmin, tmax, dtype = tf.float32)
    u = tf.zeros((N_est, 1), dtype = tf.float32)
    v = tf.zeros((N_est, 1), dtype = tf.float32)
    p = tf.zeros((N_est, 1), dtype = tf.float32)
    return x, y, t, u, v, p

def wst_dat(xmin, xmax, 
            ymin, ymax, 
            tmin, tmax, 
            N_wst):
    x = xmin * tf.ones((N_wst, 1), dtype = tf.float32)
    y = tf.random.uniform((N_wst, 1), ymin, ymax, dtype = tf.float32)
    t = tf.random.uniform((N_wst, 1), tmin, tmax, dtype = tf.float32)
    u = tf.ones((N_wst, 1), dtype = tf.float32)
    v = tf.zeros((N_wst, 1), dtype = tf.float32)
    p = tf.zeros((N_wst, 1), dtype = tf.float32)
    return x, y, t, u, v, p

def pde_dat(xmin, xmax, 
            ymin, ymax, 
            tmin, tmax, 
            N_pde):
    x = tf.random.uniform((N_pde, 1), xmin, xmax, dtype = tf.float32)
    y = tf.random.uniform((N_pde, 1), ymin, ymax, dtype = tf.float32)
    t = tf.random.uniform((N_pde, 1), tmin, tmax, dtype = tf.float32)
    return x, y, t


