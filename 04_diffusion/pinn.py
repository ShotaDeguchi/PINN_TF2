"""
********************************************************************************
Author: Shota DEGUCHI
        Yosuke SHIBATA
        Structural Analysis Laboratory, Kyushu University (Jul. 19th, 2021)
implementation of PINN - Physics-Informed Neural Network on TensorFlow 2
********************************************************************************
"""

import os
import time
import datetime
import tensorflow as tf
import numpy as np

class PINN(tf.keras.Model):
    def __init__(self, 
                 t_0, x_0, u_0, 
                 t_b, x_b, u_b, 
                 t_r, x_r, 
                 Rm, Rn, Rl, depth, activ = "tanh", BN = False, 
                 w_init = "glorot_normal", b_init = "zeros", 
                 lr = 1e-3, opt = "Adam", w_0 = 1., w_b = 1., w_r = 1.,
                 f_mntr = 10, r_seed = 1234):
        
        # initialize the configuration
        super().__init__()
        self.Rm     = Rm       # input dimension
        self.Rn     = Rn       # output dimension
        self.Rl     = Rl       # internal dimension
        self.depth  = depth    # (# of hidden layers) + output layer
        self.activ  = activ    # activation function
        self.BN     = BN       # BatchNorm? 
        self.w_init = w_init   # initial weight
        self.b_init = b_init   # initial bias
        self.lr     = lr       # learning rate
        self.opt    = opt      # name of your optimizer ("SGD", "RMSprop", "Adam", etc.)
        self.w_0    = w_0      # weight for initial loss
        self.w_b    = w_b      # weight for boundary loss
        self.w_r    = w_r      # weight for residual loss
        self.r_seed = r_seed
        self.f_mntr = f_mntr
        self.data_type = tf.float32
        self.random_seed(seed = self.r_seed)
        
        # input-output pair
        self.t_0 = t_0; self.x_0 = x_0; self.u_0 = u_0   # evaluates initial condition
        self.t_b = t_b; self.x_b = x_b; self.u_b = u_b   # evaluates boundary condition
        self.t_r = t_r; self.x_r = x_r                   # evaluates domain residual
        
        # bounds
        X_r     = tf.concat([t_r, x_r], 1)
        self.lb = tf.cast(tf.reduce_min(X_r, axis = 0), self.data_type)
        self.ub = tf.cast(tf.reduce_max(X_r, axis = 0), self.data_type)
        
        # call
        self.dnn = self.dnn_init(Rm, Rn, Rl, depth)
        self.params = self.dnn.trainable_variables
        self.optimizer = self.opt_alg(self.lr, self.opt)
        
        # parameter setting
        self.D = tf.constant(.01, dtype = self.data_type)

        # track loss
        self.ep_log = []
        self.loss_log = []
        
        print("\n************************************************************")
        print("****************     MAIN PROGRAM START     ****************")
        print("************************************************************")
        print(">>>>> start time:", datetime.datetime.now())
        print(">>>>> configuration;")
        print("         dtype        :", self.data_type)
        print("         activ func   :", self.activ)
        print("         weight init  :", self.w_init)
        print("         learning rate:", self.lr)
        print("         optimizer    :", self.opt)
        print("         summary      :", self.dnn.summary())
        
    def random_seed(self, seed = 1234):
        os.environ["PYTHONHASHSEED"] = str(seed)
        np.random.seed(seed)
        tf.random.set_seed(seed)
        
    def dnn_init(self, Rm, Rn, Rl, depth):
        # network configuration (N: Rm -> Rn (Rm -> Rl -> ... -> Rl -> Rn))
        network = tf.keras.Sequential()
        network.add(tf.keras.layers.InputLayer(Rm))
        network.add(tf.keras.layers.Lambda(lambda x: 2. * (x - self.lb) / (self.ub - self.lb) - 1.))
        
        if self.BN == True:   # False by default
            for l in range(depth - 1):
                network.add(tf.keras.layers.Dense(Rl, activation = self.activ, use_bias = False,
                                                  kernel_initializer = self.w_init, bias_initializer = self.b_init, 
                                                  kernel_regularizer = None, bias_regularizer = None, 
                                                  activity_regularizer = None, kernel_constraint = None, bias_constraint = None))
                network.add(tf.keras.layers.BatchNormalization(axis = -1, momentum = 0.99, epsilon = 0.001, 
                                                               center = True, scale = True,
                                                               beta_initializer = "zeros", gamma_initializer = "ones",
                                                               moving_mean_initializer = "zeros",
                                                               moving_variance_initializer = "ones", 
                                                               beta_regularizer = None, gamma_regularizer = None, 
                                                               beta_constraint  = None, gamma_constraint  = None))
            
        else:   # False by default
            for l in range(depth - 1):
                network.add(tf.keras.layers.Dense(Rl, activation = self.activ, use_bias = True,
                                                  kernel_initializer = self.w_init, bias_initializer = self.b_init, 
                                                  kernel_regularizer = None, bias_regularizer = None, 
                                                  activity_regularizer = None, kernel_constraint = None, bias_constraint = None))
        network.add(tf.keras.layers.Dense(Rn))
        return network
    
    def opt_alg(self, lr, opt):
        if   opt == "SGD":
            optimizer = tf.keras.optimizers.SGD(learning_rate = lr, momentum = 0.0, nesterov = False)
        elif opt == "RMSprop":
            optimizer = tf.keras.optimizers.RMSprop(learning_rate = lr, rho = 0.9, momentum = 0.0, centered = False)
        elif opt == "Adam":
            optimizer = tf.keras.optimizers.Adam(learning_rate = lr, beta_1 = 0.9, beta_2 = 0.999, amsgrad = False)
        elif opt == "Adamax":
            optimizer = tf.keras.optimizers.Adamax(learning_rate = lr, beta_1 = 0.9, beta_2 = 0.999)
        elif opt == "Nadam":
            optimizer = tf.keras.optimizers.Nadam(learning_rate = lr, beta_1 = 0.9, beta_2 = 0.999)
        else:
            raise Exception(">>>>> Exception: optimizer not specified correctly")
            
        return optimizer
    
    def PDE(self, t, x):
        t = tf.convert_to_tensor(t, dtype = self.data_type)
        x = tf.convert_to_tensor(x, dtype = self.data_type)

        with tf.GradientTape(persistent = True) as tp:
            tp.watch(t)
            tp.watch(x)
            
            u = self.dnn(tf.concat([t, x], 1))
            u_x = tp.gradient(u, x)
        u_t  = tp.gradient(u, t)
        u_xx = tp.gradient(u_x, x)
        del tp
        
        gv = u_t - self.D * u_xx
        
        return u, gv
    
    def loss_prd(self, t, x, u):
        u_hat, dummy = self.PDE(t, x)
        loss_prd = tf.reduce_mean(tf.square(u - u_hat))
        return loss_prd
        
    def loss_pde(self, t, x):
        dummy, gv_hat = self.PDE(t, x)
        loss_pde = tf.reduce_mean(tf.square(gv_hat))
        return loss_pde
    
    @tf.function
    def loss_glb(self, 
                 t_0, x_0, u_0, 
                 t_b, x_b, u_b, 
                 t_r, x_r):
        loss_0   = self.w_0 * self.loss_prd(t_0, x_0, u_0)
        loss_b   = self.w_b * self.loss_prd(t_b, x_b, u_b)
        loss_r   = self.w_r * self.loss_pde(t_r, x_r)
        loss_glb = loss_0 + loss_b + loss_r
        return loss_glb

    def loss_grad(self, 
                  t_0, x_0, u_0, 
                  t_b, x_b, u_b, 
                  t_r, x_r): 
        with tf.GradientTape(persistent = True) as tp:
            loss = self.loss_glb(t_0, x_0, u_0, 
                                 t_b, x_b, u_b, 
                                 t_r, x_r)
        grad = tp.gradient(loss, self.params)
        del tp
        return loss, grad
    
    @tf.function
    def grad_desc(self, 
                  t_0, x_0, u_0, 
                  t_b, x_b, u_b, 
                  t_r, x_r):
        loss, grad = self.loss_grad(t_0, x_0, u_0, 
                                    t_b, x_b, u_b, 
                                    t_r, x_r)
        self.optimizer.apply_gradients(zip(grad, self.params))
        return loss
        
    def train(self, epoch = 10 ** 5, batch = 2 ** 6, tol = 1e-5): 
        print(">>>>> training setting;")
        print("         # of epoch     :", epoch)
        print("         batch size     :", batch)
        print("         convergence tol:", tol)
        
        t0 = time.time()
        
        # I had to convert input data (tf.tensor) into numpy style in order for mini-batch training (slicing)
        # and this works well for both full-batch and mini-batch training
        t_0 = self.t_0.numpy(); x_0 = self.x_0.numpy(); u_0 = self.u_0.numpy()
        t_b = self.t_b.numpy(); x_b = self.x_b.numpy(); u_b = self.u_b.numpy()
        t_r = self.t_r.numpy(); x_r = self.x_r.numpy()
        
        for ep in range(epoch):
            ep_loss = 0
            
            if batch == 0:   # full-batch training
                ep_loss = self.grad_desc(t_0, x_0, u_0, 
                                         t_b, x_b, u_b, 
                                         t_r, x_r)
            
            else:   # mini-batch training
                n_0 = self.x_0.shape[0]; idx_0 = np.random.permutation(n_0)
                n_b = self.x_b.shape[0]; idx_b = np.random.permutation(n_b)
                n_r = self.x_r.shape[0]; idx_r = np.random.permutation(n_r)

                n_data = self.x_0.shape[0] + self.x_b.shape[0] + self.x_r.shape[0]
                shf_idx = np.random.permutation(n_data)

                for idx in range(0, n_r, batch):
                    # batch for initial condition
#                     t_0_btch = tf.convert_to_tensor(t_0[idx_0[idx: idx + batch if idx + batch < n_0 else n_0]], dtype = self.data_type)
#                     x_0_btch = tf.convert_to_tensor(x_0[idx_0[idx: idx + batch if idx + batch < n_0 else n_0]], dtype = self.data_type)
#                     u_0_btch = tf.convert_to_tensor(u_0[idx_0[idx: idx + batch if idx + batch < n_0 else n_0]], dtype = self.data_type)
                    t_0_btch = t_0
                    x_0_btch = x_0
                    u_0_btch = u_0
                    # batch for boudary condition
#                     t_b_btch = tf.convert_to_tensor(t_b[idx_b[idx: idx + batch if idx + batch < n_b else n_b]], dtype = self.data_type)
#                     x_b_btch = tf.convert_to_tensor(x_b[idx_b[idx: idx + batch if idx + batch < n_b else n_b]], dtype = self.data_type)
#                     u_b_btch = tf.convert_to_tensor(u_b[idx_b[idx: idx + batch if idx + batch < n_b else n_b]], dtype = self.data_type)
                    t_b_btch = t_b
                    x_b_btch = x_b
                    u_b_btch = u_b
                    # batch for domain residual
                    t_r_btch = tf.convert_to_tensor(t_r[idx_r[idx: idx + batch if idx + batch < n_r else n_r]], dtype = self.data_type)
                    x_r_btch = tf.convert_to_tensor(x_r[idx_r[idx: idx + batch if idx + batch < n_r else n_r]], dtype = self.data_type)
                    # compute loss and perform gradient descent
                    loss_btch = self.grad_desc(t_0_btch, x_0_btch, u_0_btch, 
                                               t_b_btch, x_b_btch, u_b_btch, 
                                               t_r_btch, x_r_btch)
                    ep_loss += loss_btch / int(n_r / batch)
                
            if ep % self.f_mntr == 0:
                elps = time.time() - t0
                
                self.ep_log.append(ep)
                self.loss_log.append(ep_loss)
                print("ep: %d, loss: %.6e, elps: %.3f" % (ep, ep_loss, elps))
                t0 = time.time()
            
            if ep_loss < tol:
                print(">>>>> program terminating with the loss converging to its tolerance.")
                print("\n************************************************************")
                print("*****************     MAIN PROGRAM END     *****************")
                print("************************************************************")
                print(">>>>> end time:", datetime.datetime.now())
                break
        
        print("\n************************************************************")
        print("*****************     MAIN PROGRAM END     *****************")
        print("************************************************************")
        print(">>>>> end time:", datetime.datetime.now())
                
    def infer(self, t, x):
        u_hat, gv_hat = self.PDE(t, x)
        return u_hat, gv_hat
