"""
********************************************************************************
Author: Shota DEGUCHI
        Yosuke SHIBATA
        Structural Analysis Lab. Kyushu Univ. (Feb. 3, 2022)

implementation of PINN - Physics-Informed Neural Network on TensorFlow 2

Reference;
[1] M. Raissi, P. Perdikaris, G.E. Karniadakis, 
    Physics-informed neural networks: A deep learning framework for solving forward and inverse problems involving nonlinear partial differential equations,
    Journal of Computational Physics, 2019.
[2] Ameya D. Jagtap, Kenji Kawaguchi, George Em Karniadakis,
    Adaptive activation functions accelerate convergence in deep and physics-informed neural networks,
    Journal of Computational Physics, 2020. 
[3] Jagtap, Ameya D. and Kawaguchi, Kenji and Em Karniadakis, George,
    Locally adaptive activation functions with slope recovery for deep and physics-informed neural networks,
    Proceedings of the Royal Society A: Mathematical, Physical and Engineering Sciences, 2020.
********************************************************************************
"""

import os
import time
import datetime
import tensorflow as tf
import numpy as np

class PINN:
    def __init__(self, 
                 x_nth, y_nth, t_nth, u_nth, v_nth, p_nth, 
                 x_sth, y_sth, t_sth, u_sth, v_sth, p_sth, 
                 x_est, y_est, t_est, u_est, v_est, p_est, 
                 x_wst, y_wst, t_wst, u_wst, v_wst, p_wst, 
                 x_pde, y_pde, t_pde, 
                 in_dim, out_dim, width, depth, 
                 w_init = "Glorot", b_init = "zeros", act = "tanh", 
                 lr = 1e-3, opt = "Adam", 
                 f_scl = "minmax", laaf = False, inv = False, 
                 rho = .1, nu = .1, 
                 w_nth = 1., w_sth = 1., w_est = 1., w_wst = 1., w_pde = 1., 
                 f_mntr = 10, r_seed = 1234):
        
        # initialize the configuration
        self.r_seed = r_seed
        self.random_seed(seed = self.r_seed)
        self.dat_typ = tf.float32
        self.in_dim  = in_dim    # input dimension
        self.out_dim = out_dim   # output dimension
        self.width   = width     # middle dimension
        self.depth   = depth     # (# of hidden layers) + output layer
        self.w_init  = w_init    # weight initializer
        self.b_init  = b_init    # bias initializer
        self.act     = act       # activation function
        self.lr      = lr        # learning rate
        self.opt     = opt       # optimizer (SGD, RMSprop, Adam, etc.)
        self.f_scl   = f_scl     # feature scaling
        self.laaf    = laaf      # LAAF? (L-LAAF, GAAF / N-LAAF not implemented)
        self.inv     = inv       # inverse problem? 
        self.f_mntr  = f_mntr    # monitoring frequency
        self.w_nth   = w_nth
        self.w_sth   = w_sth
        self.w_est   = w_est
        self.w_wst   = w_wst
        self.w_pde   = w_pde
        
        # boundary & PDE
        self.x_nth = x_nth; self.y_nth = y_nth; self.t_nth = t_nth; self.u_nth = u_nth; self.v_nth = v_nth; self.p_nth = p_nth
        self.x_sth = x_sth; self.y_sth = y_sth; self.t_sth = t_sth; self.u_sth = u_sth; self.v_sth = v_sth; self.p_sth = p_sth
        self.x_est = x_est; self.y_est = y_est; self.t_est = t_est; self.u_est = u_est; self.v_est = v_est; self.p_est = p_est
        self.x_wst = x_wst; self.y_wst = y_wst; self.t_wst = t_wst; self.u_wst = u_wst; self.v_wst = v_wst; self.p_wst = p_wst
        self.x_pde = x_pde; self.y_pde = y_pde; self.t_pde = t_pde
        
        # bounds (for feature scaling)
        bounds  = tf.concat([x_pde, y_pde, t_pde], 1)
        self.lb = tf.cast(tf.reduce_min (bounds, axis = 0), self.dat_typ)
        self.ub = tf.cast(tf.reduce_max (bounds, axis = 0), self.dat_typ)
        self.mn = tf.cast(tf.reduce_mean(bounds, axis = 0), self.dat_typ)

        # build a network
        self.structure = [self.in_dim] + [self.width] * (self.depth - 1) + [self.out_dim]
        self.weights, self.biases, self.alphas, self.params = self.dnn_init(self.structure)
        
        # system params
        self.rho = rho
        self.nu  = nu
        if self.inv == True:
            self.rho = tf.Variable(self.rho, dtype = self.dat_typ)
            self.nu  = tf.Variable(self.nu , dtype = self.dat_typ)
            self.params.append(self.rho)
            self.params.append(self.nu)
            self.rho_log = []
            self.nu_log  = []
        elif self.inv == False:
            self.rho = tf.constant(self.rho, dtype = self.dat_typ)
            self.nu  = tf.constant(self.nu , dtype = self.dat_typ)
        else:
            raise NotImplementedError(">>>>> system params")

        # optimization
        self.optimizer    = self.opt_(self.lr, self.opt)
        self.ep_log       = []
        self.loss_glb_log = []
        self.loss_bnd_log = []
        self.loss_pde_log = []
        
        print("\n************************************************************")
        print("****************     MAIN PROGRAM START     ****************")
        print("************************************************************")
        print(">>>>> start time:", datetime.datetime.now())
        print(">>>>> configuration;")
        print("         random seed  :", self.r_seed)
        print("         data type    :", self.dat_typ)
        print("         activation   :", self.act)
        print("         weight init  :", self.w_init)
        print("         bias   init  :", self.b_init)
        print("         learning rate:", self.lr)
        print("         optimizer    :", self.opt)
        print("         width        :", self.width)
        print("         depth        :", self.depth)
        print("         structure    :", self.structure)
        
    def random_seed(self, seed = 1234):
        os.environ["PYTHONHASHSEED"] = str(seed)
        np.random.seed(seed)
        tf.random.set_seed(seed)
        
    def dnn_init(self, strc):
        weights = []
        biases  = []
        alphas  = []
        params  = []
        for d in range(0, self.depth):   # depth = self.depth
            w = self.weight_init(shape = [strc[d], strc[d + 1]], depth = d)
            b = self.bias_init  (shape = [      1, strc[d + 1]], depth = d)
            weights.append(w)
            biases .append(b)
            params .append(w)
            params .append(b)
            if self.laaf == True and d < self.depth - 1:
                a = tf.Variable(1., dtype = self.dat_typ, name = "a" + str(d))
                params.append(a)
            else:
                a = tf.constant(1., dtype = self.dat_typ)
            alphas .append(a)
        return weights, biases, alphas, params
        
    def weight_init(self, shape, depth):
        in_dim  = shape[0]
        out_dim = shape[1]
        if self.w_init == "Glorot":
            std = np.sqrt(2 / (in_dim + out_dim))
        elif self.w_init == "He":
            std = np.sqrt(2 / in_dim)
        elif self.w_init == "LeCun":
            std = np.sqrt(1 / in_dim)
        else:
            raise NotImplementedError(">>>>> weight_init")
        weight = tf.Variable(
            tf.random.truncated_normal(shape = [in_dim, out_dim], \
            mean = 0., stddev = std, dtype = self.dat_typ), \
            dtype = self.dat_typ, name = "w" + str(depth)
            )
        return weight
    
    def bias_init(self, shape, depth):
        in_dim  = shape[0]
        out_dim = shape[1]
        if self.b_init == "zeros":
            bias = tf.Variable(
                tf.zeros(shape = [in_dim, out_dim], dtype = self.dat_typ), \
                dtype = self.dat_typ, name = "b" + str(depth)
                )
        elif self.b_init == "ones":
            bias = tf.Variable(
                tf.ones(shape = [in_dim, out_dim], dtype = self.dat_typ), \
                dtype = self.dat_typ, name = "b" + str(depth)
                )
        else:
            raise NotImplementedError(">>>>> bias_init")
        return bias
    
    def opt_(self, lr, opt):
        if opt == "SGD":
            optimizer = tf.keras.optimizers.SGD(
                learning_rate = lr, momentum = 0.0, nesterov = False
                )
        elif opt == "RMSprop":
            optimizer = tf.keras.optimizers.RMSprop(
                learning_rate = lr, rho = 0.9, momentum = 0.0, centered = False
                )
        elif opt == "Adam":
            optimizer = tf.keras.optimizers.Adam(
                learning_rate = lr, beta_1 = 0.9, beta_2 = 0.999, amsgrad = False
                )
        elif opt == "Adamax":
            optimizer = tf.keras.optimizers.Adamax(
                learning_rate = lr, beta_1 = 0.9, beta_2 = 0.999
                )
        elif opt == "Nadam":
            optimizer = tf.keras.optimizers.Nadam(
                learning_rate = lr, beta_1 = 0.9, beta_2 = 0.999
                )
        else:
            raise NotImplementedError(">>>>> opt_")
        return optimizer
    
    def forward_pass(self, x):
        # feature scaling
        if self.f_scl == "minmax":
            z = 2. * (x - self.lb) / (self.ub - self.lb) - 1.
        elif self.f_scl == "mean":
            z = (x - self.mn) / (self.ub - self.lb)
        else:
            raise NotImplementedError(">>>>> forward_pass (f_scl)")
        # forward pass
        for d in range(0, self.depth - 1):
            w = self.weights[d]
            b = self.biases [d]
            a = self.alphas [d]
            u = tf.add(tf.matmul(z, w), b)
            u = tf.multiply(a, u)
            if self.act == "tanh":
                z = tf.tanh(u)
            elif self.act == "swish":
                z = tf.multiply(u, tf.sigmoid(u))
            elif self.act == "gelu":
                z = tf.multiply(u, tf.sigmoid(1.702 * u))
            elif self.act == "mish":
                z = tf.multiply(u, tf.tanh(tf.nn.softplus(u)))
            else:
                raise NotImplementedError(">>>>> forward_pass (act)")
        w = self.weights[-1]
        b = self.biases [-1]
        a = self.alphas [-1]
        u = tf.add(tf.matmul(z, w), b)
        u = tf.multiply(a, u)
        z = u   # identity mapping
        y = z
        return y
        
    def pde(self, x, y, t):
        x = tf.convert_to_tensor(x, dtype = self.dat_typ)
        y = tf.convert_to_tensor(y, dtype = self.dat_typ)
        t = tf.convert_to_tensor(t, dtype = self.dat_typ)
        with tf.GradientTape(persistent = True) as tp:
            tp.watch(x)
            tp.watch(y)
            tp.watch(t)
            u_v_p = self.forward_pass(tf.concat([x, y, t], 1))
            u = u_v_p[:,0:1]
            v = u_v_p[:,1:2]
            p = u_v_p[:,2:3]            
            u_x = tp.gradient(u, x); u_y = tp.gradient(u, y)
            v_x = tp.gradient(v, x); v_y = tp.gradient(v, y)
        u_t  = tp.gradient(u, t);   v_t  = tp.gradient(v, t)
        u_xx = tp.gradient(u_x, x); u_yy = tp.gradient(u_y, y)
        v_xx = tp.gradient(v_x, x); v_yy = tp.gradient(v_y, y)
        p_x  = tp.gradient(p, x);   p_y  = tp.gradient(p, y)
        del tp
        gv_c = u_x + v_y                                                            # continuity
        gv_x = u_t + u * u_x + v * u_y + p_x / self.rho - self.nu * (u_xx + u_yy)   # momentum
        gv_y = v_t + u * v_x + v * v_y + p_y / self.rho - self.nu * (v_xx + v_yy)
        return u, v, p, gv_c, gv_x, gv_y

    # def loss_dat(self, x, y, t, u, v, p):
    #     u_, v_, p_, _, _, _ = self.pde(x, y, t)
    #     loss =   tf.reduce_mean(tf.square(u - u_)) \
    #            + tf.reduce_mean(tf.square(v - v_)) \
    #            + tf.reduce_mean(tf.square(p - p_))
    #     return loss

    def loss_nth(self, x, y, t):
        with tf.GradientTape(persistent = True) as tp:
            tp.watch(x)
            tp.watch(y)
            tp.watch(t)
            u_, v_, p_, _, _, _ = self.pde(x, y, t)
        p_y = tp.gradient(p_, y)
        del tp
        loss = tf.reduce_mean(tf.square(p_y))
        return loss

    def loss_sth(self, x, y, t):
        with tf.GradientTape(persistent = True) as tp:
            tp.watch(x)
            tp.watch(y)
            tp.watch(t)
            u_, v_, p_, _, _, _ = self.pde(x, y, t)
        p_y = - tp.gradient(p_, y)
        del tp
        loss = tf.reduce_mean(tf.square(p_y))
        return loss

    def loss_est(self, x, y, t, u, v, p):
        _, _, p_, _, _, _ = self.pde(x, y, t)
        loss = tf.reduce_mean(tf.square(p - p_))
        return loss

    def loss_wst(self, x, y, t, u, v, p):
        u_, _, _, _, _, _ = self.pde(x, y, t)
        loss = tf.reduce_mean(tf.square(u - u_))
        return loss
        
    def loss_pde(self, x, y, t):
        _, _, _, gv_c_, gv_x_, gv_y_ = self.pde(x, y, t)
        loss =   tf.reduce_mean(tf.square(gv_c_)) \
               + tf.reduce_mean(tf.square(gv_x_)) \
               + tf.reduce_mean(tf.square(gv_y_))
        return loss
    
    @tf.function
    def loss_glb(self, 
                 x_nth, y_nth, t_nth, 
                 x_sth, y_sth, t_sth, 
                 x_est, y_est, t_est, u_est, v_est, p_est, 
                 x_wst, y_wst, t_wst, u_wst, v_wst, p_wst, 
                 x_pde, y_pde, t_pde):
        loss =   self.w_nth * self.loss_nth(x_nth, y_nth, t_nth) \
               + self.w_sth * self.loss_sth(x_sth, y_sth, t_sth) \
               + self.w_est * self.loss_est(x_est, y_est, t_est, u_est, v_est, p_est) \
               + self.w_wst * self.loss_wst(x_wst, y_wst, t_wst, u_wst, v_wst, p_wst) \
               + self.w_pde * self.loss_pde(x_pde, y_pde, t_pde)
        if self.laaf == True:
            loss += 1. / tf.reduce_mean(tf.exp(self.alphas))
        else:
            pass
        return loss

    def loss_grad(self, 
                  x_nth, y_nth, t_nth, 
                  x_sth, y_sth, t_sth, 
                  x_est, y_est, t_est, u_est, v_est, p_est, 
                  x_wst, y_wst, t_wst, u_wst, v_wst, p_wst, 
                  x_pde, y_pde, t_pde):
        with tf.GradientTape(persistent = True) as tp:
            loss = self.loss_glb(x_nth, y_nth, t_nth, 
                                 x_sth, y_sth, t_sth, 
                                 x_est, y_est, t_est, u_est, v_est, p_est, 
                                 x_wst, y_wst, t_wst, u_wst, v_wst, p_wst, 
                                 x_pde, y_pde, t_pde)
        grad = tp.gradient(loss, self.params)
        del tp
        return loss, grad
    
    @tf.function
    def grad_desc(self, 
                  x_nth, y_nth, t_nth, 
                  x_sth, y_sth, t_sth, 
                  x_est, y_est, t_est, u_est, v_est, p_est, 
                  x_wst, y_wst, t_wst, u_wst, v_wst, p_wst, 
                  x_pde, y_pde, t_pde):
        loss, grad = self.loss_grad(x_nth, y_nth, t_nth, 
                                    x_sth, y_sth, t_sth, 
                                    x_est, y_est, t_est, u_est, v_est, p_est, 
                                    x_wst, y_wst, t_wst, u_wst, v_wst, p_wst, 
                                    x_pde, y_pde, t_pde)
        self.optimizer.apply_gradients(zip(grad, self.params))
        return loss
        
    def train(self, epoch = int(1e3), batch = 2 ** 6, tol = 1e-5): 
        print(">>>>> training setting;")
        print("         # of epoch     :", epoch)
        print("         batch size     :", batch)
        print("         convergence tol:", tol)

        # boundary & PDE
        x_nth = self.x_nth; y_nth = self.y_nth; t_nth = self.t_nth; u_nth = self.u_nth; v_nth = self.v_nth; p_nth = self.p_nth
        x_sth = self.x_sth; y_sth = self.y_sth; t_sth = self.t_sth; u_sth = self.u_sth; v_sth = self.v_sth; p_sth = self.p_sth
        x_est = self.x_est; y_est = self.y_est; t_est = self.t_est; u_est = self.u_est; v_est = self.v_est; p_est = self.p_est
        x_wst = self.x_wst; y_wst = self.y_wst; t_wst = self.t_wst; u_wst = self.u_wst; v_wst = self.v_wst; p_wst = self.p_wst
        x_pde = self.x_pde; y_pde = self.y_pde; t_pde = self.t_pde

        t0 = time.time()
        for ep in range(epoch):
            if batch == 0:
                ep_loss = self.grad_desc(
                    x_nth, y_nth, t_nth, 
                    x_sth, y_sth, t_sth, 
                    x_est, y_est, t_est, u_est, v_est, p_est, 
                    x_wst, y_wst, t_wst, u_wst, v_wst, p_wst, 
                    x_pde, y_pde, t_pde)
            else:
                n_b = x_pde.shape[0]
                idx_b = np.random.permutation(n_b)

                for idx in range(0, n_b, batch):
                    x_nth_b = x_nth[idx_b[idx:idx+batch if idx+batch<n_b else n_b]]


            if ep % self.f_mntr == 0:
                elps = time.time() - t0

                loss_nth = self.loss_nth(x_nth, y_nth, t_nth)
                loss_sth = self.loss_sth(x_sth, y_sth, t_sth)
                loss_est = self.loss_est(x_est, y_est, t_est, u_est, v_est, p_est)
                loss_wst = self.loss_wst(x_wst, y_wst, t_wst, u_wst, v_wst, p_wst)
                loss_bnd = 1 / 4 * (loss_nth + loss_sth + loss_est + loss_wst)
                loss_pde = self.loss_pde(x_pde, y_pde, t_pde)

                self.ep_log.append(ep)
                self.loss_glb_log.append(ep_loss)
                self.loss_bnd_log.append(loss_bnd)
                self.loss_pde_log.append(loss_pde)
                
                if self.inv == True:
                    ep_rho = self.rho.numpy()
                    ep_nu  = self.nu .numpy()
                    self.rho_log.append(ep_rho)
                    self.nu_log .append(ep_nu)
                    print("ep: %d, loss: %.3e, loss_bnd: %.3e, loss_pde: %.3e, rho: %.3e, nu: %.3e, elps: %.3f"
                    % (ep, ep_loss, loss_bnd, loss_pde, ep_rho, ep_nu, elps))
                elif self.inv == False:
                    print("ep: %d, loss: %.3e, loss_bnd: %.3e, loss_pde: %.3e, elps: %.3f"
                    % (ep, ep_loss, loss_bnd, loss_pde, elps))
                else:
                    raise NotImplementedError(">>>>> system params")
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
                
    def inference(self, x, y, t):
        u_, v_, p_, gv_c_, gv_x_, gv_y_ = self.pde(x, y, t)
        return u_, v_, p_, gv_c_, gv_x_, gv_y_

    # def statistics(self, x, y, t, u, v, p, u_, v_, p_):
    #     n = x.shape[0]
    #     u - u_
    #     mean = np.mean(a, dtype = np.float32)
    #     std = np.std(a, dtype = np.float32, ddof = 0)
    #     sem = np.std(a, dtype = np.float32, ddof = 1) / np.sqrt(n)
    #     return mean, std, sem
