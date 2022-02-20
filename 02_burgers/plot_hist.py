"""
********************************************************************************
plot loss history
********************************************************************************
"""

import matplotlib.pyplot as plt

def plot_loss(ep_log, loss_log):
    plt.figure(figsize = (8, 4))
    plt.plot(ep_log, loss_log, alpha = .7, label = "loss")
    plt.xlabel("epoch")
    plt.ylabel("loss")
    plt.yscale("log")
    # plt.ylim(1e-5, 1e0)
    plt.grid(alpha = .5)
    plt.legend(loc = "upper right")
    plt.show()

def plot_inv(ep_log, lambda_, lambda_log, name):
    plt.figure(figsize = (8, 4))
    plt.plot(ep_log, lambda_log, alpha = .7, label = name)
    plt.xlabel("epoch")
    plt.ylabel("lambda")
    plt.ylim(.8 * lambda_, 1.2 * lambda_)
    plt.grid(alpha = .5)
    plt.legend(loc = "upper right")
    plt.show()