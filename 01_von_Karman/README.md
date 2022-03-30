# PINN(s): Physics-Informed Neural Network(s) for 2D Navier-Stokes equation

This is an implementation of [PINN(s)](https://doi.org/10.1016/j.jcp.2018.10.045) on TensorFlow 2 for the inverse analysis of 2D Navier-Stokes equation. This code especially focuses on von Karman vortex streets induced by a square obstacle in the flow field. Evaluation of Physics-Informed loss and parameter estimate is computed via [automatic differentiation](https://arxiv.org/abs/1502.05767), which is a generalization of [back-propagation](https://doi.org/10.1038/323533a0). Here, estimated parameters are fluid density (rho) and kinematic viscosity (nu), which are set 1 [kg/m3] and 0.01 [m2/s]. Reference solution is obtained by finite volume method and data is stored in <code>input</code> directory. While training could be accelerated with GPU-utilized learning, this code also implements [L-LAAF](https://doi.org/10.1098/rspa.2020.0334) for further speed-up (scaling factor is set to 1.0). 

## Usage
Simply type
<br>
<code>
  python main.py
</code>
<br>
to run the entire code. Basic parameters (e.g., network architecture, batch size, initializer, etc.) are found in 
<br>
<code>
  params.py
</code>
<br>
and could be modified depending on the problem setup. 

## Dependencies
Tested on 
<br>
<code>
  python 3.8.10
</code>
<br>
with the following:

|Package                      |Version|
| :---: | :---: |
|numpy                        |1.22.1|
|scipy                        |1.7.3|
|tensorflow                   |2.8.0|

## Reference
[1] Raissi, M., Perdikaris, P., Karniadakis, G.E.: Physics-informed neural networks: A deep learning framework for solving forward and inverse problems involving nonlinear partial differential equations, *Journal of Computational Physics*, Vol. 378, pp. 686-707, 2019. 
<br>
[2] Baydin, A.G., Pearlmutter, B.A., Radul, A.A., Siskind, J.M.: Automatic Differentiation in Machine Learning: A Survey, *Journal of Machine Learning Research*, Vol. 18, No. 1, pp. 5595–5637, 2018. 
<br>
[3] Rumelhart, D., Hinton, G., Williams, R.: Learning representations by back-propagating errors, *Nature*, Vol. 323, pp. 533–536, 1986. 
<br>
[4] Jagtap, A.D., Kawaguchi, K., Karniadakis, GE.: Locally adaptive activation functions with slope recovery for deep and physics-informed neural networks, *Proceedings of Royal Society A*, pp. 4762020033420200334, 2020. 
