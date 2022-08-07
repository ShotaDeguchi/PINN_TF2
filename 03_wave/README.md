# PINN(s): Physics-Informed Neural Network(s) for 2D wave equation

This is an implementation of [PINN(s)](https://doi.org/10.1016/j.jcp.2018.10.045) on TensorFlow 2 to solve 2D wave equation under Dirichlet / Neumann boundary condition without training data (data to fit initial & boundary conditions need to be provided). This is enabled by [automatic differentiation](https://arxiv.org/abs/1502.05767), which is a generalization of [back-propagation](https://doi.org/10.1038/323533a0). Within this, PINN-derived solution is compared with FDM (Finite Difference Method) approximation to show a quantitative agreement. While training could be accelerated with GPU-utilized learning, this code also implements [L-LAAF](https://doi.org/10.1098/rspa.2020.0334) for further speed-up (scaling factor is set to 1.0). 

## Solution
FDM simulation (at t = 0.5, 1.5, 5.0 (s)):
<img src="./figures/FDM.png">

PINN solution (at t = 0.5, 1.5, 5.0 (s)):
<img src="./figures/PINN.png">

With the present setting, the error between FDM and PINN is as follows:
|t|0.5|1.5|5.0|
|:---:|:---:|:---:|:---:|
|mse (&plusmn;sem) (scale: 1e-5)|0.297 (&plusmn;0.003)|0.551 (&plusmn;0.004)|4.273 (&plusmn;0.028)|

## Usage
Simply type
<br>
<code>
  python main.py
</code>
<br>
to run the entire code (this includes FDM simulation, PINN training, inferece, and comparison). Basic parameters (e.g., network architecture, batch size, initializer, etc.) are found in 
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
