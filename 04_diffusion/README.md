# PINN(s): Physics-Informed Neural Network(s) for diffusion equation

This is an implementation of [PINN(s)](https://doi.org/10.1016/j.jcp.2018.10.045) on TensorFlow 2 to solve 1D linear diffusion equation under Dirichlet boundary condition without training data (data to fit initial & boundary conditions need to be provided). This is enabled by [automatic differentiation](https://arxiv.org/abs/1502.05767), which is a generalization of [back-propagation](https://doi.org/10.1038/323533a0). Within this code, PINN-derived solution is compared with FDM (Finite Difference Method) approximation to show a quantitative agreement. While [original work](https://github.com/maziarraissi/PINNs) is bulit on TensorFlow 1, this repository implements on TensorFlow 2 for GPU-utilized acceleration. 

## Solution
The following is the solution plot of the derived solution. Initial condition is negative sine curve, boundary condition is zero (Dirichlet). 
<img src="./figures/Burgers.png">

## Usage
Simply type
<br>
<code>
  python main.py
</code>
<br>
to run code (this includes PINN training, inferece, and visualization). Basic parameters (e.g., network architecture, batch size, initializer, etc.) are found in 
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
