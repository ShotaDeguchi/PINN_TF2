# PINN(s): Physics-Informed Neural Network(s) on TensorFlow 2

This repository stores [PINN(s)](https://doi.org/10.1016/j.jcp.2018.10.045) implementation on TensorFlow 2 to solve von Karman vortex streets (inverse problem, <code>01_von_Karman</code>), Burgers equation (forward problem, <code>02_Burgers</code>), 2D wave equation (forward problem, <code>03_wave</code>), 1D diffusion equation (forward problem, <code>04_diffusion</code>). [Automatic differentiation](https://arxiv.org/abs/1502.05767), which is a generalization of [back-propagation](https://doi.org/10.1038/323533a0), is utilized to leverage the convenctional neural network architecture's representation power and to satisfy govering equations, initial, and boundary conditions. This enables PINN(s) to learn the dynamics from scarce dataset / solve initial and boundary value problems, which is very challenging for standard neural network architectures. While [original work](https://github.com/maziarraissi/PINNs) bulids PINNs on TensorFlow 1, this repository's codes implement on TensorFlow 2 for GPU-based acceleration + [L-LAAF](https://doi.org/10.1098/rspa.2020.0334). 

Further descriptions (usage, option, etc.) can be found in the corresponding directories. 

## Examples
|1D Burgers (FDM vs. PINN)|1D diffusion (FDM vs. PINN)|2D wave (PINN)|
|:---:|:---:|:---:|
|![burgers](https://user-images.githubusercontent.com/49257696/162746099-bd030010-c819-4bba-87e9-cd1c26a59913.gif)|![diffusion](https://user-images.githubusercontent.com/49257696/162752724-ac8b022a-ab7a-4e45-9a74-e3dbee60113a.gif)|![wave](https://user-images.githubusercontent.com/49257696/162746233-4151ea3c-57b4-48ff-9f1c-d6fd69fe3dbb.gif)|

## CPU vs. GPU
By default, our code trains PINNs on GPU. To run on CPU, one should refer to
<br>
<code>
  main.py
</code>
<br>
and change 
<br>
<code>
  with tf.device("/device:GPU:0"):
</code>
<br>
to
<br>
<code>
  with tf.device("/device:CPU:0"):
</code>
<br>
in the corresponding directories. For our environment, GPU speed-up marked **25~30 times faster** training time than CPU-based learning (Intel Core i7-9700 & NVIDIA GeForce RTX 2070 / AMD Core Ryzen9 5950X & NVIDIA GeForce RTX 3090). Mini-batch training is also possible if the model does not fit CPU/GPU memory, however, we recommend full-batching to appreciate the best possible speed-up (or large mini-batch size). This is because CPU-GPU communication becomes frequent and slows down the overall performance when small mini-batch size is chosen. 

## FDM simulation vs. PINN inference
For most of the problems, this repo compares solutions yielded by FDM (Finite Difference Method) and PINN. Difference between them (we define this as PINN solution error) is reported in each directory. Regarding computational cost, PINN inference is faster than numerical integration by **~40x** for <code>04_diffusion</code> (same execution environment). Fair comparison was challenging for other problems, because we had to re-mesh the grid for FDM to converge (i.e. same mesh was employed for FDM and PINN in diffusion). 

## Dependencies
Tested on 
<br>
<code>
  python 3.8.10
</code>
<br>
with the following:
|Package|Version|
| :---: | :---: |
|numpy|1.22.1|
|scipy|1.7.3|
|tensorflow|2.8.0|

## Reference
[1] Raissi, M., Perdikaris, P., Karniadakis, G.E.: Physics-informed neural networks: A deep learning framework for solving forward and inverse problems involving nonlinear partial differential equations, *Journal of Computational Physics*, Vol. 378, pp. 686-707, 2019. ([paper](https://doi.org/10.1016/j.jcp.2018.10.045))
<br>
[2] Baydin, A.G., Pearlmutter, B.A., Radul, A.A., Siskind, J.M.: Automatic Differentiation in Machine Learning: A Survey, *Journal of Machine Learning Research*, Vol. 18, No. 1, pp. 5595–5637, 2018. ([paper](https://arxiv.org/abs/1502.05767))
<br>
[3] Rumelhart, D., Hinton, G., Williams, R.: Learning representations by back-propagating errors, *Nature*, Vol. 323, pp. 533–536, 1986. ([paper](https://doi.org/10.1038/323533a0))
<br>
[4] Jagtap, A.D., Kawaguchi, K., Karniadakis, GE.: Locally adaptive activation functions with slope recovery for deep and physics-informed neural networks, *Proceedings of Royal Society A*, pp. 4762020033420200334, 2020. ([paper](https://doi.org/10.1098/rspa.2020.0334))
