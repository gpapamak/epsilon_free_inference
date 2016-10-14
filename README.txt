
==========================================================
 Code for reproducing the experiments in the paper:

 "Fast epsilon-free Inference of Simulation Models with
  Bayesian Conditional Density Estimation"
  
 George Papamakarios, University of Edinburgh, 2016
==========================================================


*** This folder contains: ***

+ demos
  Folder containing four subfolders, one for each demo in the paper.

  + mixture_of_gaussians_demo
    - mog_main.py         sets up the model
    - mog_abc.py          runs abc methods
    - mog_mdn.py          runs mdn methods
    - mog_res.py          collects and plots results

  + bayesian_linear_regression_demo
    - blr_main.py         sets up the model
    - blr_abc.py          runs abc methods
    - blr_mdn.py          runs mdn methods
    - blr_res.py          collects and plots results

  + lotka_volterra_demo
    - lv_main.py          sets up the model
    - lv_abc.py           runs abc methods
    - lv_mdn.py           runs mdn methods
    - lv_res.py           collects and plots results

  + mg1_queue_demo
    - mg1_main.py         sets up the model
    - mg1_abc.py          runs abc methods
    - mg1_mdn.py          runs mdn methods
    - mg1_res.py          collects and plots results

+ util
  Folder with utility classes and functions.

  - helper.py             various helper functions
  - pdf.py                Gaussians and mixtures of Gaussians
  - NeuralNet.py          neural nets with and without SVI
  - mdn.py                MDNs with and without SVI
  - DataStream.py         provides data minibatches for training
  - LossFunction.py       loss functions for training
  - StepStrategy.py       optimization algorithms, including Adam
  - Trainer.py            trains a neural net or MDN, SVI or not
  - MarkovJumpProcess.py  markov jump processes, including lotka-volterra

