# Fast ε-free Inference of Simulation Models with Bayesian Conditional Density Estimation

Code for reproducing the experiments in the paper:

> G. Papamakarios, and I. Murray. _Fast ε-free Inference of Simulation Models with Bayesian Conditional Density Estimation_. Advances in Neural Information Processing Systems Conference. 2016.
> [[pdf]](https://arxiv.org/pdf/1605.06376v2.pdf) [[bibtex]](http://homepages.inf.ed.ac.uk/s1459647/bibtex/epsilon_free_inference.bib)


## This folder contains

#### **`demos`**

Folder containing four subfolders, one for each demo in the paper.

* `mixture_of_gaussians_demo`
  - `mog_main.py` --- sets up the model
  - `mog_abc.py` --- runs ABC methods
  - `mog_mdn.py` --- runs MDN methods
  - `mog_res.py` --- collects and plots results

* `bayesian_linear_regression_demo`
  - `blr_main.py` --- sets up the model
  - `blr_abc.py` --- runs ABC methods
  - `blr_mdn.py` --- runs MDN methods
  - `blr_res.py` --- collects and plots results

* `lotka_volterra_demo`
  - `lv_main.py` --- sets up the model
  - `lv_abc.py` --- runs ABC methods
  - `lv_mdn.py` --- runs MDN methods
  - `lv_res.py` --- collects and plots results

* `mg1_queue_demo`
  - `mg1_main.py` --- sets up the model
  - `mg1_abc.py` --- runs ABC methods
  - `mg1_mdn.py` --- runs MDN methods
  - `mg1_res.py` --- collects and plots results

#### **`util`**

Folder with utility classes and functions.

* `pdf.py`
    Gaussians and mixtures of Gaussians

* `NeuralNet.py`
  neural nets with and without SVI

* `mdn.py`
  MDNs with and without SVI

* `DataStream.py`
  provides data minibatches for training

* `LossFunction.py`
  loss functions for training

* `StepStrategy.py`
  optimization algorithms, including Adam

* `Trainer.py`
  trains a neural net or MDN, SVI or not

* `MarkovJumpProcess.py`
  Markov jump processes, including Lotka--Volterra
  
* `helper.py`
    various helper functions

