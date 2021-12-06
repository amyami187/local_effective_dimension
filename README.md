# Effective dimension of machine learning models

In this repository, there are several folders containing code to reproduce the results/figures from 
the manuscript titled "Effective dimension of machine learning models" (arXiv: to be linked soon). All code was generated using Python v3.7 and PyTorch v1.3.1 which can be pip installed. We add an addiitonal function to the [NNGeometry](https://nngeometry.readthedocs.io/en/latest/) library and provide the recommended installation below. The details of the implementation are contained in the arXiv paper with an explanation of each folder's contents and installation here.

## Installation 
This project requires Python version 3.7 and above, as well as PyTorch v1.3.1 and NNGeometry. Installation of PyTorch and all dependencies, can be done using pip:

`$ python -m pip install torch==1.3.1`

And for NNGeometry, we recommend cloning the repo:

`git clone https://github.com/amyami187/nngeometry.git`

##
### Local_ED_size
This folder contains standalone Python files that pertain to a particular feedforward neural network of a specific size. Every network consists of 2 hidden layers with differing numbers of neurons per layer. Each Python file trains a network on either MNIST or CIFAR10 datasets and subsequently computes the test error. Thereafter, the effective dimension is estimated using the final (trained) parameters. This is experiment is repeated 10 times per network on the same data split, but with different parameter initializations, in order to get an idea of the standard deviation around the effective dimension and test error estimates.

#### Expected run time
Models range in size from $d = 10^5$ parameters to $d = 10^7$. Depending on the size of the data, the network and the chosen batch size, the main bottleneck is ultimately the estimation of the Fisher information matrix (which is $d \cross d$ in size). We bypass this through use of the [KFAC approximation](https://arxiv.org/abs/1602.01407) of the Fisher, which has a [PyTorch implementation](https://nngeometry.readthedocs.io/en/latest/). The run time is thus greatly reduced and most of the computational time then goes to training the models. Our largest simulation takes at most 1 day to complete on 1 GPU with 1TB memory. 

### Randomization_experiment
We follow the experiment originally outlined in [Zhang et al.](https://arxiv.org/abs/1611.03530) which increasingly randomizes training labels and computes the generalization (test) error thereafter. We do this for two models: first with $d = 10^5$ on MNIST and second with $d = 10^7$ on CIFAR10. At each level of randomization, we also compute the effective dimension with the trained parameters. Each experiment is repeated 10 times.

This folder contains the standalone scripts for both models with increasing levels of randomization (in increments of 20 percent).


### Gamma_analysis


### Sensitivity_to_samples


________________________________________________________________________________________________________________________________________________________________
## License
**Apache 2.0** (see https://github.com/amyami187/local_effective_dimension/blob/master/LICENSE)
