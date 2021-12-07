# Effective dimension of machine learning models

In this repository, there are several folders containing code to reproduce the results/figures from 
the manuscript titled "Effective dimension of machine learning models" (arXiv: to be linked soon). All code was generated using Python v3.7 and PyTorch v1.3.1 which can be pip installed. We add an addiitonal function to the [NNGeometry](https://nngeometry.readthedocs.io/en/latest/) library and provide the recommended installation below. The details of the implementation are contained in the arXiv paper with an explanation of each folder's contents and installation here.

## Installation 
This project requires Python version 3.7 and above, as well as PyTorch v1.3.1 and NNGeometry. Installation of PyTorch and all dependencies, can be done using pip:

`$ python -m pip install torch==1.3.1`

And for NNGeometry, we recommend cloning the repo:

`git clone https://github.com/amyami187/nngeometry.git`

##
### [Local_ED](https://github.com/amyami187/local_effective_dimension/tree/main/Local_ED)
This folder contains two Python files. `ffnn_local_ed.py` contains all the functions necessary to train a feedforward neural network of a specified size on either MNIST or CIFAR10. The user has the freedom to specify the size of the model, as well as the data to train on. The `compute_and_save_local_ed()` function computes the final training error, the test error and the local effective dimension using trained parameters. This is inherently repeated 10 times on the same data split, but with different parameter initializations in order to get an idea of the standard deviation around the effective dimension and test error estimates. All results are then saved. The user also has freedom to choose a proportion at which to randomize the training data labels.

In our experiemnts, every network consists of 2 hidden layers with differing numbers of neurons per layer. The details of our specifications can be found in `run_experiments.py`.

### [Gamma_analysis](https://github.com/amyami187/local_effective_dimension/tree/main/Gamma_analysis)
We conduct a small search for the largest value of `gamma` such that generalization bound remains non-vacuous. For more details, see Table 2 and Appendix C in the "Effective dimension of machine learning models" manuscript. 

We use a model of the order `d = 100 000` trained on MNIST and calculate the effective dimension over different values for `n` (the number of data available as per the definition of the effective dimension). We begin by choosing `gamma = 0.001` and increase it by increments of `0.0001` till the bound becomes vacuous. 

### [Sensitivity_to_samples](https://github.com/amyami187/local_effective_dimension/tree/main/Sensitivity_to_samples)
In this script, we can specify the number of samples we wish to have to estimate the effective dimension for a model of the order `d = 10 000 000`. We use CIFAR10 and train the model. Thereafter, we draw samples from an epsilon ball around the trained parameters and estimate the effective dimension. 

###
## Expected run time
Models range in size from approximately `d = 20 000` parameters to `d = 10 000 000`. Depending on the size of the data, the network and the chosen batch size, the main bottleneck is ultimately the estimation of the Fisher information matrix (which is `d x d` in size). We bypass this through use of the [KFAC approximation](https://arxiv.org/abs/1602.01407) of the Fisher, which has a [PyTorch implementation](https://nngeometry.readthedocs.io/en/latest/). The run time is thus greatly reduced and most of the computational time then goes to training the models. Our largest simulation takes at most 1 day to complete on 1 GPU with 1TB memory. 

________________________________________________________________________________________________________________________________________________________________
## License
**Apache 2.0** (see https://github.com/amyami187/local_effective_dimension/blob/master/LICENSE)
