from functions import *
torch.manual_seed(42)

#####################################################################################################################

# Experiment 1: increase model size, train and compute local ED

# MNIST
model_sizes = [[30, 30], [60, 60], [90, 90], [110, 110]]
model_names = ['mnist_20k', 'mnist_50k', 'mnist_80k', 'mnist_100k']

# run the local ED experiment with increasing model size for MNIST, this loop could be parallelized to run faster
for i in range(len(model_sizes)):
    # computes and saves the local ED, training error and test error
    compute_and_save_local_ed(size=model_sizes[i], file_extension=model_names[i], MNIST=True)

# CIFAR10
model_sizes = [[200, 200], [400, 300], [900, 900], [1500, 1500], [1900, 1900], [2200, 2200]]
model_names = ['cifar_500k', 'cifar_1m', 'cifar_3m', 'cifar_5m', 'cifar_8m', 'cifar_10m']

# run the local ED experiment with increasing model size for CIFAR10
for i in range(len(model_sizes)):
    # computes and saves the local ED, training error and test error
    compute_and_save_local_ed(size=model_sizes[i], file_extension=model_names[i], MNIST=False)

#####################################################################################################################

# Experiment 2: fix model size, train on increasingly random labels and compute/save local ED

random_proportions = [0.2, 0.4, 0.6, 0.8, 1]

# MNIST
model_names = ['mnist_rand20', 'mnist_rand40', 'mnist_rand60', 'mnist_rand80', 'mnist_rand100']
model_size = [110, 110]

for i in range(len(random_proportions)):
    compute_and_save_local_ed(size=model_size, file_extension=model_names[i], MNIST=True, random=random_proportions[i])

# CIFAR10
model_names = ['cifar_rand20', 'cifar_rand40', 'cifar_rand60', 'cifar_rand80', 'cifar_rand100']
model_size = [2200, 2200]

for i in range(len(random_proportions)):
    compute_and_save_local_ed(size=model_size, file_extension=model_names[i], MNIST=False, random=random_proportions[i])
