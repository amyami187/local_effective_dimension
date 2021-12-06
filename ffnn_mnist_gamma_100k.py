import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import torch.nn.functional as tF
import torch.nn as nn
import numpy as np
from nngeometry.layercollection import LayerCollection
from nngeometry.metrics import FIM
from nngeometry.object import PMatKFAC
import torch.optim as optim
import matplotlib.pyplot as plt
plt.rcParams.update({'font.size': 18})
torch.manual_seed(42)


def RHS(gamma, number, ed):
    """
    Computes the log of the RHS of equation (6) in the Effective Dimension of Machine Learning Models paper.
    :param gamma: specified as per the effective dimension definition
    :param number: number of data samples assumed to be available
    :param ed: int, value of the effective dimension
    :return: log of RHS of eqn 6
    """
    return np.log(2*np.sqrt(d)) + ed/2 * np.log(gamma*number/2*np.pi*np.log(number)) - 16*np.pi*np.log(number)/gamma


def ed(number, gamma):
    """
    Computes the effective dimension after the normalized Fisher information is stored.
    :param number: number of data samples assumed to be available
    :param gamma: specified as per the effective dimension definition
    :return: effective dimension as a scalar value
    """
    n = number
    const = (gamma * n) / (2 * np.pi * np.log(n))
    numerator = np.sum(np.log(1/const + FI))
    ed = d + numerator / np.log(const)
    return ed


class ClassicalNeuralNetwork(nn.Module):
    def __init__(self, size):
        super(ClassicalNeuralNetwork, self).__init__()
        self.size = size
        self.layers = nn.ModuleList(
            [nn.Linear(self.size[i - 1], self.size[i], bias=False) for i in range(1, len(self.size))])
        self.d = sum(size[i] * size[i + 1] for i in range(len(size) - 1))

    def forward(self, x):
        x = x.view(x.size(0), -1)
        for i in range(len(self.size) - 2):
            x = tF.leaky_relu(self.layers[i](x))
        x = self.layers[-1](x)
        return x


#################################### DATA LOAD ######################################
batch_size = 50
trainset = datasets.MNIST(root='./data/', train=True, download=True,
                          transform=transforms.ToTensor())

trainloader = DataLoader(
    dataset=trainset,
    batch_size=batch_size,
    shuffle=True)

testset = datasets.MNIST(root='./data/', train=False, download=True,
                           transform=transforms.ToTensor())
testloader = DataLoader(
    dataset=testset,
    batch_size=batch_size,
    shuffle=True)

# specify the model
convnet = ClassicalNeuralNetwork(size=[28*28, 110, 110, 10]).to('cpu')
layer_collection = LayerCollection.from_model(convnet)
d = layer_collection.numel()
print('d= ', d)

# Train the model
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(convnet.parameters(), lr=0.01,
                      momentum=0.9)

epochs = 200
for epoch in range(epochs):  # loop over the dataset multiple times
    running_loss = 0.0
    epoch_loss = 0
    for i, data in enumerate(trainloader, 0):
        # get the inputs; data is a list of [inputs, labels]
        inputs, labels = data
        # zero the parameter gradients
        optimizer.zero_grad()
        outputs = convnet(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        epoch_loss += outputs.shape[0] * loss.item()
        # print statistics
        running_loss += loss.item()
    print('epoch: ', epoch + 1, ' loss: ', epoch_loss / len(trainset))

final_train_loss = epoch_loss / len(trainset)
print('Train loss: ', final_train_loss)

# Check the generalization error
correct = 0
total = 0
with torch.no_grad():
    for data in testloader:
        images, labels = data
        outputs = convnet(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

print('Test loss: %.5f' %
      (1 - correct / total))
test_e = 1 - (correct / total)

# compute FIM
FI = FIM(model=convnet,
         loader=trainloader,
         representation=PMatKFAC,
         device='cpu',
         n_output=10)

# normalize the Fisher
eigs = FI.get_eig_F()
tr = FI.trace().detach().numpy()
FI = d * eigs / tr


# Compute largest gamma value for which the error bound is non-vacuous (i.e. RHS remains negative)
ns = [500000, 1000000, 2000000, 5000000, 10000000]
for number in ns:
    # specify starting gamma value
    gamma = 0.001
    effdim = ed(number, gamma)
    RS = RHS(gamma, number, effdim)
    print('###########################################################')
    print('n = ', number)
    print('###########################################################')
    print('gamma = ', gamma, 'and RHS = ', RS)
    print('effective dim = ', effdim)
    while RS <= 0:
        gamma = gamma + 0.0001
        if gamma > 1:
            break
        effdim = ed(number, gamma)
        RS = RHS(gamma, number, effdim)
        print('gamma = ', gamma, ' and RHS = ', RS)
        print('effective dim = ', effdim)

