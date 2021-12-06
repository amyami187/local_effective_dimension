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
torch.manual_seed(42)


crop = 28
transform = transforms.Compose(
    [transforms.ToTensor(),
    transforms.CenterCrop(crop),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])


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


batch_size = 50
#trainset = datasets.CIFAR10(root='./data/', train=True, download=True,
#                          transform=transforms.ToTensor())

trainset = datasets.CIFAR10(root='./data/', train=True, download=True,
                          transform=transform)


#print(trainset.targets[:10])
# if we want to randomize the training data labels, specify % to randomize as random_perc
random_perc = 0.6
num_random_points = int(random_perc * len(trainset))

for i in range(num_random_points):
    idx = torch.randint(0, len(trainset)-1, (1,))
    trainset.targets[idx:idx + 1] = torch.randint(0, 9, (1,))

for i in range(len(trainset)):
    trainset.targets[i:i+1] = torch.tensor(trainset.targets[i:i+1])


trainloader = DataLoader(
    dataset=trainset,
    batch_size=batch_size,
    shuffle=True)

testset = datasets.CIFAR10(root='./data/', train=False, download=True,
                           transform=transform)

testloader = DataLoader(
    dataset=testset,
    batch_size=batch_size,
    shuffle=True)

effdim = []
effdim_norm = []
test_error = []
train_error = []

for j in range(10):
    convnet = ClassicalNeuralNetwork(size=[crop*crop*3, 2200, 2200, 10]).to('cpu')
    layer_collection = LayerCollection.from_model(convnet)
    if j == 0:
        d = layer_collection.numel()
        print('d= ', d)

    # Train the model
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(convnet.parameters(), lr=0.001, momentum=0.9) #optimizer = optim.Adam(convnet.parameters(), lr=0.001)
    print('iteration: ', j)
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

    train_error.append(final_train_loss)
    print('Train loss: ', final_train_loss)

    # Check the generalization error

    correct = 0
    total = 0
    # since we're not training, we don't need to calculate the gradients for our outputs
    with torch.no_grad():
        for data in testloader:
            images, labels = data
            # calculate outputs by running images through the network
            outputs = convnet(images)
            # the class with the highest energy is what we choose as prediction
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    print('Test loss: %.5f' %
          (1 - correct / total))
    test_e = 1 - (correct / total)
    test_error.append(test_e)

    # compute FIM
    FI = FIM(model=convnet,
            loader=trainloader,
            representation=PMatKFAC,
            device='cpu',
            n_output=10)

    n = 60000
    eigs = FI.get_eig_F()
    tr = FI.trace().detach().numpy()
    const = n/(2*np.pi*np.log(n))
    kappa = const*d/tr
    numerator = np.sum(np.log(1+kappa*eigs))
    ed = numerator/np.log(const)
    effdim.append(ed)
    norm_ed = ed / d
    effdim_norm.append(norm_ed)
    print('effective dimension: ', ed)
    print('normalised effective dimension: ', ed / d)

    ## OLD CODE WITH FULL FIM
    #n = torch.tensor(n)
    #FI = d * FI.get_dense_tensor() / FI.trace()
    #sgn, value = torch.slogdet(torch.eye(d) + const * FI)
    #ed = value / np.log(const)
    #ed = ed.item()
    print('###########################################################################################################')

np.save('ed_norm_cifar_rand60.npy', effdim_norm)
np.save('train_error_cifar_rand60.npy', train_error)
np.save('test_error_cifar_rand60.npy', test_error)

