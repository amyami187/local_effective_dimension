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


######################################################################################
# specify number of samples to estimate the effective dimension
samples = 1000
######################################################################################

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


#################################### DATA LOAD ######################################
batch_size = 50
trainset = datasets.CIFAR10(root='./data/', train=True, download=True,
                          transform=transform)

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

test_error = []
train_error = []

######################################################################################
# specify model
convnet = ClassicalNeuralNetwork(size=[crop*crop*3, 2200, 2200, 10]).to('cpu')
layer_collection = LayerCollection.from_model(convnet)

d = layer_collection.numel()
print('d= ', d)
# Train the model
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(convnet.parameters(), lr=0.001,
                      momentum=0.9)

# train the model
torch.manual_seed(42)
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
with torch.no_grad():
    for data in testloader:
        images, labels = data
        outputs = convnet(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

print('midpoint test loss: %.5f' %
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
const = n/(2*np.pi*np.log(n))

# Calculate the midpoint ED estimate
eigs = FI.get_eig_F()
tr = FI.trace().detach().numpy()
numerator = np.sum(np.log(1/const + d/tr*eigs))
ed = d + numerator/np.log(const)
print('midpoint ed: ', ed)
print('midpoint ed normalised: ', ed/d)
np.save('ed_midpoint.npy', ed/d)
np.save('test_loss_midpoint.npy', test_error)
np.save('midpoint_trace.npy', tr)
print('###########################################################################################################')

############# SAMPLED ED ###################

traces = []
eps = 1/torch.sqrt(torch.tensor(n))
test_losses = []
# store the traces here to calculate the average trace as per the ED calculation
torch.manual_seed(42)
for i in range(samples):
    # sample parameters from eps ball around midpoint
    r = -eps*torch.rand(1) + eps
    x = -2*torch.rand(d)+1
    x = torch.nn.functional.normalize(x, dim=0)
    x = r*x
    x = x + torch.nn.utils.parameters_to_vector(convnet.parameters())
    # reload model with updated params
    torch.nn.utils.vector_to_parameters(x, convnet.parameters())
    del x

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

    print('sample: ', i)
    print('test loss: %.5f' %
          (1 - correct / total))
    test_e = 1 - (correct / total)
    test_losses.append(test_e)

    # store the traces
    FI = FIM(model=convnet,
             loader=trainloader,
             representation=PMatKFAC,
             device='cpu',
             n_output=10)

    traces.append(FI.trace().detach().numpy())

np.save('traces.npy', traces)

# compute the average trace
traces = np.mean(np.array(traces))

# compute the zs
print('Computing sampled ED......')
z = []
torch.manual_seed(42)
for i in range(samples):
    r = -eps * torch.rand(1) + eps
    x = -2 * torch.rand(d) + 1
    x = torch.nn.functional.normalize(x, dim=0)
    x = r * x
    x = x + torch.nn.utils.parameters_to_vector(convnet.parameters())
    # reload model with updated params
    torch.nn.utils.vector_to_parameters(x, convnet.parameters())
    del x

    FI = FIM(model=convnet,
             loader=trainloader,
             representation=PMatKFAC,
             device='cpu',
             n_output=10)

    f = FI.get_eig_F()
    f = 0.5 * np.sum(np.log(1+const*(d/traces)*f))
    z.append(f)

# compute the effective dimension as per the efficient numerical evaluation technique
eta = np.amax(z)
ed = 2*eta/np.log(const) + (2/np.log(const))*np.log((1/samples)*(np.sum([np.exp(z[i]-eta) for i in range(len(z))])))
np.save('z.npy', z)
np.save('ed_sampled.npy', ed/d)
np.save('test_losses.npy', test_losses)
print('sampled ed: ', ed/d)
