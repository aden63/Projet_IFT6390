from winedataset import WineDataset
import torch
from mlp import MLP
from torch.autograd import Variable
import torch.nn as nn
import numpy as np
from torch.utils.data import SubsetRandomSampler
from torch.utils.data import ConcatDataset

# Hyper Parameters
input_size = 11
hidden_size = 500
num_classes = 3
num_epochs = 5
batch_size = 20
learning_rate = 0.01

white_wine_dataset = WineDataset(root='./data/winequality/', wine_type='white', n_classes=num_classes)
red_wine_dataset = WineDataset(root='./data/winequality/', wine_type='red', n_classes=num_classes)
wine_dataset = ConcatDataset([white_wine_dataset, red_wine_dataset])

dataset_size = len(wine_dataset)
indices = np.arange(dataset_size)
split = int(np.floor(0.3 * dataset_size))
np.random.shuffle(indices)
train_indices, test_indices = indices[split:], indices[:split]

train_sampler = SubsetRandomSampler(train_indices)
test_sampler = SubsetRandomSampler(test_indices)

# Data Loader (Input Pipeline)
train_loader = torch.utils.data.DataLoader(dataset=wine_dataset,
                                       batch_size=batch_size,
                                       sampler=train_sampler)

test_loader = torch.utils.data.DataLoader(dataset=wine_dataset,
                                       batch_size=batch_size,
                                       sampler=test_sampler)

net = MLP(input_size, hidden_size, num_classes)
net.cuda()

# Loss and Optimizer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(net.parameters(), lr=learning_rate)

# Train the Model
for epoch in range(num_epochs):
    for i, (features, labels) in enumerate(train_loader):
        # Convert torch tensor to Variable
        features = Variable(features).cuda()
        labels = Variable(labels).cuda()

        # Forward + Backward + Optimize
        optimizer.zero_grad()  # zero the gradient buffer
        outputs = net(features)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        if (i + 1) % batch_size == 0:
            print('Epoch [%d/%d], Step [%d/%d], Loss: %.4f'
                  % (epoch + 1, num_epochs, i + 1, len(train_sampler) // batch_size, loss.item()))

# Test the Model
correct = 0
total = 0
for features, labels in test_loader:
    features = Variable(features).cuda()
    outputs = net(features)
    _, predicted = torch.max(outputs.data, 1)
    total += labels.size(0)
    correct += (predicted.cpu() == labels).sum()

print('Accuracy of the network on the %d test examples: %d %%' % (len(test_indices), 100 * correct / total))
# Save the Model
torch.save(net.state_dict(), 'model.pkl')

