import torch
import torch.nn as nn
import torchvision.datasets as dsets
import torchvision.transforms as transforms
from torch.autograd import Variable
from skorch import NeuralNetClassifier
from skorch.dataset import CVSplit
import mnist_reader
from mlp import MLP
from sklearn.model_selection import RandomizedSearchCV

# Hyper Parameters
input_size = 784
hidden_size = 500
num_classes = 10
num_epochs = 5
batch_size = 100
learning_rate = 0.001
from scipy.stats import uniform as sp_rand

# MNIST Dataset
mnist_train_x, mnist_train_y = mnist_reader.load_mnist("data/fashion",'train')
mnist_test = mnist_reader.load_mnist("data/fashion", 't10k')

train_dataset = dsets.FashionMNIST(root='./data',
                            train=True,
                            transform=transforms.ToTensor(),
                            download=True)

test_dataset = dsets.FashionMNIST(root='./data',
                           train=False,
                           transform=transforms.ToTensor())

# Data Loader (Input Pipeline)
train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                           batch_size=batch_size,
                                           shuffle=True)

test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                          batch_size=batch_size,
                                          shuffle=False)


net = NeuralNetClassifier(MLP(input_size, hidden_size, num_classes),
                          criterion=nn.CrossEntropyLoss,
                          dataset=train_dataset,
                          max_epochs=25,
                          device='cuda')

params = {'lr':sp_rand}

random_search = RandomizedSearchCV(net, params, scoring='accuracy')

random_search.fit(mnist_train_x, mnist_train_y)

print(random_search.best_score_, random_search.best_params_)

# # Loss and Optimizer
# criterion = nn.CrossEntropyLoss()
# optimizer = torch.optim.SGD(net.parameters(), lr=learning_rate)
#
# # Train the Model
# for epoch in range(num_epochs):
#     for i, (images, labels) in enumerate(train_loader):
#         # Convert torch tensor to Variable
#         images = Variable(images.view(-1, 28 * 28)).cuda()
#         labels = Variable(labels).cuda()
#
#         # Forward + Backward + Optimize
#         optimizer.zero_grad()  # zero the gradient buffer
#         outputs = net(images)
#         loss = criterion(outputs, labels)
#         loss.backward()
#         optimizer.step()
#
#         if (i + 1) % 100 == 0:
#             print('Epoch [%d/%d], Step [%d/%d], Loss: %.4f'
#                   % (epoch + 1, num_epochs, i + 1, len(train_dataset) // batch_size, loss.item()))
#
# # Test the Model
# correct = 0
# total = 0
# for images, labels in test_loader:
#     images = Variable(images.view(-1, 28 * 28)).cuda()
#     outputs = net(images)
#     _, predicted = torch.max(outputs.data, 1)
#     total += labels.size(0)
#     correct += (predicted.cpu() == labels).sum()
#
# print('Accuracy of the network on the 10000 test images: %d %%' % (100 * correct / total))
# # Save the Model
# torch.save(net.state_dict(), 'model.pkl')