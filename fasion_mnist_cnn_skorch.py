import numpy as np
import torch
from torch import nn
import pandas as pd
from cnn import ConvNet
from sklearn.model_selection import train_test_split
from skorch import NeuralNetClassifier
from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import randint
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from scipy.stats import uniform
import mnist_reader

# Hyper Parameters
kernelNb1 = 32
kernelNb2 = 64
kernelSize1 = 5
kernelSize2 = 3
num_classes = 10
num_epochs = 5
batch_size = 100
learning_rate = 0.001
from scipy.stats import uniform as sp_rand

# MNIST Dataset
train_x, train_y = mnist_reader.load_mnist("data/fashion",'train')
test_x, test_y = mnist_reader.load_mnist("data/fashion", 't10k')

train_x = train_x.astype(np.float32)
train_y = train_y.astype(np.int64)

test_x = test_x.astype(np.float32)
test_y = test_y.astype(np.int64)

net = NeuralNetClassifier(
        ConvNet,
        criterion=nn.CrossEntropyLoss,
        max_epochs=25,
        module__num_classes=num_classes,
        device='cuda'
    )
params = {
    'net__lr': uniform(loc=0, scale=0.2),
    'net__module__kernelNumber1': randint(16, 256),
    'net__module__kernelNumber2': randint(16, 256),
    'net__module__kernelSize1': randint(3,7),
    'net__module__kernelSize2': randint(3,7),
    'net__optimizer__weight_decay': uniform(loc=0, scale=0.2),
    'net__batch_size': randint(10, 256)
}

model = Pipeline(steps=[("net",net)])

rs = RandomizedSearchCV(model, params, refit=True, cv=3, scoring='accuracy', n_iter=100, n_jobs=-1)

import time
start = time.time()
rs.fit(train_x, train_y)
print(rs.best_score_, rs.best_params_)
print(rs.score(test_x, test_y))
print(time.time() - start)