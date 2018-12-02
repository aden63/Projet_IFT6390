import numpy as np
import torch
from torch import nn
import pandas as pd
from mlp import MLP
from sklearn.model_selection import train_test_split
from skorch import NeuralNetClassifier
from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import randint
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from scipy.stats import uniform
import mnist_reader

# Hyper Parameters
input_size = 784
hidden_size = 500
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

classes = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat', 'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']
cm = np.zeros((10,10))
for i in range(10):
    net = NeuralNetClassifier(
            MLP,
            criterion=nn.CrossEntropyLoss,
            max_epochs=20,
            batch_size=70,
            lr=0.00086824634389979236,
            module__input_size=input_size,
            module__hidden_size=790,
            module__num_classes=num_classes,
            optimizer__weight_decay= 0.14883558801066421,
            device='cuda'
        )

    net.fit(train_x, train_y)

    predictions = net.predict(test_x)

    from sklearn.metrics import confusion_matrix

    cm += confusion_matrix(test_y, predictions) / 10

import matplotlib.pyplot as plt
import itertools
def plot_confusion_matrix(cm, classes,
                         normalize=False,
                         title='Confusion matrix',
                         cmap=plt.cm.Blues):
   """
   This function prints and plots the confusion matrix.
   Normalization can be applied by setting `normalize=True`.
   """
   print(cm)

   plt.imshow(cm, interpolation='nearest', cmap=cmap)
   plt.title(title)
   plt.colorbar()
   tick_marks = np.arange(len(classes))
   plt.xticks(tick_marks, classes, rotation=45)
   plt.yticks(tick_marks, classes)

   fmt = 'd'
   thresh = cm.max() / 2.
   for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
       plt.text(j, i, format(cm[i, j], fmt),
                horizontalalignment="center",
                color="white" if cm[i, j] > thresh else "black")

   plt.tight_layout()
   plt.ylabel('True label')
   plt.xlabel('Predicted label')

from matplotlib import rcParams
rcParams.update({'figure.autolayout': True})

plt.figure()
plot_confusion_matrix(cm.astype(np.int64), classes=classes)

plt.savefig("./results/fashion_mnist/confusion_matrix.png")