import numpy as np
from torch import nn
import pandas as pd
from mlp import MLP
from sklearn.model_selection import train_test_split

from skorch import NeuralNetClassifier

white_wine_data = pd.read_csv("./data/winequality/winequality-white.csv",
                                      sep=";",
                                      header=1)
red_wine_data = pd.read_csv("./data/winequality/winequality-red.csv",
                                      sep=";",
                                      header=1)
data = np.concatenate((white_wine_data.values, red_wine_data.values), axis=0)

input_size = 11
hidden_size = 500
num_classes = 3
num_epochs = 5
batch_size = 20
learning_rate = 0.01

X = data[:, :-1].astype(np.float32)
y = data[:, -1].astype(np.int64)
for i in range(1, 3 + 1):
    y[np.logical_and(y <= i * 10 / 3,  y >= (i - 1) * 10 / 3)] = i - 1

train_x, test_x, train_y, test_y = train_test_split(X,y,test_size=0.33)

net = NeuralNetClassifier(
    MLP,
    criterion=nn.CrossEntropyLoss,
    max_epochs=30,
    lr=0.1,
    module__input_size=11,
    module__num_classes=3,
    device='cuda'
)

from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import randint
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from scipy.stats import uniform

params = {
    'net__lr': uniform(loc=0, scale=0.2),
    'net__module__hidden_size': randint(100, 1000),
    'net__optimizer__weight_decay': uniform(loc=0, scale=0.1),
    'net__batch_size': randint(10, 200)
}

model = Pipeline(steps=[("scaler",StandardScaler()), ("net",net)])

rs = RandomizedSearchCV(model, params, refit=True, cv=3, scoring='accuracy', n_iter=100, n_jobs=-1)

rs.fit(train_x, train_y)
print(rs.best_score_, rs.best_params_)

print(rs.score(test_x, test_y))