import numpy as np
from sklearn.datasets import make_classification
from torch import nn
import torch.nn.functional as F
import pandas as pd
from mlp import MLP

from skorch import NeuralNetClassifier

data = pd.read_csv("./data/winequality/winequality-white.csv",
                                      sep=";",
                                      header=1)
data = data.values
X = data[:, :-1].astype(np.float32)
y = data[:, -1].astype(np.int64)


net = NeuralNetClassifier(
    MLP,
    criterion=nn.CrossEntropyLoss,
    max_epochs=10,
    lr=0.1,
    module__input_size=11,
    module__num_classes=10,
    device='cuda'
)

from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import randint
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from scipy.stats import uniform

params = {
    'neuralnetclassifier__lr': uniform(loc=0, scale=1),
    'neuralnetclassifier__module__hidden_size': randint(100, 1000),
    'neuralnetclassifier__max_epochs': randint(10, 30),
}

model = make_pipeline(StandardScaler(), net)

rs = RandomizedSearchCV(model, params, refit=False, cv=3, scoring='accuracy', n_iter=50, n_jobs=-1)

rs.fit(X, y)
print(rs.best_score_, rs.best_params_)