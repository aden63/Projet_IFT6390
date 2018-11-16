from torch.utils.data import Dataset
import os
import pandas as pd
import torch
import numpy as np

class WineDataset(Dataset):
    def __init__(self, root, wine_type, n_classes=10):
        self.root = os.path.expanduser(root)
        self.wine_type = wine_type
        self.n_classes = n_classes

        self.data = pd.read_csv(os.path.join(self.root, "winequality-{}.csv".format(self.wine_type)),
                                      sep=";",
                                      header=1)
        self.data = torch.Tensor(self.data.values)

    def __getitem__(self, index):
        X = self.data[index, :-1]
        y = self.data[index, -1].type(torch.LongTensor)
        for i in range(1, self.n_classes + 1):
            if y <= i*10/self.n_classes:
                y = i - 1
                break

        return X, y

    def __len__(self):
        return self.data.shape[0]