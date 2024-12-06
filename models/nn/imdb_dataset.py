"""
https://pytorch.org/tutorials/beginner/basics/data_tutorial.html
https://pytorch.org/docs/stable/data.html
"""

import torch
from torch.utils.data import Dataset, DataLoader, random_split

import pandas as pd
import numpy as np
import sys

sys.path.append(".")
from models.constants import (
    DF_NO_DIRECTORS,
    DF_DIRECTORS,
    LABEL_COLUMN,
    FEATURE_STARTING_INDEX,
    RANDOM_STATE,
)


class IMDBDataset(Dataset):
    def __init__(self, directors=False):
        if directors:
            self.df = pd.read_csv(DF_DIRECTORS).fillna(0)
        else:
            self.df = pd.read_csv(DF_NO_DIRECTORS).fillna(0)
        self.X = self.df.drop([LABEL_COLUMN], axis=1, errors="ignore").to_numpy()
        self.Y = self.df[LABEL_COLUMN].to_numpy()

    def __len__(self):
        return len(self.Y)

    def __getitem__(self, item):
        return self.X[item, FEATURE_STARTING_INDEX:].astype(np.float64), self.Y[item]

    def get_title(self, item):
        return self.X[item, 1]


def train_test_val(train_size=0.7, test_size=0.15, val_size=0.15, batch_size=64):
    train, test, val = random_split(
        IMDBDataset(),
        [train_size, test_size, val_size],
        generator=torch.Generator().manual_seed(RANDOM_STATE),
    )
    return (
        DataLoader(train, batch_size=batch_size),
        DataLoader(test, batch_size=batch_size),
        DataLoader(val, batch_size=batch_size),
    )


if __name__ == "__main__":
    train, test, val = train_test_val()
    X, Y = next(iter(train))
    x = next(iter(X))
    y = next(iter(Y))
    title = train.dataset.dataset.get_title(0)
    print(x)
    print(y)
    print(title)