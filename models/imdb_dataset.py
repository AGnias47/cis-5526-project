"""
https://pytorch.org/tutorials/beginner/basics/data_tutorial.html
https://pytorch.org/docs/stable/data.html
https://stackoverflow.com/a/50308132/8728749 - converting numpy to tensor
https://stackoverflow.com/a/67456436/8728749 - datatype errors
"""

import sys

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

sys.path.append(".")
from models.constants import (
    DF_NO_DIRECTORS,
    DF_SENTIMENT_DATA,
    FEATURE_STARTING_INDEX,
    LABEL_COLUMN,
)


class IMDBDataset(Dataset):
    def __init__(self, sentiment=False):
        if sentiment:
            self.df = pd.read_csv(DF_SENTIMENT_DATA).fillna(0)
        else:
            self.df = pd.read_csv(DF_NO_DIRECTORS).fillna(0)
        self.X = self.df.drop([LABEL_COLUMN], axis=1, errors="ignore").to_numpy()
        self.Y = self.df[LABEL_COLUMN].to_numpy()

    def __len__(self):
        return len(self.Y)

    def __getitem__(self, item):
        return torch.tensor(
            self.X[item, FEATURE_STARTING_INDEX:].astype(np.float32)
        ), torch.tensor(self.Y[item].astype(np.float32))

    def get_title(self, item):
        return self.X[item, 1]
