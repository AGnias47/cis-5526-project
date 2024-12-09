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
    DF_DIRECTORS,
    DF_NO_DIRECTORS,
    DF_SENTIMENT_DATA,
    FEATURE_STARTING_INDEX,
    LABEL_COLUMN,
)


class IMDBDataset(Dataset):
    def __init__(self, directors=False, sentiment=False):
        if directors and sentiment:
            raise ValueError("Only 1 of sentiment or directors can be used")
        if directors:
            self.df = pd.read_csv(DF_DIRECTORS).fillna(0)
        elif sentiment:
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
