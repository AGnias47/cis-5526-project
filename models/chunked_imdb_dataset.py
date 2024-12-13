"""
https://stackoverflow.com/questions/70551454/torch-dataloader-for-large-csv-file-incremental-loading
"""

import sys
from math import ceil, floor

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

sys.path.append(".")
from models.constants import (
    CHUNKS,
    DF_DIRECTORS,
    FEATURE_STARTING_INDEX,
    LABEL_COLUMN,
    ROWS,
)

LABEL_COLUMN_INDEX = 3


class ChunkedIMDBDataset(Dataset):
    def __init__(self, dataset_type):
        self.dataset_type = dataset_type
        self.chunksize = ROWS // CHUNKS

    def __len__(self):
        return CHUNKS

    def __getitem__(self, item):
        if item == 0:
            skiprows = 1
        else:
            skiprows = item * self.chunksize
        df = next(
            pd.read_csv(
                DF_DIRECTORS, skiprows=skiprows, chunksize=self.chunksize, header=None
            )
        )
        df = df.fillna(0)
        if self.dataset_type == "train":
            train_split = floor(df.shape[0] * 0.7)
            X = (
                df.drop(df.columns[LABEL_COLUMN_INDEX], axis=1)
                .to_numpy()[0:train_split, :][:, FEATURE_STARTING_INDEX:]
                .astype(np.float64)
            )
            y = df.iloc[:, LABEL_COLUMN_INDEX].to_numpy()[0:train_split]
        elif self.dataset_type == "validation":
            test_val_split = ceil(df.shape[0] * 0.7)
            X_test_val = df.to_numpy()[test_val_split:, :]
            val_split = floor(X_test_val.shape[0] * 0.5)
            X = (
                df.drop(df.columns[LABEL_COLUMN_INDEX], axis=1)
                .to_numpy()[val_split:, :][:, FEATURE_STARTING_INDEX:]
                .astype(np.float64)
            )
            y = df.iloc[:, LABEL_COLUMN_INDEX].to_numpy()[val_split:]
        elif self.dataset_type == "test":
            test_val_split = ceil(df.shape[0] * 0.7)
            X_test_val = df.to_numpy()[test_val_split:, :]
            test_split = ceil(X_test_val.shape[0] * 0.5)
            X = (
                df.drop([LABEL_COLUMN_INDEX], axis=1)
                .to_numpy()[test_split:, :][:, FEATURE_STARTING_INDEX:]
                .astype(np.float64)
            )
            y = df.iloc[:, LABEL_COLUMN_INDEX].to_numpy()[test_split:]
        else:
            raise ValueError("Invalid data type specified")
        return torch.tensor(X.astype(np.float32)), torch.tensor(y.astype(np.float32))
