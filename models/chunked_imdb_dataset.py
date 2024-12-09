"""
https://stackoverflow.com/questions/70551454/torch-dataloader-for-large-csv-file-incremental-loading
"""

import sys

import numpy as np
import pandas as pd
from math import floor, ceil
from torch.utils.data import Dataset

sys.path.append(".")
from models.constants import (
    DF_DIRECTORS,
    FEATURE_STARTING_INDEX,
    LABEL_COLUMN,
    ROWS,
    CHUNKS,
)


class ChunkedIMDBDataset(Dataset):
    def __init__(self, dataset_type):
        self.dataset_type = dataset_type
        self.chunksize = ROWS // CHUNKS

    def __len__(self):
        return ROWS

    def __getitem__(self, item):
        df = next(
            pd.read_csv(
                DF_DIRECTORS,
                skiprows=item * self.chunksize + 1,
                chunksize=self.chunksize,
            )
        )
        if self.dataset_type == "train":
            train_split = floor(df.shape[0] * 0.7)
            X = df.drop([LABEL_COLUMN], axis=1, errors="ignore").to_numpy()[
                0:train_split, :
            ][:, FEATURE_STARTING_INDEX:].astype(np.float64)
            y = df[LABEL_COLUMN].to_numpy()[0:train_split]
        elif self.dataset_type == "validation":
            test_val_split = ceil(df.shape[0] * 0.7)
            X_test_val = df.drop([LABEL_COLUMN], axis=1, errors="ignore").to_numpy()[
                test_val_split:, :
            ]
            val_split = floor(X_test_val.shape[0] * 0.5)
            X = df.drop([LABEL_COLUMN], axis=1, errors="ignore").to_numpy()[
                val_split:, :
            ][:, FEATURE_STARTING_INDEX:].astype(np.float64)
            y = df[LABEL_COLUMN].to_numpy()[val_split:]
        elif self.dataset_type == "test":
            test_val_split = ceil(df.shape[0] * 0.7)
            X_test_val = df.drop([LABEL_COLUMN], axis=1, errors="ignore").to_numpy()[
                test_val_split:, :
            ][:, FEATURE_STARTING_INDEX:].astype(np.float64)
            test_split = ceil(X_test_val.shape[0] * 0.5)
            X = df.drop([LABEL_COLUMN], axis=1, errors="ignore").to_numpy()[
                test_split:, :
            ]
            y = df[LABEL_COLUMN].to_numpy()[test_split:]
        else:
            raise ValueError("Invalid data type specified")
        return X, y
