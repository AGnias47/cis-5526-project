"""
https://pytorch.org/tutorials/beginner/basics/data_tutorial.html
https://pytorch.org/docs/stable/data.html
"""

import sys

import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, random_split

sys.path.append(".")
from models.constants import (
    DF_NO_DIRECTORS,
    LABEL_COLUMN,
    RANDOM_STATE,
    TRAIN_VAL_SIZE,
    DF_SA_DESC,
)
from models.imdb_dataset import IMDBDataset


def train_test_val_dataloaders(
    train_size=0.7, test_size=0.15, val_size=0.15, batch_size=64, directors=False
):
    train, test, val = random_split(
        IMDBDataset(directors),
        [train_size, test_size, val_size],
        generator=torch.Generator().manual_seed(RANDOM_STATE),
    )
    return (
        DataLoader(train, batch_size=batch_size),
        DataLoader(test, batch_size=batch_size),
        DataLoader(val, batch_size=batch_size),
    )


def train_test_val_df_no_dirs(sa_desc=False):
    if sa_desc:
        df_source = DF_SA_DESC
    else:
        df_source = DF_NO_DIRECTORS
    df = pd.read_csv(df_source).fillna(0)
    X_train, X_tv, y_train, y_tv = train_test_split(
        df.drop([LABEL_COLUMN], axis=1, errors="ignore").to_numpy(),
        df[LABEL_COLUMN].to_numpy(),
        test_size=TRAIN_VAL_SIZE,
        random_state=RANDOM_STATE,
    )
    X_test, X_validation, y_test, y_validation = train_test_split(
        X_tv,
        y_tv,
        test_size=0.5,
        random_state=RANDOM_STATE,
    )
    return X_train, X_test, X_validation, y_train, y_test, y_validation


if __name__ == "__main__":
    train, test, val = train_test_val_dataloaders()
    X, Y = next(iter(train))
    x = next(iter(X))
    y = next(iter(Y))
    title = train.dataset.dataset.get_title(0)
    print(x)
    print(y)
    print(title)
