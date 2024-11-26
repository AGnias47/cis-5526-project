import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

import sys
sys.path.append(".")
from models.constants import DF_NO_DIRECTORS, LABEL_COLUMN, RANDOM_STATE, TRAIN_SIZE


def test_MSE(Y):
    Y_hat = 9 * np.random.rand(Y.shape[0]) + 1
    E_test = Y - Y_hat
    return float((1 / E_test.shape[0]) * np.matmul(E_test.transpose(), E_test))


if __name__ == "__main__":
    np.random.seed(RANDOM_STATE)
    df = pd.read_csv(DF_NO_DIRECTORS).fillna(0)
    _, _, _, y_test = train_test_split(
        df.drop([LABEL_COLUMN], axis=1, errors="ignore").to_numpy(),
        df[LABEL_COLUMN].to_numpy(),
        test_size=TRAIN_SIZE,
        random_state=RANDOM_STATE,
    )
    MSE = test_MSE(y_test)
    print(f"Mean squared error from random rating generator: {MSE}")
