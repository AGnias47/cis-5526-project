import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split

sys.path.append(".")
from models.constants import DF_NO_DIRECTORS, LABEL_COLUMN, RANDOM_STATE, TRAIN_VAL_SIZE, RESULTS_FILE


def metrics(Y):
    Y_hat = 9 * np.random.rand(Y.shape[0]) + 1
    E_test = Y - Y_hat
    return float(
        (1 / E_test.shape[0]) * np.matmul(E_test.transpose(), E_test)
    ), r2_score(Y, Y_hat)


if __name__ == "__main__":
    np.random.seed(RANDOM_STATE)
    df = pd.read_csv(DF_NO_DIRECTORS).fillna(0)
    _, _, _, y_test = train_test_split(
        df.drop([LABEL_COLUMN], axis=1, errors="ignore").to_numpy(),
        df[LABEL_COLUMN].to_numpy(),
        test_size=TRAIN_VAL_SIZE / 2,
        random_state=RANDOM_STATE,
    )
    mse, r2 = metrics(y_test)
    print(f"Mean squared error from random rating generator: {mse}")
    print(f"R2 score: {r2}")
    p = Path(RESULTS_FILE)
    if not p.exists():
        with open(RESULTS_FILE, "w") as F:
            F.write("Data,Model,MSE,R2\n")
    with open(RESULTS_FILE, "a") as F:
        F.write(f"NoDir,RandomRatingGenerator,{mse},{r2}\n")
