import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
import sys

sys.path.append(".")
from models.constants import (
    LABEL_COLUMN,
    TRAIN_SIZE,
    RANDOM_STATE,
    FEATURE_STARTING_INDEX,
    DF_NO_DIRECTORS,
    DF_DIRECTORS,
)
from models.mse import mse_V


def data_without_directors():
    df = pd.read_csv(DF_NO_DIRECTORS)
    X_train, X_test, y_train, y_test = train_test_split(
        df.drop([LABEL_COLUMN], axis=1, errors="ignore").to_numpy(),
        df[LABEL_COLUMN].to_numpy(),
        test_size=TRAIN_SIZE,
        random_state=RANDOM_STATE,
    )
    X_train = X_train[:, FEATURE_STARTING_INDEX:].astype(np.float64)
    X_test = X_test[:, FEATURE_STARTING_INDEX:].astype(np.float64)
    return X_train, X_test, y_train, y_test


def random_forest(X_train, X_test, y_train, y_test, results_fname):
    for depth in [2, 25, 32, 50, 72, 100, 150, 250, None]:
        model = RandomForestRegressor(
            max_depth=depth, random_state=RANDOM_STATE, verbose=1
        )
        model.fit(X_train, y_train)
        Y_hat = model.predict(X_test)
        E = y_test - Y_hat
        MSE = mse_V(E)
        with open(results_fname, "w") as F:
            F.write(f"RandomForestRegressor,{MSE},{depth}")
        print(f"RandomForestRegressor,{MSE},{depth}")



def test_regression_models(X_train, X_test, y_train, y_test, results_fname):
    with open(results_fname, "w") as F:
        F.write("Model,MSE,depth\n")
    random_forest(X_train, X_test, y_train, y_test, results_fname)


if __name__ == "__main__":
    test_regression_models(*data_without_directors(), "results/rf_no_directors.csv")
