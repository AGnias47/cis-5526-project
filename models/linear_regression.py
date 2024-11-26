from math import ceil, floor
from uuid import uuid4

import numpy as np
import pandas as pd
from numpy.linalg import inv
from rainbow_tqdm import tqdm
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split

import sys

sys.path.append(".")
from models.constants import (
    CHUNKS,
    DF_DIRECTORS,
    DF_NO_DIRECTORS,
    FEATURE_STARTING_INDEX,
    LABEL_COLUMN,
    LINEAR_REGRESSION_MODEL,
    RANDOM_STATE,
    ROWS,
    TRAIN_SIZE,
)
from models.metrics import mse_V


def closed_form_linear_regression(X_train, y_train):
    X_train = X_train[:, FEATURE_STARTING_INDEX:].astype(np.float64)
    X_train_T = X_train.transpose()
    XTX = np.matmul(X_train_T, X_train)
    XTXI = inv(XTX)
    XTY = np.matmul(X_train_T, y_train)
    return np.matmul(XTXI, XTY)


def closed_form_ridge_regression(X_train, y_train, lambda_val=0.1):
    X = X_train[:, FEATURE_STARTING_INDEX:].astype(np.float64)
    XT = X.transpose()
    XTX = np.matmul(XT, X)
    LI = lambda_val * np.ones(XTX.shape)
    XTXI = inv(XTX + LI)
    XTY = np.matmul(XT, y_train)
    return np.matmul(XTXI, XTY)


def metrics(X_test, y_test, W):
    X_test = X_test[:, FEATURE_STARTING_INDEX:].astype(np.float64)
    Y_hat_test = np.matmul(X_test, W)
    E_test = y_test - Y_hat_test
    return mse_V(E_test), r2_score(y_test, Y_hat_test)


def gradient_descent_linear_regression(X_train, y_train, epochs=100, alpha=0.1):
    # Used for sanity check https://www.geeksforgeeks.org/gradient-descent-in-linear-regression/
    X = X_train[:, FEATURE_STARTING_INDEX:].astype(np.float64)
    XT = X.transpose()
    W = np.random.rand(X.shape[1])
    for _ in range(epochs):
        XW = np.matmul(X, W)
        XTXW = np.matmul(XT, XW)
        XTY = np.matmul(XT, y_train)
        gMSE = (2 / X.shape[0]) * (XTXW - XTY)
        W = W - alpha * gMSE
    return W


def mini_batch_gradient_descent_linear_regression(
    X_train, y_train, epochs=10, alpha=0.1, batch_size=32, W=None
):
    X_full = X_train[:, FEATURE_STARTING_INDEX:].astype(np.float64)
    if W is None:
        W = np.random.rand(X_full.shape[1])
    for _ in range(epochs):
        i = 0
        X = X_full[i : i + batch_size, :]
        Y = y_train[i : i + batch_size]
        while X.any():
            XT = X.transpose()
            XW = np.matmul(X, W)
            XTXW = np.matmul(XT, XW)
            XTY = np.matmul(XT, Y)
            gMSE = (2 / X.shape[0]) * (XTXW - XTY)
            W = W - alpha * gMSE
            i += batch_size
            X = X_full[i : i + batch_size, :]
            Y = y_train[i : i + batch_size]
    return W


def main_df_no_directors():
    with open("results/no_dir/linear_regression_results.csv", "w") as F:
        F.write("Model,MSE,R2\n")
    df = pd.read_csv(DF_NO_DIRECTORS).fillna(0)
    X_train, X_test, y_train, y_test = train_test_split(
        df.drop([LABEL_COLUMN], axis=1, errors="ignore").to_numpy(),
        df[LABEL_COLUMN].to_numpy(),
        test_size=TRAIN_SIZE,
        random_state=RANDOM_STATE,
    )
    W = closed_form_linear_regression(X_train, y_train)
    mse, r2 = metrics(X_test, y_test, W)
    print(f"Mean squared error from closed form solution: {mse}")
    with open("results/no_dir/linear_regression_results.csv", "a") as F:
        F.write(f"Closed Form,{mse},{r2}\n")
    W = closed_form_ridge_regression(X_train, y_train, lambda_val=0.8)
    mse, r2 = metrics(X_test, y_test, W)
    print(f"Mean squared error from ridge regression: {mse}")
    with open("results/no_dir/linear_regression_results.csv", "a") as F:
        F.write(f"Ridge Regression,{mse},{r2}\n")
    W = gradient_descent_linear_regression(X_train, y_train, epochs=1000, alpha=0.1)
    mse, r2 = metrics(X_test, y_test, W)
    print(f"Mean squared error from batch gradient descent linear regression: {mse}")
    with open("results/no_dir/linear_regression_results.csv", "a") as F:
        F.write(f"Batch Gradient Descent,{mse},{r2}\n")
    W = mini_batch_gradient_descent_linear_regression(
        X_train, y_train, epochs=10, alpha=0.1
    )
    mse, r2 = metrics(X_test, y_test, W)
    print(
        f"Mean squared error from mini-batch gradient descent linear regression: {mse}"
    )
    with open("results/no_dir/linear_regression_results.csv", "a") as F:
        F.write(f"Mini-batch GD,{mse},{r2}\n")
    W = mini_batch_gradient_descent_linear_regression(
        X_train, y_train, epochs=10, alpha=0.01, batch_size=1
    )
    mse, r2 = metrics(X_test, y_test, W)
    print(f"Mean squared error from SGD linear regression: {mse}")
    with open("results/no_dir/linear_regression_results.csv", "a") as F:
        F.write(f"SGD,{mse},{r2}\n")


def main_df_directors():
    X_test, y_test, W = None, None, None
    epochs = 10
    MSE = 0
    R2 = 0
    for epoch in tqdm(range(epochs + 1)):
        if epoch == epochs:
            print("Training complete, onto testing")
        for chunk in tqdm(
            pd.read_csv(DF_DIRECTORS, chunksize=ROWS // CHUNKS), total=CHUNKS
        ):
            df = chunk.fillna(0)
            if epoch < epochs:
                train_split = floor(df.shape[0] * 0.8)
                X_train = df.drop([LABEL_COLUMN], axis=1, errors="ignore").to_numpy()[
                    0:train_split, :
                ]
                y_train = df[LABEL_COLUMN].to_numpy()[0:train_split]
                W = mini_batch_gradient_descent_linear_regression(
                    X_train, y_train, epochs=1, alpha=0.1, W=W
                )
            else:
                test_split = ceil(df.shape[0] * 0.8)
                X_test = df.drop([LABEL_COLUMN], axis=1, errors="ignore").to_numpy()[
                    test_split:, :
                ]
                y_test = df[LABEL_COLUMN].to_numpy()[test_split:]
                chunk_metrics = metrics(X_test, y_test, W)
                MSE += chunk_metrics[0]
                R2 += chunk_metrics[0]
    print(
        f"Mean squared error from mini-batch gradient descent linear regression: {MSE/(CHUNKS+1)}"
    )
    with open(f"{str(uuid4())}.npy", "wb") as F:
        np.save(F, W)


def individual_row_test(df, W):
    X_test_all = df.drop([LABEL_COLUMN], axis=1, errors="ignore").to_numpy()
    X_test = X_test_all[:, FEATURE_STARTING_INDEX:].astype(np.float64)
    y_test = df[LABEL_COLUMN].to_numpy()
    Y_hat_test = np.matmul(X_test, W)
    E = abs(y_test - Y_hat_test)
    with open("results/linear_regression_sample_results.txt", "w") as F:
        F.write("Movie,Predicted Rating,Actual Rating,Error\n")
        for i in range(len(y_test)):
            F.write(
                f"{X_test_all[i][1]},{round(Y_hat_test[i], 1)},{y_test[i]},{round(E[i], 1)}\n"
            )


def test_df_directors_model():
    with open(LINEAR_REGRESSION_MODEL, "rb") as F:
        W = np.load(F)
    for chunk in pd.read_csv(DF_DIRECTORS, chunksize=ROWS // CHUNKS):
        individual_row_test(chunk[271:].fillna(0), W)
        break


if __name__ == "__main__":
    main_df_no_directors()
    # main_df_directors()
    # test_df_directors_model()
