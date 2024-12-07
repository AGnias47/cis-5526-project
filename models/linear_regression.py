import argparse
from math import ceil, floor
from uuid import uuid4

import numpy as np
import pandas as pd
from numpy.linalg import inv
from rainbow_tqdm import tqdm
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
    SAVED_MODELS_DIR,
    TRAIN_VAL_SIZE,
)
from models.metrics import mse_r2_V

LINEAR_REGRESSION_RESULTS = "results/linear_regression_results.csv"


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
    df = pd.read_csv(DF_NO_DIRECTORS).fillna(0)
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
                train_split = floor(df.shape[0] * 0.7)
                X_train = df.drop([LABEL_COLUMN], axis=1, errors="ignore").to_numpy()[
                    0:train_split, :
                ]
                y_train = df[LABEL_COLUMN].to_numpy()[0:train_split]
                W = mini_batch_gradient_descent_linear_regression(
                    X_train, y_train, epochs=1, alpha=0.1, W=W
                )
            else:
                test_val_split = ceil(df.shape[0] * 0.7)
                X_test_val = df.drop([LABEL_COLUMN], axis=1, errors="ignore").to_numpy()[
                    test_val_split:, :
                ]
                val_split = ceil(X_test_val.shape[0] * 0.5)
                X_val = df.drop([LABEL_COLUMN], axis=1, errors="ignore").to_numpy()[
                             val_split:, :
                             ]
                y_val = df[LABEL_COLUMN].to_numpy()[val_split:]
                chunk_metrics = mse_r2_V(X_val, y_val, W)
                MSE += chunk_metrics[0]
                R2 += chunk_metrics[1]
    print(
        f"Mean squared error from mini-batch gradient descent linear regression: {MSE/(CHUNKS+1)}"
    )
    with open("results/directors/linear_regression_results.csv", "w") as F:
        F.write("Model,MSE,R2\n")
        F.write(f"SGD,{MSE/(CHUNKS+1)},{R2/(CHUNKS+1)}")
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


def train():
    X_train, _, X_validation, y_train, _, y_validation = main_df_no_directors()
    W_closed_form = closed_form_linear_regression(X_train, y_train)
    W_ridge = closed_form_ridge_regression(X_train, y_train, lambda_val=0.8)
    W_gd = gradient_descent_linear_regression(X_train, y_train, epochs=1000, alpha=0.1)
    W_mini_batch = mini_batch_gradient_descent_linear_regression(
        X_train, y_train, epochs=10, alpha=0.1
    )
    W_sgd = mini_batch_gradient_descent_linear_regression(
        X_train, y_train, epochs=10, alpha=0.01, batch_size=1
    )
    for W, train_type in [
        (W_closed_form, "closed form"),
        (W_ridge, "ridge regression"),
        (W_gd, "gradient descent"),
        (W_mini_batch, "mini batch"),
        (W_sgd, "stochastic gradient descent"),
    ]:
        mse, r2 = mse_r2_V(X_validation, y_validation, W)
        print(f"Results for {train_type} linear regression: MSE: {mse}, R2: {r2}")
        with open(f"bin/W_{train_type}.npy", "wb") as F:
            np.save(F, W)


def test():
    with open(LINEAR_REGRESSION_RESULTS, "w") as F:
        F.write("Data,Model,MSE,R2\n")
    _, X_test, _, _, y_test, _ = main_df_no_directors()
    for train_type in [
        "closed form",
        "ridge regression",
        "gradient descent",
        "mini batch",
        "stochastic gradient descent",
    ]:
        with open(f"{SAVED_MODELS_DIR}/W_{train_type}.npy", "rb") as F:
            W = np.load(F)
        mse, r2 = mse_r2_V(X_test, y_test, W)
        print(f"Results for {train_type} linear regression: MSE: {mse}, R2: {r2}")
        with open(LINEAR_REGRESSION_RESULTS, "a") as F:
            F.write(
                f"NoDir,LinearRegression{train_type.title().replace(" ", "")},{mse},{r2}"
            )


if __name__ == "__main__":
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument("-a", "--all", action="store_true", help="Run all actions")
    arg_parser.add_argument(
        "-tr",
        "--train",
        action="store_true",
        help="Train the dataframe without directors",
    )
    arg_parser.add_argument(
        "--test", action="store_true", help="Test the dataframe without directors"
    )
    arg_parser.add_argument(
        "-c",
        "--chunked",
        action="store_true",
        help="Train and test the dataframe with directors",
    )
    args = arg_parser.parse_args()
    if not any([args.all, args.train, args.test, args.chunked]):
        arg_parser.print_help()
    if args.all or args.train:
        train()
    if args.all or args.test:
        test()
    if args.all or args.chunked:
        main_df_directors()
        test_df_directors_model()
