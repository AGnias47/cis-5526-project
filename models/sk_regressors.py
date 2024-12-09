"""
https://stackoverflow.com/a/20662980/8728749
"""

import sys
import argparse
import pathlib

import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR

sys.path.append(".")
from models.constants import (
    FEATURE_STARTING_INDEX,
    RANDOM_STATE,
    SAVED_MODELS_DIR,
    RESULTS_FILE,
)
import pickle
from models.metrics import mse_V
from models.data import train_test_val_df_no_dirs

JOBS = -1


def data(sentiment_desc=False):
    X_train, X_test, X_validation, y_train, y_test, y_validation = (
        train_test_val_df_no_dirs(sentiment_desc)
    )
    X_train = X_train[:, FEATURE_STARTING_INDEX:].astype(np.float64)
    X_test = X_test[:, FEATURE_STARTING_INDEX:].astype(np.float64)
    X_validation = X_validation[:, FEATURE_STARTING_INDEX:].astype(np.float64)
    return X_train, X_test, X_validation, y_train, y_test, y_validation


def rf_test_depth(X_train, X_test, y_train, y_test, results_fname):
    with open(results_fname, "w") as F:
        F.write("Model,MSE,depth\n")
    for depth in [
        2,
        5,
        9,
        10,
        11,
        12,
        13,
        14,
        15,
        16,
        17,
        21,
        25,
        27,
        29,
        32,
        50,
        72,
        100,
        150,
        250,
        None,
    ]:
        model = RandomForestRegressor(
            max_depth=depth, random_state=RANDOM_STATE, verbose=1, n_jobs=JOBS
        )
        model.fit(X_train, y_train)
        Y_hat = model.predict(X_test)
        E = y_test - Y_hat
        MSE = mse_V(E)
        with open(results_fname, "a") as F:
            F.write(f"RandomForestRegressor,{MSE},{depth}\n")
        print(f"RandomForestRegressor,{MSE},{depth}")


def train_rf(sentiment_desc=False, depth=15):
    X_train, _, X_validation, y_train, _, y_validation = data(sentiment_desc)
    model = RandomForestRegressor(
        max_depth=depth, random_state=RANDOM_STATE, verbose=1, n_jobs=JOBS
    )
    MSE, r2 = _train_model(model, X_train, X_validation, y_train, y_validation)
    print(f"RF MSE: {round(MSE, 2)}, R2: {round(r2, 2)}")
    if sentiment_desc:
        fname = f"{SAVED_MODELS_DIR}/rf_sentiment.pkl"
    else:
        fname = f"{SAVED_MODELS_DIR}/rf.pkl"
    with open(fname, "wb") as F:
        pickle.dump(model, F)
    return model


def train_svr(sentiment_desc=False):
    X_train, _, X_validation, y_train, _, y_validation = data(sentiment_desc)
    degree = 3
    C = 1.0
    epsilon = 0.1
    model = SVR(
        kernel="rbf", degree=degree, C=C, epsilon=epsilon, verbose=True, cache_size=4096
    )
    MSE, r2 = _train_model(model, X_train, X_validation, y_train, y_validation)
    print(f"SVR MSE: {round(MSE, 2)}, R2: {round(r2, 2)}")
    if sentiment_desc:
        fname = f"{SAVED_MODELS_DIR}/svr_sentiment.pkl"
    else:
        fname = f"{SAVED_MODELS_DIR}/svr.pkl"
    with open(fname, "wb") as F:
        pickle.dump(model, F)
    return model


def test(model_type, sentiment_desc=False):
    if model_type == "rf":
        model_name = "Random Forest"
    elif model_type == "svr":
        model_name = "SVR"
    else:
        raise ValueError("Model type must be either rf (Random Forest) or svr (SVR)")
    _, X_test, _, _, y_test, _ = data(sentiment_desc)
    if sentiment_desc:
        fname = f"{SAVED_MODELS_DIR}/{model_type}_sentiment.pkl"
    else:
        fname = f"{SAVED_MODELS_DIR}/{model_type}.pkl"
    with open(fname, "rb") as F:
        model = pickle.load(F)
    mse, r2 = _test_model(model, X_test, y_test)
    print(f"Results for {model_name}: MSE: {mse}, R2: {r2}")
    if sentiment_desc:
        data_type = "SentimentDesc"
    else:
        data_type = "NoDir"
    with open(RESULTS_FILE, "a") as F:
        F.write(f"{data_type},{model_name.replace(" ", "")},{mse},{r2}\n")


def _train_model(model, X_train, X_validation, y_train, y_validation):
    model.fit(X_train, y_train)
    return _test_model(model, X_validation, y_validation)


def _test_model(model, X_test, y_test):
    Y_hat = model.predict(X_test)
    E = y_test - Y_hat
    MSE = mse_V(E)
    r2 = model.score(X_test, y_test)
    return MSE, r2


if __name__ == "__main__":
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument(
        "--train",
        action="store_true",
        help="Train the specified model(s). If none specified, both Random Forest and SVR are trained",
    )
    arg_parser.add_argument(
        "--test",
        action="store_true",
        help="Test the specified model(s). If none specified, both Random Forest and SVR are trained",
    )
    arg_parser.add_argument(
        "--rf",
        action="store_true",
        help="Perform specified action only on the Random Forest model",
    )
    arg_parser.add_argument(
        "--svr",
        action="store_true",
        help="Perform specified action only on the SVR model",
    )
    arg_parser.add_argument(
        "-s",
        "--sentiment",
        action="store_true",
        help="Use the dataframe with sentiment analysis data on the description.",
    )
    args = arg_parser.parse_args()
    if not any([args.train, args.test]):
        arg_parser.print_help()
    if args.sentiment:
        sentiment_data = True
    else:
        sentiment_data = False
    run_rf = False
    run_svr = False
    if not any([args.rf, args.svr]):
        run_rf = True
        run_svr = True
    if args.rf:
        run_rf = True
    if args.svr:
        run_svr = True
    if args.train:
        if run_rf:
            train_rf(sentiment_data)
        if run_svr:
            train_svr(sentiment_data)
    if args.test:
        p = pathlib.Path(RESULTS_FILE)
        if not p.exists():
            with open(RESULTS_FILE, "w") as F:
                F.write("Data,Model,MSE,R2,depth\n")
        if run_rf:
            test("rf", sentiment_data)
        if run_svr:
            test("svr", sentiment_data)
