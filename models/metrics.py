import numpy as np
import torch
from constants import FEATURE_STARTING_INDEX, PRECISION
from sklearn.metrics import r2_score


def mse_V(E):
    return float((1 / E.shape[0]) * np.matmul(E.transpose(), E))


def mse_r2_V(X, y, W):
    X = X[:, FEATURE_STARTING_INDEX:].astype(np.float64)
    Y_hat = np.matmul(X, W)
    E = y - Y_hat
    return round(mse_V(E), PRECISION), round(r2_score(y, Y_hat), PRECISION)


def mse_r2_batched_V(X, y, W, mse, r2):
    X = X[:, FEATURE_STARTING_INDEX:].astype(np.float64)
    Y_hat = np.matmul(X, W)
    mse.update(torch.tensor(Y_hat), torch.tensor(y))
    r2.update(torch.tensor(Y_hat), torch.tensor(y))
    return mse, r2
