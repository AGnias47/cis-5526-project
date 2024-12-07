import numpy as np
from constants import FEATURE_STARTING_INDEX, PRECISION
from sklearn.metrics import r2_score

def mse_V(E):
    return float((1 / E.shape[0]) * np.matmul(E.transpose(), E))


def mse_r2_V(X_test, y_test, W):
    X_test = X_test[:, FEATURE_STARTING_INDEX:].astype(np.float64)
    Y_hat_test = np.matmul(X_test, W)
    E_test = y_test - Y_hat_test
    return round(mse_V(E_test), PRECISION), round(
        r2_score(y_test, Y_hat_test), PRECISION
    )
