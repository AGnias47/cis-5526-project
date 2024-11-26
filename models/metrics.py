import numpy as np


def mse_V(E):
    return float((1 / E.shape[0]) * np.matmul(E.transpose(), E))
