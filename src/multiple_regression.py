import sys

from src.regression import Regression
from src.fmath import *


class MultipleRegression(Regression):
    def __init__(self):
        super().__init__()

    def predict(self, alpha, beta, x):
        return sum(x_i * beta_i for x_i, beta_i in zip(x, beta)) + alpha

    def execute(self, X, y):
        try:
            n = len(y)
            M, b = [], []
            M.append([sum(x) for x in X] + [n])
            b.append(sum(y))
            for _, xl in enumerate(X):
                M.append([dot(x, xl) for x in X] + [sum(xl)])
                b.append(dot(y, xl))
            beta = gauss(M, b)
            return beta[-1], beta[:-1]
        except (Exception,):
            sys.exit(-1)
