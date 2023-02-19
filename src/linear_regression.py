from src.regression import Regression
from src.fmath import *


class LinearRegression(Regression):
    def __init__(self):
        super().__init__()

    def predict(self, alpha, beta, x):
        return beta * x + alpha

    def execute(self, x, y):
        beta = (corr(x, y) * standard_deviation(y) / standard_deviation(x))
        alpha = mean(y) - beta * mean(x)
        return alpha, beta
