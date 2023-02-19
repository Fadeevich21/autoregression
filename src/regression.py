from abc import ABC, abstractmethod
from .fmath import *


class Regression(ABC):
    def __init__(self):
        pass

    @abstractmethod
    def predict(self, alpha, beta, x):
        pass

    def error(self, alpha, beta, x, y):
        return y - self.predict(alpha, beta, x)

    @abstractmethod
    def execute(self, x, y):
        pass

    def __total_sum_of_squares(self, x):
        return sum(xm_i ** 2 for xm_i in de_mean(x))

    def __sum_of_squared_errors(self, alpha, beta, x, y):
        return sum(self.error(alpha, beta, x_i, y_i) ** 2 for x_i, y_i in zip(x, y))

    def r_squared(self, alpha, beta, x, y):
        return 1 - self.__sum_of_squared_errors(alpha, beta, x, y) / self.__total_sum_of_squares(y)
