from src.sequence_generator import xi
from src.autoregression import autoregression
from src.regression import Regression
from src.fmath import *


class AutoregressionSolver:

    __variant: int = None
    __number_known_values: int = None
    __number_of_predictions: int = None
    __regression: Regression = None
    __generator = None

    def __init__(self) -> None:
        pass

    def set_variant(self, variant) -> None:
        self.__variant = variant

    def set_number_known_values(self, number_known_values) -> None:
        self.__number_known_values = number_known_values

    def set_number_of_predictions(self, number_of_predictions) -> None:
        self.__number_of_predictions = number_of_predictions

    def set_regression(self, regression) -> None:
        self.__regression = regression()

    def __init_generator(self):
        self.__generator = xi(self.__variant, self.__number_known_values,
                              self.__number_of_predictions)

    def __get_data_generator(self):
        return next(self.__generator)

    def __autoregression(self, sequence):
        return autoregression(self.__regression, sequence, self.__number_known_values,
                              self.__variant)

    def __get_sequence(self):
        sequence, predict = self.__get_data_generator()
        sequence.append(predict)

        return sequence

    def execute(self) -> int | float:
        self.__init_generator()
        x, y = [], []
        for _ in range(self.__number_of_predictions):
            sequence = self.__get_sequence()
            x_i, y_i = self.__autoregression(sequence)
            x.append(x_i)
            y.append(y_i)

        alpha, beta = self.__regression.execute(transpose_matrix(x), y)
        quality_of_autoregression = self.__regression.r_squared(alpha, beta, x, y)
        for i in range(len(x)):
            print(self.__regression.predict(alpha, beta, x[i]))
        print(y)

        return quality_of_autoregression
