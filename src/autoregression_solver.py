from src.sequence_generator import xi
from src.autoregression import autoregression
from src.regression import Regression


class AutoregressionSolver:

    __variant: int = None
    __number_known_values: int = None
    __number_of_predictions: int = None
    __regression: Regression = None

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

    def __get_generator(self):
        return xi(self.__variant, self.__number_known_values, self.__number_of_predictions)

    def __autoregression(self, sequence, predicted):
        return autoregression(self.__regression, sequence, predicted, self.__number_known_values,
                              self.__number_of_predictions)

    def execute(self) -> int | float:
        generator = self.__get_generator()
        sequence, predicted = next(generator)

        x, y, alpha, beta = self.__autoregression(sequence, predicted)
        quality_of_autoregression = self.__regression.r_squared(alpha, beta, x, y)

        return quality_of_autoregression
