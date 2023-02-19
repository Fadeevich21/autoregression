from src.multiple_regression import MultipleRegression
from src.sequence_generator import xi
from src.autoregression import autoregression


number_variant = 3
number_known_values = 2
number_of_predictions = 10

if __name__ == "__main__":
    regression = MultipleRegression()
    generator = xi(number_variant, number_known_values, number_of_predictions)

    sequence, predicted = next(generator)
    x, y, alpha, beta = autoregression(regression, sequence, predicted, number_known_values, number_of_predictions)

    t = [regression.predict(alpha, beta, x_i) for x_i in x]
    r = regression.r_squared(alpha, beta, x, y)
