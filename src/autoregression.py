from src.fmath import *


def autoregression(regression, sequence: list, m, p):
    def get_coefficients():
        x, y = [], []
        for i in range(m):
            x.append(sequence[i:i + m + 1])
            y.append(sequence[i + m + p - 1])
        alpha, beta = regression.execute(x, y)

        return alpha, beta

    alpha, beta = get_coefficients()
    x = sequence[:-m]
    y = regression.predict(alpha, beta, x)

    return x, y
