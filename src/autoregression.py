from src.fmath import *


def autoregression(regression, sequence: list, m, p):
    def get_coefficients():
        x, y = [], []
        for i in range(m):
            x.append(sequence[i:i + m])
            y.append(sequence[i + m + p - 1])
        return regression.execute(transpose_matrix(x), y)

    alpha, beta = get_coefficients()
    x = sequence[-m:]
    y = regression.predict(alpha, beta, x)

    return x, y
