from src.fmath import *


def autoregression(regression, sequence: list, predicted, m, p):
    x = []
    y = []
    sequence.append(predicted)
    for i in range(m):
        x.append(sequence[i:i+m+1])
        y.append(sequence[i+m+p-1])

    sequence.pop(-1)
    alpha, beta = regression.execute(x, y)

    return x, y, alpha, beta
