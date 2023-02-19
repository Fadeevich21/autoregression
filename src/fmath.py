from math import sqrt
from copy import copy


# Среднее значение
def mean(x):
    return sum(x)/len(x)


# Отклонения от среднего
def de_mean(x):
    return [x_i - mean(x) for x_i in x]


# Дисперсия
def variance(x):
    return mean(list(map(lambda t: t ** 2, x))) - mean(x)**2


# Стандартное отклонение
def standard_deviation(x):
    return sqrt(variance(x))


# Скалярное произведение векторов
def dot(x, y):
    return sum(x_i * y_i for x_i, y_i in zip(x, y))


# Ковариация
def cov(x, y):
    return dot(de_mean(x), de_mean(y)) / len(x)


# Корреляция
def corr(x, y):
    if standard_deviation(x) > 0 and standard_deviation(y) > 0:
        return cov(x,y)/standard_deviation(x)/standard_deviation(y)
    else:
        return 0


def gauss(A, b):  # метод Гаусса решения СЛАУ
    def ni(l, i):  # нормировка списка l на единицу в i-той позиции
        return [lj / l[i] for lj in l]

    def ch_stack(L, i):  # перемещение строки вниз матрицы (на ее место ставится следующая строка)
        L.append(L[i])
        L.pop(i)
        return L

    def corr_row(Gm, Gn, n):  # корректировка списка Gm списком Gn, нормированным на единицу в позиции n
        gml = ni(Gm, n)
        return [gmlk - gnk for gmlk, gnk in zip(gml, Gn)]

    x = []  # инициируем список, который потом станет решением
    n = len(b)  # вычисляем порядок системы
    G = [ai + [bi] for ai, bi in zip(A, b)]  # строим расширенную матрицу системы
    while n > 1:  # в этом цикле нормируем строки на их диагональный элемент
        n -= 1
        if not G[n][n]:
            ch_stack(G, n)
        cGn = copy(G[n])
        G[n] = ni(cGn, n)
        m = n
        while m > 0:  # в этом цикле корректируем все строки, выше той, что только что отнормирована
            m -= 1
            if G[m][n]:
                cGm = copy(G[m])
                G[m] = corr_row(cGm, G[n], n)
    # прямой проход закончен
    x.append(G[0][-1] / G[0][0])  # присваиваем значение первому неизвестному
    for gi in G[1:]:  # последовательно вычисляем все остальные неизвестные
        x.append((gi[-1] - dot(x, gi)) / gi[len(x)])
    return x


def get_column_matrix(matrix: list[list], index_column):
    return [row[index_column] for row in matrix]


def transpose_matrix(matrix):
    transposed_matrix = []
    number_column = len(matrix[0])
    for index_column in range(number_column):
        transposed_matrix.append(get_column_matrix(matrix, index_column))

    return transposed_matrix
