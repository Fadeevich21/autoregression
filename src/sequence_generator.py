import random


# p - номер предсказываемого элемента (после i)
# m - размерность вектора факторов
# n - число измеряемых точек (для оценки качества авторегрессии)
# d - среднеквадратичное отклонение выдачи от идеальной
def xi(p, m, n, d=0.5):  # выдает последовательность, предшествующую отсчету, и сам отсчет
    i = 2 * m + p - 2
    N = i + p + n
    alpha = m * (random.random() - 0.5)
    beta = [random.random() - 0.5 for _ in range(m)]
    seq = [round(random.random() - 0.5, 2) for _ in range(m)]
    for j in range(m, N):
        seqm = seq[j - m:j]
        s = sum(bi * si for bi, si in zip(beta, seqm))
        gauss_distribution = random.gauss(s + alpha, d)
        seq.append(round(gauss_distribution, 2))

    for l in range(i, N):
        yield seq[l - i:l], seq[l]
