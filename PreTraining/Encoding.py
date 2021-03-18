import numpy as np


def one_hot(x):
    x_set = list(set(x))
    x_ = np.zeros((len(x), len(x_set)), dtype=int)

    x_[(tuple(range(len(x))), tuple(x_set.index(x[i]) for i in range(len(x))))] = 1

    return x_, x_set


def target_encoding(x, y):
    x, y = np.array(x), np.array(y)

    x_ = np.zeros(len(x))

    for i in range(len(x)):
        x_[i] = np.sum(y * (x == x[i])) / np.sum(x == x[i])

    return x_


def leave_one_out(x, y):
    x, y = np.array(x), np.array(y)
    sum_y = np.sum(y)
    n = len(x)
    x_ = np.zeros(len(x))

    for i in range(len(x)):
        x_[i] = (sum_y - y[i]) / (n - 1)

    return x_


if __name__ == '__main__':
    x = [1, 1, 1, 2, 3, 4]
    y = [1, 0, 1, 0, 1, 0]

    # print(one_hot(x))
    # print(target_encoding(x, y))
    print(leave_one_out(x, y))
