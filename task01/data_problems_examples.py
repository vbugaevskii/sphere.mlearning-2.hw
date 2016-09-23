import numpy as np
import matplotlib.pyplot as plt

def generate_problem(n_size=25, type='xor'):
    operation = {
        'or':  np.logical_or,
        'and': np.logical_and,
        'xor': np.logical_xor,
        'sum1': np.sum,
        'sum2': np.sum
    }

    X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]] * n_size, dtype='float')
    if type == 'sum1':
        Y = operation[type](X, axis=1)
    elif type == 'sum2':
        tmp = np.array([[0, 0], [1, 1], [1, 0], [0, 1]] * n_size, dtype='float')
        Y = operation[type](tmp, axis=1)
    else:
        Y = operation[type](X[:, 0], X[:, 1]).astype(int)

    X += (np.random.random_sample((4 * n_size, 2)) - 0.5) * 0.5
    permutation = np.random.permutation(4 * n_size)
    return X[permutation], Y[permutation]

if __name__ == '__main__':
    X, Y = generate_problem(type='sum2')
    plt.scatter(X[:, 0], X[:, 1], c=Y)
    plt.show()
