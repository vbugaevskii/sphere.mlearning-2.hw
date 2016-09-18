import numpy as np
import matplotlib.pyplot as plt

def generate_problem(n_size=25, type='xor'):
    operation = {
        'or':  np.logical_or,
        'and': np.logical_and,
        'xor': np.logical_xor
    }

    X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]] * n_size, dtype='float')
    Y = operation[type](X[:, 0], X[:, 1]).astype(int)

    X += (np.random.random_sample((4 * n_size, 2)) - 0.5) * 0.5
    permutation = np.random.permutation(4 * n_size)
    return X[permutation], Y[permutation]

if __name__ == '__main__':
    X, Y = generate_problem()
    plt.scatter(X[:, 0], X[:, 1], c=Y)
    plt.show()
