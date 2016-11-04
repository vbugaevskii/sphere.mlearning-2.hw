from abc import ABCMeta, abstractmethod

import numpy as np


class Optimizer:
    __metaclass__ = ABCMeta

    def __init__(self, eps):
        self.eps = eps

    def derivative(self, grad, p):
        deriv = []
        for i in range(p.shape[0]):
            pl_, pr_ = p.copy(), p.copy()
            pl_[i] = p[i] + self.eps
            pr_[i] = p[i] - self.eps
            deriv_ = (grad(pl_) - grad(pr_)) / (2.0 * self.eps)
            deriv.append(deriv_[i])
        return np.asarray(deriv)

    @abstractmethod
    def step(self, theta, grad, **params):
        pass


class GD(Optimizer):
    def step(self, theta, grad, **params):
        g_ = self.derivative(grad, theta)
        theta_ = theta - params['eta'] * g_
        return theta_, params


class Momentum(Optimizer):
    def step(self, theta, grad, **params):
        g_ = self.derivative(grad, theta)
        v_ = params['gamma'] * params['v'] + params['eta'] * g_
        theta_ = theta - v_

        params['v'] = v_

        return theta_, params


class NAG(Optimizer):
    def step(self, theta, grad, **params):
        g_ = self.derivative(grad, theta - params['gamma'] * params['v'])
        v_ = params['gamma'] * params['v'] + params['eta'] * g_
        theta_ = theta - v_

        params['v'] = v_

        return theta_, params


class Adagrad(Optimizer):
    def step(self, theta, grad, **params):
        g_ = self.derivative(grad, theta)
        for i, e in enumerate(g_ ** 2):
            params['G'][i, i] += e
        theta_ = theta - params['eta'] * g_ / np.sqrt(params['G'].diagonal() + params['eps'])
        return theta_, params


class RMSprop(Optimizer):
    def E(self, g, gamma, e):
        return gamma * e + (1 - gamma) * g ** 2

    def RMS(self, e, eps):
        return np.sqrt(e + eps)

    def step(self, theta, grad, **params):
        if 'e' not in params.keys():
            params['e'] = 0

        g_ = self.derivative(grad, theta)
        e_ = self.E(g_, params['gamma'], params['e'])
        theta_ = theta - params['eta'] * g_ / self.RMS(e_, params['eps'])

        params['e'] = e_

        return theta_, params


class Adadelta(RMSprop):
    def step(self, theta, grad, **params):
        if 'Eg2' not in params.keys():
            params['Eg2'] = 0
        if 'Edtheta2' not in params.keys():
            params['Edtheta2'] = 0

        g_ = self.derivative(grad, theta)
        Eg2_ = self.E(g_, params['gamma'], params['Eg2'])
        dtheta_ = self.RMS(params['Edtheta2'], params['eps']) / self.RMS(Eg2_, params['eps']) * g_
        Edtheta2_ = self.E(dtheta_, params['gamma'], params['Edtheta2'])
        theta_ = theta - dtheta_

        params['Eg2'] = Eg2_
        params['Edtheta2'] = Edtheta2_

        return theta_, params


class Adam(Optimizer):
    def step(self, theta, grad, **params):
        if 't' not in params.keys():
            params['t'] = 1
        g_ = self.derivative(grad, theta)
        m_ = params['beta1'] * params['m'] + (1 - params['beta1']) * g_
        v_ = params['beta2'] * params['v'] + (1 - params['beta2']) * g_ ** 2
        m_hat = m_ / (1 - params['beta1'] ** params['t'])
        v_hat = v_ / (1 - params['beta2'] ** params['t'])
        theta_ = theta - params['eta'] * m_hat / np.sqrt(v_hat + params['eps'])

        params['m'] = m_
        params['v'] = v_
        params['t'] += 1

        return theta_, params


def step(theta, grad, name, state):
    optimizer = {
        'GD':       GD,
        'Momentum': Momentum,
        'NAG':      NAG,
        'Adagrad':  Adagrad,
        'RMSprop':  RMSprop,
        'Adadelta': Adadelta,
        'Adam':     Adam
    }[name](1e-3)

    return optimizer.step(theta, grad, **state)

if __name__ == '__main__':
    J = lambda x: x ** 2

    theta = np.asarray([1], dtype=float)
    params = {
        'gamma': .9,
        'eps': 1e-4
    }

    print theta, params
    for i in range(10):
        theta, params = step(theta=theta, grad=J, name='Adadelta', state=params)
        print theta, params
