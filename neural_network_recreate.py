import numpy as np
import random
from abc import ABCMeta, abstractmethod


class NeuralLayer:
    __metaclass__ = ABCMeta

    def __init__(self, n_neurons, bias=True):
        self.n_neurons = n_neurons
        self.n_objects = None
        self.bias = bias

    @abstractmethod
    def _activation(self, values_in):
        pass

    def forward(self, values_in, weights):
        self.values_in = np.asarray(np.asmatrix(values_in) * weights)
        self.values_out = self._activation(self.values_in)

    @abstractmethod
    def _activation_derivative(self, values_in, values_out):
        pass

    def backward(self, errors_in, weights):
        derivative = self._activation_derivative(self.values_in, self.values_out)
        if errors_in is None:
            self.deltas = derivative * np.asarray(weights)
            self.deltas = np.c_[self.deltas, np.zeros(self.deltas.shape[0])]
            self.derivatives = None
        else:
            bias = np.ones(self.n_objects) if self.bias else np.zeros(self.n_objects)
            self.deltas = np.c_[derivative, bias] * np.asarray(np.asmatrix(errors_in) * np.asmatrix(weights).T)
            values = np.c_[self.values_out, bias]
            self.derivatives = np.asarray([
                np.asarray(np.asmatrix(values[i]).T * np.asmatrix(errors_in[i]))
                for i in range(self.n_objects)
            ])


class SigmoidLayer(NeuralLayer):
    def _activation(self, values_in):
        return 1.0 / (1.0 + np.exp(-values_in))

    def _activation_derivative(self, values_in, values_out):
        return values_out * (1.0 - values_out)


class IdentityLayer(NeuralLayer):
    def _activation(self, values_in):
        return values_in

    def _activation_derivative(self, values_in, values_out):
        return np.ones(values_out.shape)


class SoftmaxLayer(NeuralLayer):
    def __init__(self, n_neurons):
        NeuralLayer.__init__(self, n_neurons, False)

    def _activation(self, values_in):
        res = np.exp(values_in)
        return res / np.sum(res, axis=1).reshape(res.shape[0], 1)

    def _activation_derivative(self, values_in, values_out):
        return values_out * (1.0 - values_out)


class InputLayer(NeuralLayer):
    def __init__(self, X, bias):
        NeuralLayer.__init__(self, X.shape[1], bias)
        self.n_objects = X.shape[0]
        self.values_in = None
        self.values_out = X

    def _activation(self, values_in):
        return values_in

    def _activation_derivative(self, values_in, values_out):
        return np.ones(values_out.shape)


class NeuralNetwork:
    def __init__(self, layers, input_bias=True,
                 loss_function='MSE', learning_rate=0.5,
                 epsilon=0.05):
        self.layers = [None] + layers
        self.learning_rate = learning_rate
        self.loss_function = loss_function
        self.input_bias = input_bias
        self.epsilon = epsilon

    def __criteria(self, type):
        variants = {
            'MSE': lambda y, t: np.mean(np.sum(np.square(y - t), axis=1)),
            'NLL': lambda y, t: np.mean(-np.sum(t * np.log(y), axis=1))
        }
        return variants[type]

    def __criteria_derivative(self, type):
        variants = {
            'MSE': lambda y, t: y - t,
            'NLL': lambda y, t: y - t
        }
        return variants[type]

    def __forward_step(self):
        for idx, w in enumerate(self.weights):
            n_objects = self.layers[idx].n_objects
            bias = np.ones(n_objects) if self.layers[idx].bias else np.zeros(n_objects)
            values_in = np.c_[self.layers[idx].values_out, bias]
            self.layers[idx + 1].forward(values_in, w)
        return self.layers[-1].values_out

    def __backward_step(self):
        n_layers = len(self.layers)
        for idx in range(n_layers - 1, -1, -1):
            if idx == n_layers - 1:
                func = self.__criteria_derivative(self.loss_function)
                args = (None, func(self.layers[idx].values_out, self.y))
            else:
                args = (self.layers[idx + 1].deltas[:, :-1], self.weights[idx])
            self.layers[idx].backward(*args)

    def __update_weights(self):
        for idx, layer in enumerate(self.layers[:-1]):
            self.weights[idx] -= self.learning_rate * np.mean(layer.derivatives, axis=0)

    def fit(self, X, Y, n_epoch=5, batch_size=25, test_size=0.1):
        self.weights = []
        for idx in range(len(self.layers) - 1):
            M = (self.layers[idx].n_neurons if idx else X.shape[1]) + 1
            N = self.layers[idx + 1].n_neurons
            self.weights.append(np.random.rand(M, N) - 0.5)

        shuffle = random.sample(range(X.shape[0]), X.shape[0])
        border = X.shape[0] * test_size

        X = X[shuffle]
        Y = Y[shuffle]

        X_train = X[border:]
        Y_train = Y[border:]
        X_test = X[:border]
        Y_test = Y[:border]

        self.error = []
        for epoch in range(n_epoch):
            self.__epoch(X_train, Y_train, batch_size)
            error_ = self.__criteria(self.loss_function)(self.predict(X_test), Y_test)
            self.error.append(error_)

    def predict(self, X, batch_size=25):
        sample = range(X.shape[0])
        batches = [sample[i:i + batch_size] for i in range(0, X.shape[0], batch_size)]
        predicted = None
        for batch in batches:
            self.__batch_init(X, None, batch)

            predicted_batch = self.__forward_step()
            if predicted is None:
                predicted = predicted_batch
            else:
                predicted = np.r_[predicted, predicted_batch]
        return predicted

    def __batch_init(self, X, Y, batch):
        self.layers[0] = InputLayer(X[batch] if len(batch) > 1 else X[batch].reshape(1, X.shape[1]), self.input_bias)

        for layer in self.layers:
            layer.n_objects = len(batch)

        if Y is not None:
            self.y = Y[batch] if len(batch) > 1 else Y[batch].reshape(1, Y.shape[1])
        else:
            self.y = None

    def train_on_batch(self, X, Y, batch):
        self.__batch_init(X, Y, batch)
        self.__forward_step()
        self.__backward_step()
        self.__update_weights()

    def __epoch(self, X, Y, batch_size):
        rand_sample = random.sample(range(X.shape[0]), X.shape[0])
        batches = [rand_sample[i:i + batch_size] for i in range(0, len(rand_sample), batch_size)]
        for batch in batches:
            self.train_on_batch(X, Y, batch)


if __name__ == '__main__':
    multip = 100
    dfX = np.array([[.05, .10]] * multip)
    dfY = np.array([[.01, .99]] * multip)

    nn = NeuralNetwork(layers=[
        SigmoidLayer(2, bias=True),
        SigmoidLayer(2, bias=False),
    ], input_bias=True, learning_rate=.5)

    nn.fit(dfX, dfY, n_epoch=100, batch_size=10)
    r = nn.predict(dfX, batch_size=10)
    print r