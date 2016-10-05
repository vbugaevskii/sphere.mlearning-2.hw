# -*- coding: utf-8 -*-

import numpy as np
import random
from abc import ABCMeta, abstractmethod


def sigmoid(x):
    return 1. / (1. + np.exp(-x))


class NeuronLayer:
    __metaclass__ = ABCMeta

    def __init__(self, n_neurons):
        self.n_neurons = n_neurons
        self.bias = np.zeros((1, n_neurons))

    @abstractmethod
    def _sample_values(self, values, weights):
        pass

    def sample(self, values, weights, update=False):
        values_prob, values = self._sample_values(values, weights)
        if update:
            self.values_prob, self.values = values_prob, values
        return values_prob, values

    @abstractmethod
    def _prob_conditional(self, values, weights):
        pass

    def __getattr__(self, attr):
        if attr == 'shape':
            return self.values.shape
        else:
            raise Exception


class BernoulliLayer(NeuronLayer):
    def _prob_conditional(self, values, weights):
        return sigmoid(self.bias + np.dot(values, weights))

    def _sample_values(self, values, weights):
        values_prob = self._prob_conditional(values, weights)
        values = np.random.binomial(n=1, p=values_prob, size=values_prob.shape)
        return values_prob, values


class GaussianLayer(NeuronLayer):
    # TODO: implement later
    pass


class RBM:
    def __init__(self, layers, loss_function):
        if len(layers) != 2:
            raise Exception

        self.layers = layers
        self.loss_function = loss_function

    def __sample_hidden(self):
        return self.layers[1].sample(self.layers[0].values, self.weights, update=True)

    def __sample_visible(self):
        return self.layers[0].sample(self.layers[1].values, self.weights.T, update=True)

    def __gibbs_sampling_step(self):
        self.__sample_hidden()
        return self.__sample_visible()

    def __gibbs_sampling(self, n_steps):
        for k in range(n_steps):
            visible_proba, visible = self.__gibbs_sampling_step()
        return visible_proba, visible

    def __gradient_parameters_per_object(self, positive_visible, negative_visible):
        _, positive_hidden = self.layers[1].sample(positive_visible, self.weights)
        negative_hidden, _ = self.layers[1].sample(negative_visible, self.weights)

        delta_weights = np.dot(positive_visible.T, positive_hidden) - np.dot(negative_visible.T, negative_hidden)
        delta_bias_visible = positive_visible - negative_visible
        delta_bias_hidden = positive_hidden - negative_hidden

        return delta_weights, delta_bias_visible, delta_bias_hidden

    def __update_parameters(self, positive_visible, negative_visible):
        delta_weights, delta_bias_hidden, delta_bias_visible = [], [], []

        for obj in range(positive_visible.shape[0]):
            pv = positive_visible[obj].reshape(1, -1)
            nv = negative_visible[obj].reshape(1, -1)
            deltas = self.__gradient_parameters_per_object(pv, nv)

            delta_weights.append(deltas[0])
            delta_bias_visible.append(deltas[1])
            delta_bias_hidden.append(deltas[2])

        self.weights += self.learning_rate * np.mean(delta_weights, axis=0)
        self.layers[0].bias += self.learning_rate * np.mean(delta_bias_visible, axis=0)
        self.layers[1].bias += self.learning_rate * np.mean(delta_bias_hidden, axis=0)

    def __train_on_batch(self, batch):
        self.layers[0].values = batch
        self.__gibbs_sampling(self.n_gibbs_steps)
        self.__update_parameters(batch, self.layers[0].values)

    def __epoch(self, X, batch_size):
        if batch_size is None:
            batches = [range(X.shape[0])]
        else:
            rand_sample = random.sample(range(X.shape[0]), X.shape[0])
            batches = [rand_sample[i:i + batch_size] for i in range(0, len(rand_sample), batch_size)]

        for batch in batches:
            self.__train_on_batch(X[batch])

    def fit(self, X, n_epochs=10, learning_rate=.05, n_gibbs_steps=1, batch_size=10, test_size=0):
        self.n_visible = X.shape[1]
        self.learning_rate = learning_rate
        self.n_gibbs_steps = n_gibbs_steps
        self.n_epochs = n_epochs

        self._setup_parameters()

        sample = random.sample(range(X.shape[0]), X.shape[0])
        X = X[sample]
        border = X.shape[0] * test_size
        X_test, X_train = X[:border], X[border:]

        self.error_train, self.error_test = [], []
        for epoch in range(n_epochs):
            # if (epoch + 1) % 20 == 0:
            #     self.learning_rate *= .1

            self.__epoch(X_train, batch_size)
            self.error_train.append(self.criteria(X_train))

            if test_size > 0.0:
                self.error_test.append(self.criteria(X_test))

            print '\r', 'epoch = {};'.format(epoch), 'criteria = {};'.format(self.error_train[-1]), \
                'learning_rate = {}'.format(self.learning_rate),
        print ''

    def predict_proba(self, X, batch_size=None):
        if batch_size is None:
            batches = [range(X.shape[0])]
        else:
            rand_sample = random.sample(range(X.shape[0]), X.shape[0])
            batches = [rand_sample[i:i + batch_size] for i in range(0, len(rand_sample), batch_size)]

        proba = []
        for batch in batches:
            self.layers[0].values = X[batch]
            proba.append(self.__gibbs_sampling(self.n_gibbs_steps)[0])

        return np.mean(np.asarray(proba).reshape(-1, self.layers[0].n_neurons), axis=0)

    def predict(self, X, batch_size=None):
        if batch_size is None:
            batches = [range(X.shape[0])]
        else:
            rand_sample = random.sample(range(X.shape[0]), X.shape[0])
            batches = [rand_sample[i:i + batch_size] for i in range(0, len(rand_sample), batch_size)]

        predicted = None
        for batch in batches:
            self.layers[0].values = X[batch]
            predicted_batch = self.__gibbs_sampling(self.n_gibbs_steps)[1]
            predicted = predicted_batch if predicted is None else np.r_[predicted, predicted_batch]

        return predicted

    def _setup_parameters(self):
        n_visible = self.layers[0].n_neurons
        n_hidden = self.layers[1].n_neurons

        self.weights = np.random.normal(0, .01, (n_visible, n_hidden))

        print 'Initialisation...'
        print 'weights:\n', self.weights
        print 'bias (visible):\n', self.layers[0].bias
        print 'bias (hidden):\n',  self.layers[1].bias
        print ''

    def __criteria_energy(self, visible):
        def calculate_energy(v, h):
            return - np.dot(v, self.layers[0].bias.T) \
                   - np.dot(h, self.layers[1].bias.T) \
                   - np.dot(np.dot(v, self.weights), h.T)

        _, hidden = self.layers[1].sample(visible, self.weights)
        criteria = [calculate_energy(visible[obj].reshape(1, -1), hidden[obj].reshape(1, -1))
                    for obj in range(visible.shape[0])]
        return np.mean(criteria)

    def __criteria_nll(self, visible):
        self.layers[0].values = visible
        sampled, _ = self.__gibbs_sampling(self.n_gibbs_steps)

        criteria = -np.sum(visible * np.log(sampled) + (1. - visible) * np.log(1. - sampled), axis=1)
        return np.mean(criteria)

    def criteria(self, visible):
        variants = {
            'energy': self.__criteria_energy,
            'NLL': self.__criteria_nll
        }
        return variants[self.loss_function](visible)

if __name__ == '__main__':
    p = .9
    data_X = np.random.binomial(n=1, p=p, size=(100, 2))
    data_Y = np.random.binomial(n=1, p=p, size=(100, 2))

    nn = RBM(layers=[
        BernoulliLayer(data_X.shape[1]),
        BernoulliLayer(2 * data_X.shape[1])
    ], loss_function='NLL')

    nn.fit(data_X, n_epochs=50, learning_rate=.05, n_gibbs_steps=1, batch_size=10)

    print ''
    print 'Final parameters...'
    print 'weights:\n', nn.weights
    print 'bias (visible):\n', nn.layers[0].bias
    print 'bias (hidden):\n',  nn.layers[1].bias

    print ''
    print 'X, Y ~ Bi(1, p); p = {}'.format(p)
    print 'data_X :', np.mean(data_X, axis=0), nn.predict_proba(data_X, batch_size=10)
    print 'data_Y :', np.mean(data_Y, axis=0), nn.predict_proba(data_Y, batch_size=10)