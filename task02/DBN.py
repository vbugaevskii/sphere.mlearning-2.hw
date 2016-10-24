#Embedded file name: DBN.py
from RBM import *

class DBN:

    def __init__(self, layers, loss_function):
        if len(layers) < 3:
            raise NameError('Use RBM instead of DBN' if len(layers) == 2 else 'Wrong number of layers!')
        self.layers = layers
        self.loss_function = loss_function

    def fit(self, X, n_epochs = 10, learning_rate = 0.05, n_gibbs_steps = 1, batch_size = 10):
        self.weights = []
        for layer_idx in range(1, len(self.layers)):
            rbm = RBM(layers=[self.layers[layer_idx - 1], self.layers[layer_idx]], loss_function='NLL')
            X_sampled = X
            for i in range(len(self.weights)):
                _, X_sampled = self.layers[i + 1].sample(X_sampled, self.weights[i])

            rbm.fit(X_sampled, n_epochs, learning_rate, n_gibbs_steps, batch_size)
            self.weights.append(rbm.weights)

    def predict(self, X, batch_size = None):
        if batch_size is None:
            batches = [range(X.shape[0])]
        else:
            rand_sample = range(X.shape[0])
            batches = [ rand_sample[i:i + batch_size] for i in range(0, len(rand_sample), batch_size) ]
        predicted = None
        for batch in batches:
            predicted_batch = X[batch]
            for i, w in enumerate(self.weights):
                _, predicted_batch = self.layers[i + 1].sample(predicted_batch, w)

            N = len(self.weights)
            for i, w in enumerate(self.weights[-1::-1]):
                _, predicted_batch = self.layers[N - 1 - i].sample(predicted_batch, w.T)

            predicted = predicted_batch if predicted is None else np.r_[predicted, predicted_batch]

        return predicted
