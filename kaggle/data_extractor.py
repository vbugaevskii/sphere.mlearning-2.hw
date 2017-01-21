import numpy as np


class DataExtractor:
    def __init__(self, X_data, Y_data=None, batch_size=20):
        if Y_data is not None and len(X_data) != len(Y_data):
            raise TypeError('Data and labels should have same length!')

        self.X_data = X_data.copy()
        self.Y_data = Y_data.copy() if Y_data else None

        self.num_examples = X_data.shape[0]
        self.batch_size = batch_size

    def set_batch_size(self, size):
        self.batch_size = size

    def shuffle(self):
        indexes = np.arange(self.num_examples)
        np.random.shuffle(indexes)

        self.X_data = self.X_data[indexes]
        self.Y_data = self.Y_data[indexes] if self.Y_data else None

    def __iter__(self):
        for i in xrange(0, self.num_examples, self.batch_size):
            yield self.X_data[i:i+self.batch_size], \
                  self.Y_data[i:i+self.batch_size] if self.Y_data else None

if __name__ == '__main__':
    df = DataExtractor(np.arange(0, 100, 1).reshape(50, -1), np.ones(50))
    df.set_batch_size(15)
    df.shuffle()
    for x, y in df:
        print x