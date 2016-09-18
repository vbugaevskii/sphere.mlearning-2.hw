import numpy as np
import matplotlib.pyplot as plt

PATH = './data/'

IMAGES_TEST = 't10k-images.idx3-ubyte'
LABELS_TEST = 't10k-labels.idx1-ubyte'

IMAGES_TRAIN = 'train-images.idx3-ubyte'
LABELS_TRAIN = 'train-labels.idx1-ubyte'


def get_hex_from_file(f, b):
    return int(f.read(b).encode('hex'), 16)


def read_label_file(file_name, n_objects=None, offset=0):
    with open(PATH + file_name, 'rb') as f:
        magic = get_hex_from_file(f, 4)
        n_items = get_hex_from_file(f, 4)

        n_size = n_objects if n_objects else n_items - offset
        if offset >= n_items:
            raise NameError('Out of bounds!')
        elif offset + n_size > n_items:
            n_size = n_items - offset

        f.seek(offset, 1)
        labels = [get_hex_from_file(f, 1) for i in range(n_size)]
    return labels


def read_image_file(file_name, n_objects=None, offset=0):
    with open(PATH + file_name, 'rb') as f:
        magic = get_hex_from_file(f, 4)
        n_items = get_hex_from_file(f, 4)

        n_size = n_objects if n_objects else n_items - offset
        if offset >= n_items:
            raise NameError('Out of bounds!')
        elif offset + n_size > n_items:
            n_size = n_items - offset

        n_rows = get_hex_from_file(f, 4)
        n_cols = get_hex_from_file(f, 4)

        f.seek(offset * n_cols * n_rows, 1)
        images = [np.array([[get_hex_from_file(f, 1) for col in range(n_cols)] for row in range(n_rows)])
                  for i in range(n_size)]

    return images


if __name__ == '__main__':
    batch_size = 30
    offset = 10
    image_index = 15

    labels = read_label_file(LABELS_TRAIN, batch_size, offset)
    images = read_image_file(IMAGES_TRAIN, batch_size, offset)

    plt.imshow(images[image_index], cmap='gray')
    plt.title('label = {}'.format(labels[image_index]))
    plt.show()