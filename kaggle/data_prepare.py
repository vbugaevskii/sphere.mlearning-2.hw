import os
import numpy as np
import pandas as pd

from scipy.misc import imread, imresize, imrotate

H_IMAGE, W_IMAGE = 139, 139

IMAGES_DIR = './data/roof_images/'
IMAGES_DIR_COMPRESSED = './data/'


def crop_image(img):
    h, w, _ = img.shape
    if h < w:
        size_ = (H_IMAGE, H_IMAGE * w / h)
    else:
        size_ = (H_IMAGE * h / w, H_IMAGE)
    img = imresize(img, size_, )

    h, w, _ = img.shape
    img = img[(h - H_IMAGE) // 2: (h + H_IMAGE) // 2,
          (w - H_IMAGE) // 2: (w + H_IMAGE) // 2]

    return img


def augmentation(img, y):
    images, labels = [], []
    
    for alpha in range(0, 360, 90):
        images.append(imrotate(img, alpha))
        if y in [1, 2] and alpha % 180 > 0:
            labels.append(3 - y)
        else:
            labels.append(y)
            
    return images, labels


def write_images_to_file(image_ids, image_labels, category='test'):
    category = category.lower()
    
    if category not in ['train', 'test']:
        raise NameError('Category should be either TRAIN or TEST')
    
    print 'Compressing ' + category.upper() + ' IMAGES...'

    if not os.path.exists(IMAGES_DIR_COMPRESSED):
        os.makedirs(IMAGES_DIR_COMPRESSED)

    images_path = IMAGES_DIR_COMPRESSED + category + '-images.part{}.ubyte'
    labels_path = IMAGES_DIR_COMPRESSED + category + '-labels.part{}.ubyte'

    labels_all, images_all = [], []
    n_parts = 0

    for image_id, y in zip(image_ids, image_labels):
        image = imread(IMAGES_DIR + str(image_id) + '.jpg')
        image = crop_image(image)
        if category == 'train':
            images, labels = augmentation(image, y)
            labels_all.extend(labels)
            images_all.extend(images)
        else:
            labels_all.append(y)
            images_all.append(image)

        if len(images_all) == 3000:
            with open(images_path.format(n_parts), 'wb') as f_name:
                images_all = np.asarray(images_all, dtype=float)
                images_all.tofile(f_name)

            with open(labels_path.format(n_parts), 'wb') as f_name:
                labels_all = np.asarray(labels_all, dtype=int)
                labels_all.tofile(f_name)

            labels_all, images_all = [], []
            n_parts += 1

            print '{} part is compressed!'.format(n_parts)

    if len(images_all) > 0:
        with open(images_path.format(n_parts), 'wb') as f_name:
            images_all = np.asarray(images_all, dtype=float)
            images_all.tofile(f_name)

        with open(labels_path.format(n_parts), 'wb') as f_name:
            labels_all = np.asarray(labels_all, dtype=int)
            labels_all.tofile(f_name)

        labels_all, images_all = [], []
        n_parts += 1

        print '{} part is compressed!'.format(n_parts)

    print category.upper() + ' IMAGES are compressed!\n'


def read_images_from_file(file_name, n_images=None):
    with open(file_name, 'rb') as f_name:
        count  = n_images * H_IMAGE * W_IMAGE * 3 if n_images else -1
        images = np.fromfile(f_name, count=count)
    return images.reshape(-1, H_IMAGE, W_IMAGE, 3)
    
    
def read_labels_from_file(file_name, n_labels=None):
    with open(file_name, 'rb') as f_name:
        count  = n_labels if n_labels else -1
        images = np.fromfile(f_name, count=count, dtype=int)
    return images


if __name__ == "__main__":
    df_train = pd.read_csv('./data/id_train.csv', sep=',')
    df_test = pd.read_csv('./data/sample_submission4.csv', sep=',')

    write_images_to_file(df_train['Id'], df_train['label'], 'train')
    write_images_to_file(df_test['Id'], df_test['label'], 'test')
