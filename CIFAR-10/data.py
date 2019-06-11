import os
import numpy as np
import random
import matplotlib

if os.environ.get('DISPLAY','') == '':
    print('no display found. Using non-interactive Agg backend')
    matplotlib.use('Agg')

from skimage.io import imread
from skimage.transform import resize
#from tensorflow.python.keras._impl.keras.datasets.cifar10 import load_data
from keras.datasets import cifar10
import tensorflow as tf

tf.set_random_seed(9)
np.random.seed(9)

def read_CIFAR10_subset():
    """
    Load the CIFAR-10 data subset from keras helper module
    and perform preprocessing for training ResNet.
    :return: X_set: np.ndarray, shape: (N, H, W, C).
             y_set: np.ndarray, shape: (N, num_channels) or (N,).
    """

    # Download CIFAR-10 data and load data
    (x_train, y_train), (x_test, y_test) = cifar10.load_data()

    y_train_oh = np.zeros((len(y_train), 10), dtype=np.uint8)
    for i in range(len(y_train)):
        y_train_oh[i, y_train[i]] = 1
    y_train_one_hot = y_train_oh

    y_test_oh = np.zeros((len(y_test), 10), dtype=np.uint8)
    for i in range(len(y_test)):
        y_test_oh[i, y_test[i]] = 1
    y_test_one_hot = y_test_oh
    
    '''
    x_train = x_train/127.5
    x_test = x_test/127.5
    x_train = x_train -1 
    x_test = x_test -1 
    '''
    x_train = x_train/255.0
    x_test = x_test/255.0
    cifar_mean = np.array([0.4914, 0.4822, 0.4465])
    cifar_std = np.array([0.2470, 0.2435, 0.2616])

    for i in range(len(x_train)):
        x_train[i] -= cifar_mean
        x_train[i] /= cifar_std

    for j in range(len(x_test)):
        x_test[j] -= cifar_mean
        x_test[j] /= cifar_std
    
    print('x_train shape : ', x_train.shape, end='\n')
    print('x_test shape : ', x_test.shape, end='\n')
    print('y_train_one_hot shape : ', y_train_one_hot.shape, end='\n')
    print('y_test_one_hot shape : ', y_test_one_hot.shape, end='\n')
    print('\nDone')

    return x_train, x_test, y_train_one_hot, y_test_one_hot

def cifar_augment(images):
    """
    Perform data augmentation from cifar images.
    :param images: np.ndarray, shape: (N, C, H, W).
    :return: np.ndarray, shape: (N, C, H, W).
    """
    augmented_images = []
    for image in images:    # image.shape: (C, H, W)
	# horizontal flip with 0.5 probability
        reflection = bool(np.random.randint(2))
        if reflection:
             image = np.fliplr(image)

        # random cropping with padding
        image_pad = np.pad(image, ((4,4), (4,4), (0,0)), mode='constant')

        crop_x1 = random.randint(0, 8)
        crop_x2 = crop_x1 + 32
        crop_y1 = random.randint(0, 8)
        crop_y2 = crop_y1 + 32

        image_crop = image_pad[crop_x1:crop_x2, crop_y1:crop_y2]

        augmented_images.append(image_crop)

    return np.stack(augmented_images)    # shape: (N, C, H, W)

class DataSet(object):
    def __init__(self, images, labels=None):
        """
        Construct a new DataSet object.
        :param images: np.ndarray, shape: (N, H, W, C).
        :param labels: np.ndarray, shape: (N, num_classes) or (N,).
        """
        if labels is not None:
            assert images.shape[0] == labels.shape[0], (
                'Number of examples mismatch, between images and labels.'
            )
        self._num_examples = images.shape[0]
        self._images = images
        self._labels = labels    # NOTE: this can be None, if not given.
        self._indices = np.arange(self._num_examples, dtype=np.uint)    # image/label indices(can be permuted)
        self._reset()

    def _reset(self):
        """Reset some variables."""
        self._epochs_completed = 0
        self._index_in_epoch = 0

    @property
    def images(self):
        return self._images

    @property
    def labels(self):
        return self._labels

    @property
    def num_examples(self):
        return self._num_examples

    def next_batch(self, batch_size, shuffle=True, augment=True, is_train=True,
                   fake_data=False):
        """
        Return the next `batch_size` examples from this dataset.
        :param batch_size: int, size of a single batch.
        :param shuffle: bool, whether to shuffle the whole set while sampling a batch.
        :param augment: bool, whether to perform data augmentation while sampling a batch.
        :param is_train: bool, current phase for sampling.
        :param fake_data: bool, whether to generate fake data (for debugging).
        :return: batch_images: np.ndarray, shape: (N, h, w, C) or (N, 10, h, w, C).
                 batch_labels: np.ndarray, shape: (N, num_classes) or (N,).
        """

        start_index = self._index_in_epoch

        # Shuffle the dataset, for the first epoch
        if self._epochs_completed == 0 and start_index == 0 and shuffle:
            np.random.shuffle(self._indices)

        # Go to the next epoch, if current index goes beyond the total number of examples
        if start_index + batch_size > self._num_examples:
            # Increment the number of epochs completed
            self._epochs_completed += 1
            # Get the rest examples in this epoch
            rest_num_examples = self._num_examples - start_index
            indices_rest_part = self._indices[start_index:self._num_examples]

            # Shuffle the dataset, after finishing a single epoch
            if shuffle:
                np.random.shuffle(self._indices)

            # Start the next epoch
            start_index = 0
            self._index_in_epoch = batch_size - rest_num_examples
            end_index = self._index_in_epoch
            indices_new_part = self._indices[start_index:end_index]

            images_rest_part = self.images[indices_rest_part]
            images_new_part = self.images[indices_new_part]
            batch_images = np.concatenate((images_rest_part, images_new_part), axis=0)
            if self.labels is not None:
                labels_rest_part = self.labels[indices_rest_part]
                labels_new_part = self.labels[indices_new_part]
                batch_labels = np.concatenate((labels_rest_part, labels_new_part), axis=0)
            else:
                batch_labels = None
        else:
            self._index_in_epoch += batch_size
            end_index = self._index_in_epoch
            indices = self._indices[start_index:end_index]
            batch_images = self.images[indices]
            if self.labels is not None:
                batch_labels = self.labels[indices]
            else:
                batch_labels = None

        if augment and is_train:
            # Perform data augmentation, for training phase
            batch_images = cifar_augment(batch_images)
        else:
            # Don't perform data augmentation
            batch_images = batch_images

        return batch_images, batch_labels
