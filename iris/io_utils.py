import numpy as np
import tensorflow as tf
from tensorflow.contrib.learn.python.learn.datasets.base import Dataset


NUM_EPOCHS = 2000
BATCH_SIZE = 25
MIN_AFTER_DEQUEUE = 512


def generic_input_fn(attributes: np.ndarray, target: np.ndarray, shuffle: bool, num_epochs: int, batch_size: int):
    limited_attributes = tf.train.limit_epochs(tf.constant(attributes), num_epochs=num_epochs, name='attributes')
    limited_target = tf.train.limit_epochs(tf.constant(target), num_epochs=num_epochs, name='target')

    capacity = MIN_AFTER_DEQUEUE + 2 * batch_size

    batch_args = {
        'tensors': [limited_attributes, limited_target],
        'batch_size': batch_size,
        'capacity': capacity,
        'enqueue_many': True,
        'allow_smaller_final_batch': True
    }
    if shuffle:
        batch_args['min_after_dequeue'] = MIN_AFTER_DEQUEUE
        attributes_batches, target_batches = tf.train.shuffle_batch(**batch_args)
    else:
        attributes_batches, target_batches = tf.train.batch(**batch_args)

    # We treat concatenation of four attributes as one feature
    features = {
        'attributes': attributes_batches
    }

    targets = {
        'targets': target_batches
    }

    return features, targets


def prepare_data() -> (Dataset, Dataset):
    iris = tf.contrib.learn.datasets.load_iris()
    train_data = iris.data[::2]
    test_data = iris.data[1::2]
    train_target = iris.target[::2]
    test_target = iris.target[1::2]

    return Dataset(train_data, train_target), Dataset(test_data, test_target)


def create_input_fn(dataset: Dataset, shuffle=False, num_epochs=NUM_EPOCHS, batch_size=BATCH_SIZE):

    def input_fn():
        return generic_input_fn(dataset.data, dataset.target, shuffle, num_epochs, batch_size)

    return input_fn
