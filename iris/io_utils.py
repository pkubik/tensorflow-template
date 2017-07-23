import os

import numpy as np
import tensorflow as tf
import json
from tensorflow.contrib.learn.python.learn.datasets.base import Dataset


MIN_AFTER_DEQUEUE = 512
PARAMS_FILENAME = 'params.json'


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


def create_input_fn(dataset: Dataset, num_epochs: int, batch_size: int, shuffle=False):

    def input_fn():
        return generic_input_fn(dataset.data, dataset.target, shuffle, num_epochs, batch_size)

    return input_fn


def get_model_dir(name: str = None):
    import tempfile

    models_dir = os.environ.get('TENSORFLOW_MODELS_DIR')
    if models_dir is None:
        tmp_dir = tempfile.gettempdir()
        default_models_subdir = 'tf_models'
        models_dir = os.path.join(tmp_dir, default_models_subdir)

    if name is not None:
        model_dir = os.path.join(models_dir, name)
    else:
        unnamed_models_dir = os.path.join(models_dir, 'unnamed')
        os.makedirs(unnamed_models_dir, exist_ok=True)
        model_dir = tempfile.mkdtemp(dir=unnamed_models_dir, prefix='')

    return model_dir


def store_params(params: dict, model_dir: str):
    with open(os.path.join(model_dir, PARAMS_FILENAME), 'w') as params_file:
        json.dump(params, params_file)


def load_params(model_dir: str) -> dict:
    os.makedirs(model_dir, exist_ok=True)
    try:
        with open(os.path.join(model_dir, PARAMS_FILENAME), 'r') as params_file:
            return json.load(params_file)
    except FileNotFoundError:
        return {}
