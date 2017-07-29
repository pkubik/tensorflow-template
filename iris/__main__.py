from functools import partial

import tensorflow as tf

import iris.io_utils as io
from iris.cli import CLI
from iris.model import model_fn, DEFAULT_PARAMS


def run(action: str, model_dir: str, overrides: dict):
    print("Using a model from '{}' ({})".format(model_dir, action))

    params = {}
    params.update(DEFAULT_PARAMS)
    params.update(io.load_params(model_dir))
    params.update(overrides)

    num_epochs = params['num_epochs']
    batch_size = params['batch_size']

    trainset, testset = io.prepare_data()
    create_input_fn = partial(io.create_input_fn, batch_size=batch_size)

    # Create estimator
    e = tf.estimator.Estimator(model_fn, model_dir=model_dir, params=params)
    # Create/update the parameters file
    io.store_params(params, model_dir)

    # Train model
    if action == 'train':
        e.train(input_fn=create_input_fn(trainset, num_epochs=num_epochs, shuffle=True))

    # Evaluate model
    metrics = e.evaluate(input_fn=create_input_fn(trainset, num_epochs=1))
    print('Train set accuracy: {}'.format(metrics['accuracy']))
    metrics = e.evaluate(input_fn=create_input_fn(testset, num_epochs=1))
    print('Test set accuracy: {}'.format(metrics['accuracy']))

    predictions = e.predict(input_fn=create_input_fn(testset, num_epochs=1))
    predicted_targets = [p['target'] for p in predictions]

    print()
    print('Prediction results:')
    print(''.join(str(p) for p in predicted_targets))
    print(''.join(str(p) for p in testset.target))
    print(''.join(' ' if p == t else '!' for p, t in zip(predicted_targets, testset.target)))


def main():
    allowed_actions = ['train', 'test']
    allowed_params = sorted(DEFAULT_PARAMS.keys())
    cli = CLI(allowed_actions, allowed_params)
    run(cli.action, cli.model_dir, cli.overrides)


if __name__ == '__main__':
    main()
