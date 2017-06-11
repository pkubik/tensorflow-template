import tensorflow as tf
from iris.io_utils import create_input_fn, prepare_data
from iris.model import model_fn


def main():
    trainset, testset = prepare_data()

    params = {
        "learning_rate": 0.01,
        "num_hidden": 16
    }
    # By default it creates a new model on every run. To reuse one model pass a path to the model_dir parameter.
    e = tf.estimator.Estimator(model_fn, params=params)

    e.train(input_fn=create_input_fn(trainset, shuffle=True))
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


if __name__ == '__main__':
    main()
