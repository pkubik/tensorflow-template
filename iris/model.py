import tensorflow as tf


DEFAULT_PARAMS = {
    'num_epochs': 100,
    'batch_size': 25,
    'num_classes': 3,
    "learning_rate": 0.01,
    "num_hidden": 16
}


def model_fn(mode, features, labels, params):
    # Feature passed from the input_fn
    attributes = features['attributes']
    # Network hyperparameter
    num_classes = params['num_classes']
    num_hidden = params['num_hidden']

    # Actual neural network with one hidden layer
    hidden = tf.layers.dense(attributes, num_hidden, activation=tf.tanh, name='hidden')
    output = tf.layers.dense(hidden, num_classes, name='logits')

    # Assign a default value to the train_op and loss to be passed for modes other than TRAIN
    loss = None
    train_op = None
    # Following part of the network will be constructed only for training
    if mode != tf.estimator.ModeKeys.PREDICT:
        # Targets batch passed from the input_fn
        targets = labels['targets']
        # Transform the targets vector into a batch of one-hot vectors
        hot_targets = tf.one_hot(targets, num_classes)

        # We do not assign our loss to a variable because all losses are added to losses collection internally
        tf.losses.softmax_cross_entropy(hot_targets, output)
        loss = tf.losses.get_total_loss()
        train_op = tf.contrib.layers.optimize_loss(
            loss=loss,  # Total loss from the losses collection
            global_step=tf.contrib.framework.get_global_step(),
            learning_rate=params["learning_rate"],
            optimizer="SGD")

    # We could skip making predictions in the train mode, but they don't do any harm
    target_predictions = tf.arg_max(output, -1, name='predictions')

    eval_metric_ops = None
    if mode == tf.estimator.ModeKeys.EVAL:
        accuracy = tf.metrics.accuracy(targets, target_predictions, name='accuracy')
        eval_metric_ops = {
            'accuracy': accuracy
        }

    predictions = {
        'target': target_predictions
    }

    return tf.estimator.EstimatorSpec(
        mode=mode,
        predictions=predictions,
        loss=loss,
        train_op=train_op,
        eval_metric_ops=eval_metric_ops)
