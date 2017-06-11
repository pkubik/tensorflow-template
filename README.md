# TensorFlow Estimator Template

This repository contains a simple *TensorFlow* model for classification which utilise the
`tensorflow.estimator` submodule. This is an example of a very flexible pipeline written
using core *TensorFlow* constructs.

Typical usage of this template would involve making minor modifications to all of
it's files. It is more like an example of how to create a simple TensorFlow training
and evaluation pipeline without writing some tedious tasks by hand (like epochs loop,
batches loop, examples shuffling and storing/restoring checkpoints).

For more detailed example see: <https://www.tensorflow.org/extend/estimators>.

## Sample problem

For completeness the example solves *iris* classification problem using 4 numerical
features. The dataset is available in the `tensorflow.contrib.learn.dataset`.

## Files

- `__main__.py` - usage example.
- `io_utils.py` - data input utilities. Implements loading the *iris* dataset, splitting
 it into train/test subsets and preparing an `input_fn` with *epochs* limit and *batching*.
- `model.py` - contains `model_fn` where the actual neural network is being built (a
 simple 3-layer perceptron with a `tanh` activation function).

## Output Example

Usage has been shown in the `__main__.py`. To run invoke `python3 -m iris` from the repository's
root. A sample output has been placed below (TensorFlow logs has been omitted).

```
Train set accuracy: 0.9733333587646484
Test set accuracy: 0.9466666579246521

Prediction results:
000000000000000000000000011111111111111112111111112222222222222211122222222
000000000000000000000000011111111111111111111111112222222222222222222222222
                                         !                      !!! 
```

Prediction results consist of a line of concatenated predictions and a line of
concatenated true values for comparison.
