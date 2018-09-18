from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.multiclass import unique_labels
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted
import tensorflow as tf
import numpy as np
import pandas as pd

#https://www.kaggle.com/salekali/logistic-regression-classification-with-tensorflow


class TFWrapper(object, BaseEstimator, ClassifierMixin):

    def __init__(self, param1):

        # HyperParameters
        self.param1 = param1
        self.batch_size = 128
        self.num_epochs = 100

        # Class specific vars
        self.feat_cols = None
        self.model = None

    def fit(self, X, y):

        # Check that X and y have correct shape
        X, y = check_X_y(X, y)

        # generate feature columns
        # NOTE: ONLY NUMERIC COLUMNS CAN BE PASSED TO THIS WRAPPER
        self.feat_cols = TFWrapper.gen_feat_columns(X)

        # Create input fn for tensorflow
        input_fn = self.create_input_fn(X, y)

        self.model = tf.estimator.LinearClassifier(feature_columns=self.feat_cols, n_classes=2)
        self.model.train(input_fn=input_fn, steps=1000)

        return self

    def predict(self, X):
        input_fn = self.create_input_fn(X)
        self.model.predict(input_fn)
        return -1

    def create_input_fn(self, X, y=None):
        assert isinstance(X, pd.DataFrame)
        # Note to self" following method throws OutOfRangeError when epochs too small during reads
        # https://www.tensorflow.org/api_docs/python/tf/estimator/inputs/pandas_input_fn
        input_fn = tf.estimator.inputs.pandas_input_fn(
            X,
            y=y,
            batch_size=self.batch_size,
            num_epochs=self.num_epochs,
            shuffle=None,
            queue_capacity=1000,
            num_threads=1,
        )
        return input_fn

    @staticmethod
    def gen_feat_columns(X):
        feat_cols = []
        for i in range(len(X.columns) - 1):
            assert(np.asarray(X[:,i]).dtype.kind in set('buifc')) # assert numerical column
            feat_cols.append(tf.feature_column.numeric_column(X.columns[i]))
        return feat_cols

    @staticmethod
    def train_input_fn(features, labels, batch_size):
        """An input function for training"""
        # Convert the inputs to a Dataset.
        dataset = tf.data.Dataset.from_tensor_slices((dict(features), labels))

        # Shuffle, repeat, and batch the examples.
        dataset = dataset.shuffle(1000).repeat().batch(batch_size)

        # Return the dataset.
        return dataset