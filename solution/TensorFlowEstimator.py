import sklearn
from sklearn.base import BaseEstimator, ClassifierMixin
import tensorflow as tf
import numpy as np
import pandas as pd
from pandas.api.types import is_numeric_dtype
from tensorflow.python.estimator.canned.dnn import DNNClassifier

# Wrapper class for Tensorflow model
class TensorFlowEstimator(BaseEstimator, ClassifierMixin):

    def __init__(self, batch_size=100, num_epochs=1000, dropout=0.27, hidden_units=[64, 32], training_steps=950):

        # HyperParameters
        self.batch_size = batch_size
        self.num_epochs = num_epochs
        self.dropout = dropout
        self.hidden_units = hidden_units.copy()
        self.training_steps = training_steps

        # Class specific vars
        self.feat_cols = None
        self.model = None

    # NOTE: ONLY NUMERIC COLUMNS CAN BE PASSED TO THIS WRAPPER
    # code for supporting categorical: https://stackoverflow.com/questions/46834680/creating-many-feature-columns-in-tensorflow
    def fit(self, X, y):
        self.feat_cols = TensorFlowEstimator._gen_feat_columns(X)

        # Create input fn for tensorflow
        input_fn = self._create_input_fn(X, y, num_epochs=self.num_epochs, batch_size=self.batch_size, shuffle=True)

        self.model = DNNClassifier(
            feature_columns=self.feat_cols,
            n_classes=2,
            dropout=self.dropout,
            hidden_units=self.hidden_units)
        self.model.train(input_fn=input_fn, steps=self.training_steps)

        return self

    def predict(self, X):
        probs_1 = self.predict_proba(X)
        rounded = list(map((lambda x: 0 if x <= 0.5 else 1), probs_1.tolist()))
        return rounded

    def predict_proba(self, X):
        input_fn = TensorFlowEstimator._create_input_fn(X)
        preds = self.model.predict(input_fn)
        probs_raw = []
        predictions = []
        for dict in list(preds):
            try:
                probs_raw.append(dict['probabilities'].tolist())
                predictions.append(int(dict['class_ids']))
            except Exception:
                pass

        return np.array(probs_raw)[:, 1]


    @staticmethod
    def _create_input_fn(X, y=None, num_epochs=1, batch_size=10, shuffle=False):
        """
        Create input fn function for tensorflow model
        :param X: pandas dataframe
        :param y: pandas series or none
        :param num_epochs: epoch
        :param batch_size: batch size
        :param shuffle: shuffle data y/n
        :return: tensorflow input function for dataset generation
        """
        assert isinstance(X, pd.DataFrame)
        try:
            input_fn = tf.estimator.inputs.pandas_input_fn(
                    X,
                    y=y,
                    batch_size=batch_size,
                    num_epochs=num_epochs,
                    shuffle=shuffle,
                    queue_capacity=1000,
                    num_threads=1,
            )
            return input_fn
        except ValueError: # gridsearch cv changes rangeIndex of pandas series to int64 index instead of RangeIndex
            y.index = X.index
            input_fn = tf.estimator.inputs.pandas_input_fn(
                X,
                y=y,
                batch_size=batch_size,
                num_epochs=num_epochs,
                shuffle=shuffle,
                queue_capacity=1000,
                num_threads=1,
            )
            return input_fn

    @staticmethod
    def _gen_feat_columns(X):
        """
        Generate feature columns for TF model.
        :param X: pandas dataframe
        :return: list of tf.featurecolumn
        """
        feat_cols = []
        for col in X.columns:
            assert (is_numeric_dtype(X[col]))
            feat_cols.append(tf.feature_column.numeric_column(col))
        return feat_cols
