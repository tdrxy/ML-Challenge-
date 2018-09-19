
# https://stackoverflow.com/questions/39001956/sklearn-pipeline-how-to-apply-different-transformations-on-different-columns
# https://ramhiser.com/post/2018-04-16-building-scikit-learn-pipeline-with-pandas-dataframe/
# https://stackoverflow.com/questions/47790854/how-to-perform-onehotencoding-in-sklearn-getting-value-error

## joblib dump
# https://stackoverflow.com/questions/34143829/sklearn-how-to-save-a-model-created-from-a-pipeline-and-gridsearchcv-using-jobli
import sklearn
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.linear_model import LogisticRegression

from challenge import get_data
import pandas as pd

from solution.TFWrapper import TFWrapper

pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)

from sklearn.pipeline import Pipeline, make_pipeline, FeatureUnion
from sklearn.preprocessing import StandardScaler, Imputer, OneHotEncoder, LabelEncoder
from sklearn.utils import shuffle


X,y = get_data()

class ColumnSelector(BaseEstimator, TransformerMixin):
    """
    Custom transformer for column selection on pandas df adhering to sklearn's transform iface.
    Usage ex:
        cs = ColumnSelector(columns=["feat1", "feat2"])
        cs.fit_transform(df).head()
    """
    def __init__(self, columns):
        self.columns = columns

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        assert isinstance(X, pd.DataFrame)

        try:
            return X[self.columns]
        except KeyError:
            cols_error = list(set(self.columns) - set(X.columns))
            raise KeyError("The DataFrame does not include the columns: %s" % cols_error)

class TypeSelector(BaseEstimator, TransformerMixin):
    """
        Custom transformer for selecting columns based on type in pandas df adhering to sklearn's transform iface.
            Usage ex:
            cs = TypeSelector(dtypes=["int64", "float32"])
            cs.fit_transform(df).head()
    """
    def __init__(self, dtypes):
        self.dtypes = dtypes

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        assert isinstance(X, pd.DataFrame)
        return X.select_dtypes(include=self.dtypes)


class DenseToDataFrameTransformer(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        p = pd.DataFrame(X)
        p.columns = [str(x) for x in p.columns]
        return p

# Selected columns
# fnlgwt dropped due to low value for predictive power
col_selection = X.columns.tolist()
col_selection.remove("fnlwgt")
# education can be removed as edu-num is more informative and replaces it.

preprocess_pipeline = make_pipeline(
    # Convert column dtypes of df
    ColumnSelector(columns=col_selection),
    TypeSelector(dtypes=["int64"]),
    Imputer(strategy="median"),
    StandardScaler(),
    DenseToDataFrameTransformer()
)

classifier_pipeline = make_pipeline(
    preprocess_pipeline,
    #LogisticRegression()
    TFWrapper()
)

def get_pipeline():
    return classifier_pipeline