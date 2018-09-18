
# https://stackoverflow.com/questions/39001956/sklearn-pipeline-how-to-apply-different-transformations-on-different-columns
# https://ramhiser.com/post/2018-04-16-building-scikit-learn-pipeline-with-pandas-dataframe/
from sklearn.base import BaseEstimator, TransformerMixin

from challenge import get_data
import pandas as pd
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)

from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import StandardScaler


X,y = get_data()
print(type(X))
print(type(y))
print(X.head(10))
print(X.describe())

print(X.isnull().sum())
print(X.dtypes)


#print(X['fnlwgt'].size)
#print(len(X['fnlwgt'].unique()))
# too many unique fnlwgt values for predictive power, discard
#print(len(X['fnlwgt'].unique()))

#print(len(X['education-num'].unique()))


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

class CustomTypeTransformer(BaseEstimator, TransformerMixin):
    """
        Custom transformer for column type convertion on pandas df adhering to sklearn's transform iface.
    """
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        assert isinstance(X, pd.DataFrame)
        X['workclass'] = X['workclass'].astype('category')
        X['education'] = X['education'].astype('category')
        X['education-num'] = X['education-num'].astype('category')
        X['marital-status'] = X['marital-status'].astype('category')
        X['occupation'] = X['occupation'].astype('category')
        X['relationship'] = X['relationship'].astype('category')
        X['race'] = X['race'].astype('category')
        X['sex'] = X['sex'].astype('category')
        X['native-country'] = X['native-country'].astype('category')
        return X


cols = ["workclass", "education", "marital-status"]
preprocess_pipeline = make_pipeline(
    CustomTypeTransformer(),
    ColumnSelector(columns=cols)
)

preprocess_pipeline.fit(X)
a = preprocess_pipeline.transform(X)
print(a.head())
print(a.dtypes)