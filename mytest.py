
# https://stackoverflow.com/questions/39001956/sklearn-pipeline-how-to-apply-different-transformations-on-different-columns
# https://ramhiser.com/post/2018-04-16-building-scikit-learn-pipeline-with-pandas-dataframe/
# https://stackoverflow.com/questions/47790854/how-to-perform-onehotencoding-in-sklearn-getting-value-error
from sklearn.base import BaseEstimator, TransformerMixin

from challenge import get_data
import pandas as pd
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)

from sklearn.pipeline import Pipeline, make_pipeline, FeatureUnion
from sklearn.preprocessing import StandardScaler, Imputer, OneHotEncoder, LabelEncoder, LabelBinarizer, \
    MultiLabelBinarizer
import numpy as np


X,y = get_data()
#print(type(X))
#print(type(y))
print(X.head(3))
#print(X.describe())

print(X.isnull().sum())
print(X.dtypes)

X = X.head(4)

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
        #X['education-num'] = X['education-num'].apply(str)
        X['marital-status'] = X['marital-status'].astype('category')
        X['occupation'] = X['occupation'].astype('category')
        X['relationship'] = X['relationship'].astype('category')
        X['race'] = X['race'].astype('category')
        X['sex'] = X['sex'].astype('category')
        X['native-country'] = X['native-country'].astype('category')
        return X

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

class MyLabelBinarizer(TransformerMixin):
    def __init__(self, *args, **kwargs):
        self.encoder = LabelBinarizer(*args, **kwargs)
    def fit(self, x, y=0):
        self.encoder.fit(x)
        return self
    def transform(self, x, y=0):
        return self.encoder.transform(x)

class LabelEncodeTransformer():

    def __init__(self, columns):
        self.columns = columns

    def transform(self, X, y=None, **fit_params):
        for col in self.columns:
            X[col] = LabelEncoder().fit_transform(X[col])
        return X

    def fit_transform(self, X, y=None, **fit_params):
        self.fit(X, y, **fit_params)
        return self.transform(X)

    def fit(self, X, y=None, **fit_params):
        return self

class MissingValuesTransformer(BaseEstimator, TransformerMixin):
    """
        Custom transformer for handling missing data in pandas df adhering to sklearn's transform iface.
        Usage ex:
            cs = TypeSelector(dtypes=["int64", "float32"])
            cs.fit_transform(df).head()
    """
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        assert isinstance(X, pd.DataFrame)
        X = X.dropna()
        # .fillna instead
        return X

class PipelineAwareLabelEncoder(TransformerMixin, BaseEstimator):
    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        return LabelEncoder().fit_transform(X).reshape(-1, 1)

# Selected columns
# fnlgwt dropped due to low value for predictive power
col_selection = X.columns.tolist()
col_selection.remove("fnlwgt")
col_selection.remove("education")
col_selection.remove("education-num")
col_selection.remove("marital-status")
col_selection.remove("occupation")
col_selection.remove("relationship")
col_selection.remove("race")
col_selection.remove("sex")
col_selection.remove("native-country")

preprocess_pipeline = make_pipeline(
    CustomTypeTransformer(),
    ColumnSelector(columns=col_selection),
    MissingValuesTransformer(),
    #LabelEncodeTransformer(columns=cols_to_encode),
    FeatureUnion(transformer_list=[
        ("numeric_features", make_pipeline(
            TypeSelector(dtypes=["int64"]),
            #Imputer(strategy="median"),
            StandardScaler()
        )),
        ("categorical_feature1", make_pipeline(
            ColumnSelector(columns=["workclass"]),
            PipelineAwareLabelEncoder(),
            OneHotEncoder()
        )),
        # ("categorical_feature2", make_pipeline(
        #     ColumnSelector(columns=["education"]),
        #     PipelineAwareLabelEncoder(),
        #     OneHotEncoder()
        # )),
        # ("categorical_feature3", make_pipeline(
        #     ColumnSelector(columns=["education-num"]),
        #     PipelineAwareLabelEncoder(),
        #     OneHotEncoder()
        # )),
        # ("categorical_feature4", make_pipeline(
        #     ColumnSelector(columns=["marital-status"]),
        #     PipelineAwareLabelEncoder(),
        #     OneHotEncoder()
        # )),
        # ("categorical_feature5", make_pipeline(
        #     ColumnSelector(columns=["occupation"]),
        #     PipelineAwareLabelEncoder(),
        #     OneHotEncoder()
        # )),
        # ("categorical_feature6", make_pipeline(
        #     ColumnSelector(columns=["relationship"]),
        #     PipelineAwareLabelEncoder(),
        #     OneHotEncoder()
        # )),
        # ("categorical_feature7", make_pipeline(
        #     ColumnSelector(columns=["race"]),
        #     PipelineAwareLabelEncoder(),
        #     OneHotEncoder()
        # )),
        # ("categorical_feature8", make_pipeline(
        #     ColumnSelector(columns=["sex"]),
        #     PipelineAwareLabelEncoder(),
        #     OneHotEncoder()
        # )),
        # ("categorical_feature9", make_pipeline(
        #     ColumnSelector(columns=["native-country"]),
        #     PipelineAwareLabelEncoder(),
        #     OneHotEncoder()
        # ))

    ])
)
print(X)
preprocess_pipeline.fit(X)
a = preprocess_pipeline.transform(X)
print(a.todense())