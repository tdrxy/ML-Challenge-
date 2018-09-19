
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

class CustomTypeTransformer(BaseEstimator, TransformerMixin):
    """
        Custom transformer for column type convertion on pandas df adhering to sklearn's transform iface.
    """
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        assert isinstance(X, pd.DataFrame)
        X.loc[:, 'workclass'] = X.loc[:, 'workclass'].astype('category')
        X.loc[:, 'education'] = X.loc[:, 'education'].astype('category')
        #X.loc[:, 'education-num'] = X.loc[:, 'education-num'].astype('category')
        #X['education-num'] = X['education-num'].apply(str)
        X.loc[:, 'marital-status'] = X.loc[:, 'marital-status'].astype('category')
        X.loc[:, 'occupation'] = X.loc[:, 'occupation'].astype('category')
        X.loc[:, 'relationship'] = X.loc[:, 'relationship'].astype('category')
        X.loc[:, 'race'] = X.loc[:, 'race'].astype('category')
        X.loc[:, 'sex'] = X.loc[:, 'sex'].astype('category')
        X.loc[:, 'native-country'] = X.loc[:, 'native-country'].astype('category')
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

class MissingCategoricalsTransformer(BaseEstimator, TransformerMixin):
    """
    Custom transformer for filling NaNs in pandas df adhering to sklearn's transform iface.
    Two strategies can be used: 'most_frequent' or 'none'. The latter fillsna with 'Unknown'
    """
    def __init__(self, columns, strategy='none'):
        self.columns = columns
        self.strategy = strategy

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        assert isinstance(X, pd.DataFrame)
        for col in self.columns:
            if self.strategy == 'none':
                X.loc[:, col] = X.loc[:, col].cat.add_categories("Unknown").fillna('Unknown')
            elif self.strategy == 'most_frequent':
                X.loc[:, col] = X.loc[:, col].fillna(X.loc[:, col].value_counts().index[0])
            else:
                raise Exception('Unknown strategy to fill na categoricals')
        return X

class PipelineAwareLabelEncoder(TransformerMixin, BaseEstimator):
    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        return LabelEncoder().fit_transform(X).reshape(-1, 1)

class ShuffleTransformer(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        assert isinstance(X, pd.DataFrame)
        return shuffle(X)

class DenseToDataFrameTransformer(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        p = pd.DataFrame(X.toarray())
        p.columns = [str(x) for x in p.columns]
        return p

# Selected columns
# fnlgwt dropped due to low value for predictive power
col_selection = X.columns.tolist()
col_selection.remove("fnlwgt")
# education can be removed as edu-num is more informative and replaces it.
col_selection.remove("education")


categorical_colums = ["workclass", "education", "marital-status", "occupation", "relationship", "race", "sex", "native-country"]
# when columns are deselected, these changes need to be reflected in the categorical
# columns as well as code later depends on this data structure
categorical_colums = [x for x in categorical_colums if x in col_selection]

## TODO SHUFFLE
preprocess_pipeline = make_pipeline(
    # Convert column dtypes of df
    CustomTypeTransformer(),
    # Select the columns we want to use for prediction
    ColumnSelector(columns=col_selection),
    MissingCategoricalsTransformer(columns=categorical_colums, strategy="none"),
    # Standard scaling and one hot encoding
    FeatureUnion(transformer_list=[
        ("numeric_features", make_pipeline(
            TypeSelector(dtypes=["int64"]),
            Imputer(strategy="median"),
            StandardScaler()
        )),
        ("categorical_feature1", make_pipeline(
            ColumnSelector(columns=["workclass"]),
            PipelineAwareLabelEncoder(),
            OneHotEncoder()
        )),
        ("categorical_feature4", make_pipeline(
            ColumnSelector(columns=["marital-status"]),
            PipelineAwareLabelEncoder(),
            OneHotEncoder()
        )),
        ("categorical_feature5", make_pipeline(
            ColumnSelector(columns=["occupation"]),
            PipelineAwareLabelEncoder(),
            OneHotEncoder()
        )),
        ("categorical_feature6", make_pipeline(
            ColumnSelector(columns=["relationship"]),
            PipelineAwareLabelEncoder(),
            OneHotEncoder()
        )),
        ("categorical_feature7", make_pipeline(
            ColumnSelector(columns=["race"]),
            PipelineAwareLabelEncoder(),
            OneHotEncoder()
        )),
        ("categorical_feature8", make_pipeline(
            ColumnSelector(columns=["sex"]),
            PipelineAwareLabelEncoder(),
            OneHotEncoder()
        )),
        ("categorical_feature9", make_pipeline(
            ColumnSelector(columns=["native-country"]),
            PipelineAwareLabelEncoder(),
            OneHotEncoder()
        ))
    ]),
    DenseToDataFrameTransformer()
)

# Z = preprocess_pipeline.fit_transform(X)
#
# ZZ = Z.toarray()
# print(type(Z.toarray()))
# #print(pd.DataFrame(Z.todense()).shape)
# #print(list(pd.DataFrame(Z.todense()).columns.values))
classifier_pipeline = make_pipeline(
    preprocess_pipeline,
    #LogisticRegression()
    TFWrapper()
)

# import numpy as np
# classifier_pipeline.fit(X, y)
# preds = classifier_pipeline.predict(X)


def get_pipeline():
    return classifier_pipeline