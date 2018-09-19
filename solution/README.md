# Machine Learning challenge

## Feature selection
    - fnlgwt was dropped due to low predictive power. (eg 2/3 of rows had unique value for this)
    - education was dropped due to education-num serving as a proxy, with on top of that an ordinal characteristic.

## Feature engineering
    - Pandas 'object' dtypes were changed to 'category'.
    - Standardscaler was applied all numerical columns. This worked better than MinMax although some variables were not normally distributed.
    - Categorical imputation: NaN were filled with 'unknown' category value (instead of most-frequent).
    - Numerical imputation: Median
    - Categorical columns one hot encoded
    - Class imbalance problem left unchanged, although AUC should be a good indicator for these kind of datasets.

## Tensorflow model selection
    - LinearClassifier was tried first (logistic reg) and achieved 0.9 AUCROC
    - DNNClassifier was used in the end with +- 0.91 AUCROC
    - Hyperparameters were found via mixture of trial-and-error and gridSearchCV


## Possible other paths to explore
# Feature Crossing
    - For example make new feature based on cap_gains - cap_losses

# Distribution transformation
    - StandardScaler worked better than MinMaxScaler, however former assumes Gaussian distri.
      One could try to eg log transform columns to fit a Gaussian distri better, as some variables aren't normally
      distributed.
