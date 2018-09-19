# Machine Learning challenge

## Feature selection
    - fnlgwt was dropped due to low predictive power. (eg 2/3 of rows had unique value for this)
    - education was dropped due to education-num serving as a proxy, with on top of that an ordinal characteristic.
    - All other columns were used.

## Feature engineering
    - Pandas 'object' dtypes were changed to 'category'.
    - Standardscaler was applied all numerical columns. This worked better than MinMax although some variables were not normally distributed.
    - Categorical imputation: NaN were filled with 'unknown' category value (instead of most-frequent).
    - Numerical imputation: Median
    - Categorical columns one hot encoded
    - Class imbalance problem left unchanged, although AUC should be a good indicator for these kind of datasets.

## Tensorflow model selection
    - LinearClassifier was tried first (logistic reg) and achieved 0.9 AUCROC
    - DNNClassifier was used in the end with +- 0.911 AUCROC
    - Hyperparameters were found via mixture of trial-and-error and gridSearchCV

## Possible other paths to explore (*: stretch goals)
    - Feature Crossing: For example make new feature based on cap_gains - cap_losses
    - Column Distribution transformation: StandardScaler worked better than MinMaxScaler, however former assumes Gaussian distri.
      One could try to eg log transform columns to fit a Gaussian distri better, as some variables aren't normally
      distributed.
    - Bin certain numerical columns such as age and hours-per-week and construct indicator columns to avoid continuous domain.
    - Resample data for class imbalance problem - although AUCROC suggests not such a problem.
    - (*) More exotic imputations: use a ML model to impute missing values for variables based on dataset.
    - (*) Joblib dump: I couldn't manage to save the tensorflow estimator with a simple joblib.dump() call. However, I suspect a more
      custom approach is possible where the tensorflow model would be saved using checkpoints/Saver() and the sklearn
      pipeline with a joblib.dump for example. You could afterwards use this data to reconstruct the original pipeline.
    - More exotic neural nets (width/depth/connections)
    - Select K best features
    - Better gridsearch: sampling randomly on log scale. Finding best first rudimentary parameters, then zoom in on that
      subspace, and perform gridsearch on that zoomed in subspace
    - ...
