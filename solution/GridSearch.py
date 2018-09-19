from sklearn.model_selection import GridSearchCV
import solution
import numpy as np
#print(solution.get_pipeline().get_params().keys())

# param_grid = dict(
#     #pipeline__missingcategoricalstransformer__strategy=["none"],
#     tensorflowestimator__dropout=[0.15, 0.20, 0.25, 0.3, 0.35, 0.40],
#     tensorflowestimator__hidden_units=[[64,32], [24], [48], [30, 12], [128, 48], [128,48,12]],
#     tensorflowestimator__training_steps=[900]
# )
# TensorFlowEstimator(dropout=0.3, hidden_units=[128, 48, 12], training_steps=900)
param_grid = dict(
    #pipeline__missingcategoricalstransformer__strategy=["none"],
    tensorflowestimator__dropout=[0.35, 0.40, 0.45, 0.5],
    tensorflowestimator__hidden_units=[[128,48,12], [1024, 514, 256, 128, 64], [200, 100, 40], [56, 28, 12]],
    tensorflowestimator__training_steps=[900]
)

#{'tensorflowestimator__dropout': 0.4, 'tensorflowestimator__hidden_units': [200, 100, 40], 'tensorflowestimator__training_steps': 900}

gs = GridSearchCV(solution.get_pipeline(), param_grid, cv=6, n_jobs=4)

from challenge import get_dataw
X, y = get_data()


# print(X.index.values)
# index = np.arange(X.index.values[0], X.index.values[-1]+1)
# print(index)
#
# X.set_index(index, inplace=True)
# print(X.index.values)

gs.fit(X, y)
print(gs.best_params_)