from sklearn.model_selection import GridSearchCV
import solution
import numpy as np
#print(solution.get_pipeline().get_params().keys())

param_grid = dict(
    #pipeline__missingcategoricalstransformer__strategy=["none"],
    tensorflowestimator__dropout=[0.20, 0.25, 0.3, 0.35],
    tensorflowestimator__hidden_units=[[64,32], [48], [128, 48], [128,48,12]],
    tensorflowestimator__training_steps=[900]
)

gs = GridSearchCV(solution.get_pipeline(), param_grid, cv=5, n_jobs=1)

from challenge import get_data
X, y = get_data()


# print(X.index.values)
# index = np.arange(X.index.values[0], X.index.values[-1]+1)
# print(index)
#
# X.set_index(index, inplace=True)
# print(X.index.values)

gs.fit(X, y)
print(gs.best_params_)