"""
===========================================================
Hyper-parameter optimization example with random forests
===========================================================

An example with usage an input with a dict format.
"""
import numpy as np
import matplotlib.pyplot as plt
from scipy import maximum

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score

from fluentopt import Bandit
from fluentopt.bandit import ucb_maximize
from fluentopt.utils import RandomForestRegressorWithUncertainty
from fluentopt.transformers import Wrapper

from sklearn.datasets import load_breast_cancer

np.random.seed(42)

data = load_breast_cancer()
data_X, data_y = data['data'], data['target']


def sampler(rng):
    return {'max_depth': rng.randint(1, 100), 'n_estimators': rng.randint(1, 300)}


def feval(d):
    max_depth = d['max_depth']
    n_estimators = d['n_estimators']
    clf = RandomForestClassifier(n_jobs=-1, max_depth=max_depth, n_estimators=n_estimators)
    scores = cross_val_score(clf, data_X, data_y, cv=5, scoring='accuracy')
    return np.mean(scores) - np.std(scores)

opt = Bandit(sampler=sampler, score=ucb_maximize,
             model=Wrapper(RandomForestRegressorWithUncertainty()))
n_iter = 100
for i in range(n_iter):
    print('iter {}...'.format(i))
    x = opt.suggest()
    y = feval(x)
    opt.update(x=x, y=y)

idx = np.argmax(opt.output_history_)
best_input = opt.input_history_[idx]
best_output = opt.output_history_[idx]
print('best input : {}, best output : {:.2f}'.format(best_input, best_output))

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
iters = np.arange(len(opt.output_history_))
ax1.plot(iters, maximum.accumulate(opt.output_history_))
ax1.set_xlabel('iteration')
ax1.set_ylabel('current max')
X = [[inp['max_depth'], inp['n_estimators']] for inp in opt.input_history_]
X = np.array(X)

sc = ax2.scatter(X[:, 0], X[:, 1], c=iters, cmap='viridis', s=20)
ax2.set_xlabel('max_depth')
ax2.set_ylabel('n_estimators')
ax2.set_title('Explored points (color is iteration number)')
fig.colorbar(sc)
plt.show()
