"""
======================================================
Plotting bandit ucb behavior
======================================================

Show the evolution of the current min and the selected input
with bandit ucb.
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy import minimum

from fluentopt import Bandit
from fluentopt.bandit import ucb_minimize

np.random.seed(42)

def sampler(rng):
    return rng.uniform(-1, 1)

def feval(x):
    return (x ** 2 - 2)

opt = Bandit(sampler=sampler, score=ucb_minimize)
n_iter = 100
opt.update(x=1., y=feval(1.))
for _ in range(n_iter):
    x = opt.suggest()
    y = feval(x)
    opt.update(x=x, y=y)

idx = np.argmin(opt.output_history_)
best_input = opt.input_history_[idx]
best_output = opt.output_history_[idx]
print('best input : {:.2f}, best output : {:.2f}'.format(best_input, best_output))
iters = np.arange(len(opt.output_history_))
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))

ax1.plot(iters, minimum.accumulate(opt.output_history_))
ax1.set_xlabel('iteration')
ax1.set_ylabel('current $min_x({x**2-2})$')

ax2.plot(iters, opt.input_history_)
ax2.set_xlabel('iteration')
ax2.set_ylabel('selected input')

plt.show()
