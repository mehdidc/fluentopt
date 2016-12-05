from fluentopt import RandomSearch
import numpy as np

def sampler(rng):
    return rng.uniform(-1, 1)

def feval(x):
    return (x ** 2 - 2)

opt = RandomSearch(sampler=sampler)
n_iter = 1000

for _ in range(n_iter):
    x = opt.suggest()
    y = feval(x)
    opt.update(x=x, y=y)

idx = np.argmin(opt.output_history_)
best_input = opt.input_history_[idx]
best_output = opt.output_history_[idx]
print('best input : {:.2f}, best output : {:.2f}'.format(best_input, best_output))
