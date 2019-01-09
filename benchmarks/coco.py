"""
This benchmark example uses the coco benchmark set of functions
(<http://coco.gforge.inria.fr/>, <https://github.com/numbbo/coco>)
to compare optimizers provided by fluentopt between themselves and also
with CMA-ES[1].
To run these benchmarks, the package 'cocoex' must be installed,
check <https://github.com/numbbo/coco> to see how to install it.
Also, the package 'cma' is needed and can be installed by pip.
For each function, each algorithm is ran for independent trials
and the results are all written in a csv file (by default benchmarks.csv).
each row correspond to a trial for a given algo and function.
The columns are:
    - 'func' : function name (str)
    - 'algo' : algo name (str)
    - 'nbeval' : nb of evaluations performed (int)
    - 'ybest' : the best output value found (float)
    - 'duration' : duration in seconds (float)
[1] Nikolaus Hansen and Andreas Ostermeier, Completely derandomized
    self-adaptation in evolution strategies.
    Evolutionary computation, 9(2):159–195, 2001
"""
import time

import numpy as np
import pandas as pd

from cocoex import Suite, Observer

from fluentopt import Bandit
from fluentopt.bandit import ucb_minimize
from fluentopt.transformers import Wrapper
from fluentopt import RandomSearch

from cma import fmin as cma_fmin
from cma import CMAEvolutionStrategy

from clize import run


def cma(fun, budget):
    sigma0 = 0.02
    range_ = fun.upper_bounds - fun.lower_bounds
    center = fun.lower_bounds + range_ / 2
    x0 = center
    options = dict(
        scaling=range_ / range_[0], maxfevals=budget, verb_log=0, verb_disp=1, verbose=1
    )
    es = CMAEvolutionStrategy(x0, sigma0 * range_[0], options)
    res = es.optimize(fun).result()
    xbest, ybest, nbeval, *rest = res
    return xbest, ybest, nbeval


def ucb(fun, budget):
    sampler = _uniform_sampler(low=fun.lower_bounds, high=fun.upper_bounds)
    opt = Bandit(sampler=sampler, score=ucb_minimize, nb_suggestions=100)
    return _run_opt(opt, fun, budget)


def random_search(fun, budget):
    sampler = _uniform_sampler(low=fun.lower_bounds, high=fun.upper_bounds)
    opt = RandomSearch(sampler=sampler)
    return _run_opt(opt, fun, budget)


def _uniform_sampler(low, high):
    low = np.array(low)
    high = np.array(high)
    dim = len(low)

    def sampler_(rng):
        return rng.uniform(0, 1, size=dim) * (high - low) + low

    return sampler_


def _run_opt(opt, feval, budget):
    for _ in range(budget):
        x = opt.suggest()
        y = feval(x)
        opt.update(x=x, y=y)
    idx = np.argmin(opt.output_history_)
    xbest = opt.input_history_[idx]
    ybest = opt.output_history_[idx]
    nbeval = budget
    return xbest, ybest, nbeval


def main(nb_trials=15, budget_per_dim=100, output="benchmark.csv"):
    suite_instance = "year:2016"
    suite_name = "bbob"
    suite_options = ""
    suite = Suite(suite_name, suite_instance, suite_options)
    algos = [random_search, cma, ucb]
    stats = []
    for i, fun in enumerate(suite):
        print("Function {}".format(fun.name))
        for algo in algos:
            algo_name = algo.__name__
            print('Algo : "{}"'.format(algo_name))
            for trial in range(nb_trials):
                print("Running trial {}...".format(trial + 1))
                t0 = time.time()
                xbest, ybest, nbeval = algo(fun, budget_per_dim * fun.dimension)
                delta_t = time.time() - t0
                stats.append(
                    {
                        "func": fun.id,
                        "algo": algo_name,
                        "nbeval": nbeval,
                        "ybest": ybest,
                        "duration": delta_t,
                    }
                )
    stats = pd.DataFrame(stats)
    stats.to_csv(output, index=False)


if __name__ == "__main__":
    run(main)
