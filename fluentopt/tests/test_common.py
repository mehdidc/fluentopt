from functools import partial
import pytest

import numpy as np

from fluentopt import RandomSearch
from fluentopt import Bandit
from fluentopt.bandit import ucb_minimize
from fluentopt.bandit import ucb_maximize

opts = [
    RandomSearch,
    partial(Bandit, score=ucb_minimize),
    partial(Bandit, score=ucb_maximize),
]


def unif_sampler(rng):
    return rng.uniform(-1, 1)


def dict_unif_sampler(rng):
    return {
        'x': unif_sampler(rng),
        'y': unif_sampler(rng)
    }


@pytest.mark.parametrize("optimizer_cls", opts)
def test_update(optimizer_cls):
    opt = optimizer_cls(unif_sampler)
    assert len(opt.input_history_) == 0
    assert len(opt.output_history_) == 0
    opt.update(x=1, y=5)
    opt.update(x=2, y=6)
    opt.update(x=3, y=7)
    opt.update(x=4, y=8)
    assert opt.input_history_ == [1, 2, 3, 4]
    assert opt.output_history_ == [5, 6, 7, 8]

    opt = optimizer_cls(unif_sampler)
    opt.update_many(xlist=[1, 2, 3, 4], ylist=[5, 6, 7, 8])

    assert opt.input_history_ == [1, 2, 3, 4]
    assert opt.output_history_ == [5, 6, 7, 8]


@pytest.mark.parametrize("optimizer_cls", opts)
def test_coherence(optimizer_cls):
    opt = optimizer_cls(unif_sampler)
    opt.update(x=1, y=2)
    pytest.raises(AssertionError, opt.update, x=[7], y=2)
    pytest.raises(AssertionError, opt.update, x={'a': 5}, y=2)

    opt = optimizer_cls(unif_sampler)
    pytest.raises(AssertionError, opt.update_many, xlist=[1, [7]], ylist=[2])
    pytest.raises(AssertionError, opt.update_many, xlist=[
                  1, 2, 3, 4, {'a': 5}], ylist=[2, 2, 2, 2, 2])

    opt = optimizer_cls(unif_sampler)
    pytest.raises(AssertionError, opt.update_many, xlist=[1, 2], ylist=[2])

    opt = optimizer_cls(unif_sampler)
    pytest.raises(AssertionError, opt.update_many, xlist=[1, 2], ylist=[2, None])
