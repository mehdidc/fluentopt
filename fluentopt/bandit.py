"""
This module provides bandit based optimizers that use
surrogates.
"""

from .base import OptimizerWithSurrogate
from .transformers import as_2d
from .utils import check_random_state
from .utils import check_sampler
from .utils import argmax
from .transformers import Wrapper

from sklearn.gaussian_process import GaussianProcessRegressor

__all__ = [
    "Bandit"
]


def ucb_maximize(model, inputs):
    pred, std = model.predict(inputs, return_std=True)
    return pred + std

def ucb_minimize(model, inputs):
    pred, std = model.predict(inputs, return_std=True)
    return -(pred - std)

class Bandit(OptimizerWithSurrogate):
    """
    a bandit based optimizer which uses a surrogate to model
    the mapping between inputs and outputs. Each time `suggest`
    is called, a total of `nb_suggestions` inputs are sampled from
    `sampler`. A score is then calculated for each sampled input,
    then the next input to evaluate is the one which has the
    maximum score (reward).

    Parameters
    ----------
    sampler : callable
        a callable used to sample an input for further evaluation.
        it takes one argument, a random number generator following the API
        of numpy.random and returns a dict, a list or a scalar.

    model : scikit-learn like model instance, optional(default is GaussianProcessRegressor)

    nb_suggestions : int, optional[default=100]
        number of suggestions to sample from the `sampler` used
        to select the next input to evaluate.

    score : callable, optional[default=ucb]
        score function to use when selecting the next input to evaluate.
        it takes two arguments, a model and a list of inputs.
        it returns a list of scores.

    random_state : int or None, optional
        controls the random seed used by `sampler`.

    Attributes
    ----------
        input_history_ : list of inputs evaluated
        output_history_: outputs corresponding to the evaluated inputs

    """
    def __init__(self,
                 sampler,
                 model=Wrapper(GaussianProcessRegressor()),
                 nb_suggestions=100,
                 score=ucb_maximize,
                 random_state=None):
        super(Bandit, self).__init__(model)
        self.sampler = check_sampler(sampler)
        self.rng = check_random_state(random_state)
        self.nb_suggestions = nb_suggestions
        self.score = score

    def get_scores(self, inputs):
        """ use `score` to get the list of scores of the `inputs`"""
        return self.score(self.model, inputs)

    def suggest(self):
        xnext = [self.sampler(self.rng) for _ in range(self.nb_suggestions)]
        scores = self.get_scores(xnext)
        return xnext[argmax(scores)]
