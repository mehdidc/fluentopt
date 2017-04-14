"""
This module provides bandit based optimizers that use
surrogates.
"""

from sklearn.gaussian_process import GaussianProcessRegressor

from .base import OptimizerWithSurrogate
from .transformers import Wrapper
from .utils import check_random_state
from .utils import check_sampler
from .utils import argmax

__all__ = [
    "Bandit",
    "ucb_maximize",
    "ucb_minimize"
]


def ucb_maximize(model, inputs, kappa=1.96):
    """
    UCB score that can be used as
    the `score` parameter of the `Bandit` optimizer.
    Use this score if the objective is maximization with ucb.
    UCB scores assume that the model can return std, that is,
    `model.predict` shoud accept a `return_std` parameter.
    An exception will be thrown if this is not the case.

    Parameters
    ==========

    model : scikit-learn like estimator with return_std
    inputs : numpy array
    kappa : float
        controls the tradeoff between exploration and exploitation
        (higher value = more exploration)
    """
    # ucb scores assume that the model can return std
    # an exception will be thrown if this is not the case
    pred, std = model.predict(inputs, return_std=True)
    return pred + kappa * std


def ucb_minimize(model, inputs, kappa=1.96):
    """
    UCB score that can be used as
    the `score` parameter of the `Bandit` optimizer.
    Use this score if the objective is minimization.
    UCB scores assume that the model can return std, that is,
    `model.predict` shoud accept a `return_std` parameter.
    An exception will be thrown if this is not the case.

    model : scikit-learn like estimator with return_std
    inputs : numpy array
    kappa : float
        controls the tradeoff between exploration and exploitation
        (higher value = more exploration)
    """
    # ucb scores assume that the model can return std
    # an exception will be thrown if this is not the case
    pred, std = model.predict(inputs, return_std=True)
    # the -(...) because we always maximize in the Bandit `Optimizer`
    return -(pred - kappa * std)


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

    model : scikit-learn like model instance, optional
        default is fluentopt.transformers.Wrapper(GaussianProcessRegressor(normalize_y=True)).
        Alternatives :
            - fluentopt.transformers.Wrapper(fluentopt.utils.RandomForestRegressorWithUncertainty())
            - or use another model which supports returning uncertainty in prediction:
                fluentopt.transformers.Wrapper(your_model())
            - you can also extend or change the Wrapper, the goal of the wrapper is to feed a vectorized
              input to the wrapped model.

    nb_suggestions : int, optional[default=100]
        number of random samples to draw from the `sampler` in each
        call of `suggest` to select the next input to evaluate.

    score : callable, optional[default=ucb_maximize]
        score function to use when selecting the next input to evaluate.
        it takes two arguments, a model and a list of inputs.
        it returns a list of scores.
        Available scores are : `ucb_maximize`, `ucb_minimize`.

    random_state : int or None, optional
        controls the random seed used by `sampler`.

    Attributes
    ----------
        input_history_ : list of inputs evaluated
        output_history_: outputs corresponding to the evaluated inputs

    """

    def __init__(self,
                 sampler,
                 model=Wrapper(GaussianProcessRegressor(normalize_y=True)),
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

        # if the history is empty, just sample randomly (because we don't have yet a surrogate)
        if len(self.input_history_) == 0:
            return self.sampler(self.rng)
        else:
            xnext = [self.sampler(self.rng) for _ in range(self.nb_suggestions)]
            scores = self.get_scores(xnext)
            return xnext[argmax(scores)]
