"""

"""
from .base import Optimizer
from .utils import check_random_state
from .utils import check_sampler

__all__ = [
    "RandomSearch"
]

class RandomSearch(Optimizer):
    """
    a random search optimizer.
    This optimizer is completely random, it does not use
    any surrogate model. It uses a `sampler`, which have to be
    a callable (e.g function) that takes a random state (integer)
    as input and returns a random sample. The `suggest` method
    just calls `sampler` each time.

    Parameters
    ----------
    sampler : callable
        a callable used to sample an input for further evaluation.
        it takes one argument, a random number generator following the API
        of numpy.random and returns a dict, a list or a scalar.
    random_state : int or None, optional
        controls the random seed used by `sampler`.

    Attributes
    ----------
        input_history_ : list of inputs evaluated
        output_history_: outputs corresponding to the evaluated inputs
    """

    def __init__(self, sampler, random_state=None):
        self.sampler = check_sampler(sampler)
        self.rng = check_random_state(random_state)
        self.input_history_ = []
        self.output_history_ = []

    def update(self, x, y):
        self.update_many([x], [y])

    def update_many(self, xlist, ylist):
        self.input_history_.extend(xlist)
        self.output_history_.extend(ylist)

    def suggest(self):
        return self.sampler(self.rng)
