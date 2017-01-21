"""
This module contains a base class of optimizers.
This module purpose is to describe the API that optimizers
should follow.
"""
from .utils import check_types_coherence
from .utils import check_if_list_of_scalars

__all__ = [
    "Optimizer",
    "OptimizerWithHistory",
    "OptimizerWithSurrogate"
]


class Optimizer(object):
    """
    Optimizer base class
    """

    def update(self, x, y):
        """
        Update the surrogate used by the optimizer using a single evaluation.

        Parameters
        ----------
        x: dict, or list or scalar
        """
        self.update_many([x], [y])

    def update_many(self, xlist, ylist):
        """
        Update the surrogate used by the optimizer using a list of evaluations.

        Parameters
        ----------
        xlist : list of dicts, or list of lists or list of scalars
        """
        raise NotImplementedError()

    def suggest(self):
        """
        Use the surrogate to suggest a next input to evaluate

        Returns
        -------
        a dict, a list or a scalar
        """
        raise NotImplementedError()


class OptimizerWithHistory(Optimizer):

    def __init__(self):
        self.input_history_ = []
        self.output_history_ = []

    def update_many(self, xlist, ylist):
        assert len(xlist) == len(ylist), "xlist and ylist should have the same length"
        check_types_coherence(self.input_history_[-1:] + xlist)
        check_if_list_of_scalars(ylist)
        self.input_history_.extend(xlist)
        self.output_history_.extend(ylist)


class OptimizerWithSurrogate(OptimizerWithHistory):

    def __init__(self, model):
        super(OptimizerWithSurrogate, self).__init__()
        self.model = model

    def update_many(self, xlist, ylist):
        super(OptimizerWithSurrogate, self).update_many(xlist, ylist)
        self.model.fit(self.input_history_, self.output_history_)
