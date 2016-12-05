"""
This module contains a base class of optimizers.
This module purpose is to describe the API that optimizers
should follow.
"""
from .utils import check_types_coherence
from .utils import check_if_list_of_scalars

class Optimizer:
    """
    Optimizer base class
    """

    def update(self, x, y):
        """
        Update the surrogate used by the optimizer using a single evaluation.

        Parameters
        ----------
        x: dict, or list or scalar
        outputs: scalar
        """
        self.update_many([x], [y])

    def update_many(self, xlist, ylist):
        """
        Update the surrogate used by the optimizer using a list of evaluations.

        Parameters
        ----------
        xlist : list of dicts, or lists or scalars
        outputs: list of scalars
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

    def update(self, x, y):
        self.update_many([x], [y])

    def update_many(self, xlist, ylist):
        assert len(xlist) == len(ylist), "xlist and ylist should have the same length"
        check_types_coherence(self.input_history_[-1:] + xlist)
        check_if_list_of_scalars(ylist)
        self.input_history_.extend(xlist)
        self.output_history_.extend(ylist)
