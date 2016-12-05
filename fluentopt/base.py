"""
This module contains a base class of optimizers.
This module purpose is to describe the API that optimizers
should follow.
"""
class Optimizer:
    """
    Optimizer base class
    """
    def update(self, inputs, outputs):
        """
        Update the surrogate used by the optimizer.

        Parameters
        ----------
        inputs: list of dicts
        outputs: list of scalars (can be floats or integers or a mix)
        """
        raise NotImplementedError()

    def suggest(self):
        """
        Use the surrogate to suggest a next input to evaluate

        Returns
        -------
        a dict
        """
        raise NotImplementedError()
