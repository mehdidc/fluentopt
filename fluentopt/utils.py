"""
This module contains utility functions used by other modules.
It contains mostly validation functions to check for validity
of the parameters that a function or a class gets as an input.
"""
from sklearn.utils import check_random_state

__all__ = [
    "check_random_state",
    "check_sampler"
]

def check_sampler(sampler):
    assert callable(sampler), 'The sampler should be callable'
    return sampler
