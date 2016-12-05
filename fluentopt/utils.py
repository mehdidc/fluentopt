"""
This module contains utility functions used by other modules.
It contains mostly validation functions to check for validity
of the parameters that a function or a class gets as an input.
"""
from sklearn.utils import check_random_state

__all__ = [
    "check_sampler",
    "check_types_coherence",
    "check_if_list_of_scalars"
]

def check_sampler(sampler):
    """check whether sampler is a callable"""
    assert callable(sampler), 'The sampler should be callable'
    return sampler

def _types_are_coherent(xlist):
    """return True if the elements of xlist all have the same type"""
    x0 = xlist[0]
    types = map(lambda x:type(x), xlist)
    types = list(types)
    return all(map(lambda t:t == type(x0) or t == type(None), types))

def check_types_coherence(xlist, varname='xlist'):
    assert _types_are_coherent(xlist), 'Types should be coherent in {}'.format(varname)
    return xlist

def check_if_list_of_scalars(ylist, varname='ylist'):
    assert all(type(y) == float or type(y) == int for y in ylist), 'The list {} should only contain'.format(varname)
    return ylist

def argmax(x):
    return max(range(len(x)), key=lambda i:x[i])
