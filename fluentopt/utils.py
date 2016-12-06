"""
This module contains utility functions used by other modules.
It contains mostly validation functions to check for validity
of the parameters that a function or a class gets as an input.
"""
import collections

import numpy as np
from sklearn.utils import check_random_state
from sklearn.ensemble import RandomForestRegressor

__all__ = [
    "check_sampler",
    "check_types_coherence",
    "check_if_list_of_scalars",
    "argmax",
    "flatten_dict",
    "dict_vectorizer",
    "RandomForestRegressorWithUncertainty"
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
    assert all(isinstance(y, float) or isinstance(y, int) for y in ylist), 'The list {} should only contain'.format(varname)
    return ylist

def argmax(x):
    return max(range(len(x)), key=lambda i:x[i])

def flatten_dict(D):
    """
    converts a deep dict `D` into a flattened version `d`.
    it uses a recursive algo:
        - start with an empty `d`
        - iterate through the keys and values of D:
            - if the current value is a dict then update
                `d` with the flattened version of the value
                by calling `flatten_dict` on the value
            - if the current value is a list or a tuple add all elements
                of the list with keys `key_i` where i is the index of the
                element of the list and values the value of the list
                at index i.
            - else just copy the key and value of `D` into `d`
    """
    d = {}
    for k, v in D.items():
        if isinstance(v, collections.Mapping):
            d.update(flatten_dict(v))
        elif isinstance(v, list) or isinstance(v, tuple):
            for i, l in enumerate(v):
                if not isinstance(l, collections.Mapping):
                    d[k+'_{}'.format(i)] = l
                else:
                    for e in v:
                        d.update(flatten_dict(e))
        else:
            d[k] = v
    return d

def dict_vectorizer(dlist, colnames, missing=np.nan):
    """
    Converts a list of dicts into a numpy array.
    the i-th dimension of the vector will correspond to colnames[i].
    if a column does not exist in a dict from `dlist`, it takes
    the value defined by `missing`.

    Parameters
    ----------
    dlist : list of dicts

    colnames : list of strings
        list of columns to use.
        the order of the columns in the resulting numpy will
        correspond to the order in `colnames`.

    missing : scalar
        the value to use for missing columns.

    Returns
    -------

    2D numpy array
    """
    dlist_ = [[d.get(col, missing) for col in colnames] for d in dlist]
    return np.array(dlist_)

class RandomForestRegressorWithUncertainty(RandomForestRegressor):
    """
    an extension of RandomForestRegressor with support of returning uncertainty.
    it just takes the trees and compute the std of the predicted values for each
    tree.
    """
    def predict(self, X, return_std=False):
        if return_std:
            trees = self.estimators_
            y = np.concatenate([tree.predict(X)[np.newaxis, :] for tree in trees], axis=0)
            mean = y.mean(axis=0)
            std = y.std(axis=0)
            return mean, std
        else:
            return super(RandomForestRegressor, self).predict(X)
