"""
this module contains transfomers that can vectorize data like
dicts, lists of varying length.
"""

import numpy as np
from sklearn.feature_extraction import DictVectorizer

from .utils import flatten_dict
from .utils import dict_vectorizer

__all__ = [
    "Wrapper",
    "vectorize",
    "vectorize_list_of_varying_length_lists",
    "vectorize_list_of_dicts"
]

def as_2d(x):
    """
    transforms a list of scalars into a 2d numpy array by
    adding an extra dimension.
    If it is already 2d, keep it the same.
    """
    x = np.array(x)
    if x.ndim == 1:
        x = x[:, np.newaxis]
    return x

def is_list_of_dicts(X):
    if type(X) != list:
        return False
    return all(isinstance(x, dict) for x in X)

def is_list_of_varying_length_lists(X):
    if type(X) != list:
        return False
    all_lists = all(isinstance(x, list) for x in X)
    if not all_lists:
        return False
    varying_length = len(set(map(len, X))) > 1
    return varying_length

def vectorize(X):
    """
    vectorizes `X` depending on its type:
        - if it is a list of dicts, use `vectorize_list_of_dicts`.
        - it it is a list of lists of varying length across examples, use `vectorize_list_of_varying_length_lists`.
        - if it is a list of fixed length lists or list of scalars, just convert to numpy array.

    Parameters
    ----------

    `X` : a list of dicts or a list of varying length lists or a list of fixed length lists or list of scalars.

    Returns
    -------
    2D numpy array.

    """
    if is_list_of_dicts(X):
        X = vectorize_list_of_dicts(X)
    elif is_list_of_varying_length_lists(X):
        X = vectorize_list_varying_length_lists(X)
    else:
        X = np.array(X)
    X = as_2d(X)
    return X

def vectorize_list_of_dicts(dlist):
    """
    vectorize a list of dicts
    all columns are considered.
    rows that have missing columns will be replaced by `np.nan`.

    Parameters
    ----------
    dlist : list of dicts

    Returns
    -------
    2D numpy array
    """
    dlist = [flatten_dict(d) for d in dlist]
    colnames = set([k for d in dlist for k in d.keys()])
    colnames = list(colnames)
    colnames = sorted(colnames)#sort cols in alphabetical order
    arr = dict_vectorizer(dlist, colnames, missing=np.nan)
    return arr

def vectorize_list_of_varying_length_lists(X):
    # just consider it as a dict and use vectorize_list_of_dicts
    dlist = [flatten_dict({'list': x}) for x in X]
    return vectorize_list_of_dicts(dlist)

class Wrapper(object):
    """
    wraps a scikit-learn like estimator `model` to transform
    inputs and outputs using `transform_X` and `transform_y`.
    This is used to vectorize easily inputs that are passed
    to the model.

    Parameters
    ----------

    model : scikit-learn like estimator instance to wrap

    transform_X : callable
        used to transform the inputs before passing them to fit and predict

    transform_y : callable
        used to transform the outputs before passing them to fit
    """
    def __init__(self, model, transform_X=vectorize, transform_y=lambda y:y):
        self.model = model
        self.transform_X = transform_X
        self.transform_y = transform_y

    def fit(self, X, y=None):
        X = self.transform_X(X)
        if y:
            y = self.transform_y(y)
        return self.model.fit(X, y=y)

    def predict(self, X, **kwargs):
        # kwargs for handling models which have for instance
        # return_std
        X = self.transform_X(X)
        return self.model.predict(X, **kwargs)
