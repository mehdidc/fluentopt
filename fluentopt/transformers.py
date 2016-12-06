"""
this module contains transfomers that can vectorize data like
dicts, lists of varying length.
"""

import numpy as np

__all__ = [
    "Wrapper",
    "vectorize",
    "vectorize_list_of_lists",
    "vectorize_list_of_dicts"
]

def as_2d(x):
    x = np.array(x)
    if x.ndim == 1:
        x = x[:, np.newaxis]
    return x

def is_list_of_dicts(X):
    if type(X) != list:
        return False
    return all(isinstance(x, dict) for x in X)

def is_list_of_lists(X):
    if type(X) != list:
        return False
    return all(isinstance(x, list) for x in X)

def vectorize(X):
    if is_list_of_dicts(X):
        X = vectorize_list_of_dicts(X)
    elif is_list_of_lists(X):
        X = vectorize_list_of_lists(X)
    else:
        X = np.array(X)
    X = as_2d(X)
    return X

def vectorize_list_of_dicts(X):
    return X

def vectorize_list_of_lists(X):
    return X

class Wrapper(object):
    """
    wraps a scikit-learn like estimator `model` to transform
    inputs and outputs using `transform_X` and `transform_y`.
    This is used to vectorize easily inputs that are passed
    to the model.
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
        X = self.transform_X(X)
        return self.model.predict(X, **kwargs)
