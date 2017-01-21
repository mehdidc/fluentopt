import pytest

import numpy as np

from fluentopt.transformers import as_2d
from fluentopt.transformers import Wrapper
from fluentopt.transformers import is_list_of_dicts
from fluentopt.transformers import is_list_of_varying_length_lists
from fluentopt.transformers import vectorize_list_of_varying_length_lists
from fluentopt.transformers import vectorize_list_of_dicts
from fluentopt.transformers import vectorize


def test_is_list_of_dicts():
    assert is_list_of_dicts([])
    assert is_list_of_dicts([{'a': 5}, {'b': 2}])
    assert not is_list_of_dicts([1, 2])
    assert not is_list_of_dicts([[1], [2]])
    assert not is_list_of_dicts(['x', 'y'])


def test_is_list_of_varying_length_lists():
    assert not is_list_of_varying_length_lists([])
    assert is_list_of_varying_length_lists([[1], [1, 2]])
    assert is_list_of_varying_length_lists([[1], [2], [3], [1, 2]])


def test_vectorize_list_of_dicts():
    dlist = [
        {'a': 5},
        {'a': 6},
        {'a': 7}
    ]
    v = vectorize_list_of_dicts(dlist)
    assert isinstance(v, np.ndarray)
    assert v.shape == (3, 1)

    dlist = [
        {'a': 5, 'b': 2},
        {'a': 6, 'b': 3},
        {'a': 7, 'b': 4}
    ]
    v = vectorize_list_of_dicts(dlist)
    assert isinstance(v, np.ndarray)
    assert v.shape == (3, 2)
    assert np.all(v[:, 0] == np.array([5, 6, 7]))
    assert np.all(v[:, 1] == np.array([2, 3, 4]))

    dlist = [
        {'a': 1},
        {'a': 1, 'b': 2},
        {'a': 1}
    ]
    v = vectorize_list_of_dicts(dlist)
    assert isinstance(v, np.ndarray)
    assert v.shape == (3, 2)
    assert np.all(v[:, 0] == np.array([1, 1, 1]))
    assert np.isnan(v[0, 1])
    assert not np.isnan(v[1, 1])
    assert v[1, 1] == 2
    assert np.isnan(v[2, 1])
