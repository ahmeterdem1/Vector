from vectorgebra.array import api
import numpy as np
import pytest

api.random.seed(42)

def test_arange():
    arr = api.arange(0, 5)
    expected = np.arange(0, 5)
    assert arr.shape == (5,)
    assert np.allclose(arr.values, expected.ravel())

def test_linspace():
    arr = api.linspace(0, 1, 5)
    expected = np.linspace(0, 1, 5)
    assert arr.shape == (5,)
    assert np.allclose(arr.values, expected.ravel())

def test_zeros():
    arr = api.zeros((2, 3))
    expected = np.zeros((2, 3))
    assert arr.shape == (2, 3)
    assert np.allclose(arr.values, expected.ravel())

def test_ones():
    arr = api.ones((2, 2))
    expected = np.ones((2, 2))
    assert arr.shape == (2, 2)
    assert np.allclose(arr.values, expected.ravel())

def test_full():
    arr = api.full((2, 3), fill_value=7)
    expected = np.full((2, 3), 7)
    assert arr.shape == (2, 3)
    assert np.allclose(arr.values, expected.ravel())

def test_eye():
    arr = api.eye(3, 3)
    expected = np.eye(3)
    assert arr.shape == (3, 3)
    assert np.allclose(arr.values, expected.ravel())

def test_empty():
    arr = api.empty((2, 2))
    assert arr.shape == (2, 2)
    assert all(v is None for v in arr.values)

def test_tril():
    x = api.ones((3, 3))
    arr = api.tril(x)
    expected = np.tril(np.ones((3, 3)))
    assert arr.shape == (3, 3)
    assert np.allclose(arr.values, expected.ravel())

def test_triu():
    x = api.ones((3, 3))
    arr = api.triu(x)
    expected = np.triu(np.ones((3, 3)))
    assert arr.shape == (3, 3)
    assert np.allclose(arr.values, expected.ravel())

@pytest.mark.parametrize("offset, expected", [
    (0, np.triu(np.ones((3, 3)), k=0)),
    (1, np.triu(np.ones((3, 3)), k=1)),
    (-1, np.triu(np.ones((3, 3)), k=-1)),
])
def test_triu_offset(offset, expected):
    x = api.ones((3, 3))
    arr = api.triu(x, k=offset)
    assert arr.shape == (3, 3)
    assert np.allclose(arr.values, expected.ravel())

@pytest.mark.parametrize("offset, expected", [
    (0, np.tril(np.ones((3, 3)), k=0)),
    (1, np.tril(np.ones((3, 3)), k=1)),
    (-1, np.tril(np.ones((3, 3)), k=-1)),
])
def test_tril_offset(offset, expected):
    x = api.ones((3, 3))
    arr = api.tril(x, k=offset)
    assert arr.shape == (3, 3)
    assert np.allclose(arr.values, expected.ravel())

def test_asarray_and_array():
    data = [[1, 2], [3, 4]]
    arr1 = api.asarray(data)
    arr2 = api.array(data)
    expected = np.array(data)
    assert arr1.shape == (2, 2)
    assert arr2.shape == (2, 2)
    assert np.allclose(arr1.values, expected.ravel())
    assert np.allclose(arr2.values, expected.ravel())
