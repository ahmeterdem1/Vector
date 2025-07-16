from vectorgebra.array import api
import numpy as np
import pytest

api.random.seed(42)

@pytest.fixture
def get_arrays():
    api1 = api.random.randn((2,))
    api2 = api.random.randn((2, 3))
    api3 = api.random.randn((2, 3, 4))
    api4 = api.random.randn((2, 4))

    np1 = api1.values
    np2 = api2.values
    np3 = api3.values
    np4 = api4.values

    np1 = np.asarray(np1).reshape((2,))
    np2 = np.asarray(np2).reshape((2, 3))
    np3 = np.asarray(np3).reshape((2, 3, 4))
    np4 = np.asarray(np4).reshape((2, 4))

    return {"api1": api1,
            "api2": api2,
            "api3": api3,
            "api4": api4,
            "np1": np1,
            "np2": np2,
            "np3": np3,
            "np4": np4}

def test_argmax_basic(get_arrays):
    arr = get_arrays["api2"]  # shape (2, 3)
    expected = np.argmax(get_arrays["np2"])
    result = api.argmax(arr)
    assert result.shape == (0,)
    assert result.values[0] == expected

def test_argmax_axis_0(get_arrays):
    arr = get_arrays["api2"]  # shape (2, 3)
    expected = np.argmax(get_arrays["np2"], axis=0)
    result = api.argmax(arr, axis=0)
    assert result.shape == (3,)
    assert np.allclose(result.values, expected)

def test_argmax_axis_1_keepdims(get_arrays):
    arr = get_arrays["api2"]  # shape (2, 3)
    expected = np.argmax(get_arrays["np2"], axis=1, keepdims=True)
    result = api.argmax(arr, axis=1, keepdims=True)
    assert result.shape == (2, 1)
    assert np.allclose(result.values, expected.ravel())

def test_argmax_middle(get_arrays):
    arr = get_arrays["api3"]  # shape (2, 3, 4)
    expected = np.argmax(get_arrays["np3"], axis=1)
    result = api.argmax(arr, axis=1)
    assert result.shape == (2, 4)
    assert np.allclose(result.values, expected.ravel())

def test_argmax_all_equal():
    arr = api.full((4, 4), 5)
    result = api.argmax(arr)
    assert result.shape == (0,)
    assert result.values[0] == 0

def test_where_basic():
    cond = api.array([[True, False], [False, True]])
    x1 = api.array([[1, 2], [3, 4]])
    x2 = api.array([[5, 6], [7, 8]])
    result = api.where(cond, x1, x2)
    expected = np.where([[True, False], [False, True]],
                        [[1, 2], [3, 4]],
                        [[5, 6], [7, 8]])
    assert result.shape == (2, 2)
    assert np.allclose(result.values, expected.ravel())

def test_where_broadcast():
    cond = api.array([[True], [False]])
    x1 = api.array([[1, 1]])
    x2 = api.array([[2, 2]])
    result = api.where(cond, x1, x2)
    expected = np.where([[True], [False]], [[1, 1]], [[2, 2]])
    assert result.shape == (2, 2)
    assert np.allclose(result.values, expected.ravel())

def test_where_all_scalar():
    cond = api.array([True])
    x1 = api.array([10])
    x2 = api.array([20])
    result = api.where(cond, x1, x2)
    assert result.shape == (1,)
    assert result.values[0] == 10

    cond = api.array([False])
    result = api.where(cond, x1, x2)
    assert result.shape == (1,)
    assert result.values[0] == 20
