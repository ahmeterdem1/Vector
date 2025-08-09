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

def test_sum_all(get_arrays):
    arr = get_arrays["api2"]
    expected = np.sum(get_arrays["np2"])
    result = api.sum(arr)
    assert result.shape == (0,)
    assert np.allclose(result.values, expected.ravel())

def test_sum_axis(get_arrays):
    arr = get_arrays["api2"]
    expected = np.sum(get_arrays["np2"], axis=1)
    result = api.sum(arr, axis=1)
    assert result.shape == (2,)
    assert np.allclose(result.values, expected.ravel())

def test_mean_all(get_arrays):
    arr = get_arrays["api3"]
    expected = np.mean(get_arrays["np3"])
    result = api.mean(arr)
    assert result.shape == (0,)
    assert np.allclose(result.values, expected.ravel())

def test_mean_axis_tuple(get_arrays):
    arr = get_arrays["api3"]
    expected = np.mean(get_arrays["np3"], axis=(1, 2))
    result = api.mean(arr, axis=(1, 2))
    assert result.shape == (2,)
    assert np.allclose(result.values, expected.ravel())

def test_std_keepdims(get_arrays):
    arr = get_arrays["api2"]
    expected = np.std(get_arrays["np2"], axis=0, keepdims=True)
    result = api.std(arr, axis=0, keepdims=True)
    assert result.shape == (1, 3)
    assert np.allclose(result.values, expected.ravel())

def test_var(get_arrays):
    arr = get_arrays["api2"]
    expected = np.var(get_arrays["np2"], axis=1)
    result = api.var(arr, axis=1)
    assert result.shape == (2,)
    assert np.allclose(result.values, expected.ravel())

def test_max(get_arrays):
    arr = get_arrays["api4"]
    expected = np.max(get_arrays["np4"], axis=1)
    result = api.max(arr, axis=1)
    assert result.shape == (2,)
    assert np.allclose(result.values, expected.ravel())

def test_min(get_arrays):
    arr = get_arrays["api4"]
    expected = np.min(get_arrays["np4"], axis=0)
    result = api.min(arr, axis=0)
    assert result.shape == (4,)
    assert np.allclose(result.values, expected.ravel())
