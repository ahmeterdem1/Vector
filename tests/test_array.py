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

def test_reshape(get_arrays):
    api1 = get_arrays["api3"]
    np1 = get_arrays["np3"]

    api1.reshape((2, 4, 3))
    reshaped_np1 = np1.reshape((2, 4, 3))

    assert api1.shape == (2, 4, 3)
    assert np.array_equal(api1.values, reshaped_np1.ravel())

def test_transpose(get_arrays):
    api1 = get_arrays["api1"]
    api2 = get_arrays["api2"]
    api3 = get_arrays["api3"]
    np1 = get_arrays["np1"]
    np2 = get_arrays["np2"]
    np3 = get_arrays["np3"]

    res_api1 = api1.T
    res_api2 = api2.T
    res_api3 = api3.T
    res_np1 = np1.T
    res_np2 = np2.T
    res_np3 = np3.T

    assert res_api1.shape == res_np1.shape
    assert np.array_equal(res_api1.values, res_np1.ravel())
    assert res_api2.shape == res_np2.shape
    assert np.array_equal(res_api2.values, res_np2.ravel())
    assert res_api3.shape == res_np3.shape
    assert np.array_equal(res_api3.values, res_np3.ravel())

def test_matrix_transpose(get_arrays):
    api1 = get_arrays["api2"]
    api2 = get_arrays["api3"]
    np1 = get_arrays["np2"]
    np2 = get_arrays["np3"]

    res_api1 = api1.mT
    res_api2 = api2.mT
    res_np1 = np.swapaxes(np1, -1, -2)
    res_np2 = np.swapaxes(np2, -1, -2)

    assert res_api1.shape == res_np1.shape
    assert np.array_equal(res_api1.values, res_np1.ravel())
    assert res_api2.shape == res_np2.shape
    assert np.array_equal(res_api2.values, res_np2.ravel())

def test_matmul(get_arrays):
    api1 = get_arrays["api1"]
    api2 = get_arrays["api2"]
    api3 = get_arrays["api3"]
    api4 = get_arrays["api4"]

    np1 = get_arrays["np1"]
    np2 = get_arrays["np2"]
    np3 = get_arrays["np3"]
    np4 = get_arrays["np4"]

    res_api = api2 @ api2.T  # mat x mat
    res_api2 = api4 @ api4.T  # tensor x tensor
    res_api3 = api3 @ api4.T  # tensor x mat
    res_api4 = api2.T @ api1  # (3,)  mat x vector

    res_np = np2 @ np2.T
    res_np2 = np4 @ np4.T
    res_np3 = np.einsum("abc,cd->abd", np3, np4.T)
    res_np4 = np2.T @ np1  # (3,)

    assert res_api.shape == (2, 2) and res_api2.shape == (2, 2) and res_api3.shape == (2, 3, 2)
    assert res_np.shape == (2, 2) and res_np2.shape == (2, 2) and res_np3.shape == (2, 3, 2)
    assert np.all(np.isclose(res_api.values, res_np.ravel()))
    assert np.all(np.isclose(res_api2.values, res_np2.ravel()))
    assert np.all(np.isclose(res_api3.values, res_np3.ravel()))
    assert np.all(np.isclose(res_api4.values, res_np4.ravel()))

def test_broadcast(get_arrays):
    api1 = get_arrays["api1"]
    api2 = get_arrays["api2"]
    api3 = get_arrays["api3"]

    np1 = get_arrays["np1"]
    np2 = get_arrays["np2"]
    np3 = get_arrays["np3"]

    api1.reshape((2, 1, 1))
    api2.reshape((2, 3, 1))

    np1 = np1.reshape((2, 1, 1))
    np2 = np2.reshape((2, 3, 1))

    res_api1 = api1.T + api2.T
    res_api2 = api2 + api3
    res_api3 = api1 + api3

    res_np1 = np1.T + np2.T
    res_np2 = np2 + np3
    res_np3 = np1 + np3

    assert res_api1.shape == (1, 3, 2) and res_api2.shape == (2, 3, 4) and res_api3.shape == (2, 3, 4)
    assert np.array_equal(res_api1.values, res_np1.ravel())
    assert np.array_equal(res_api2.values, res_np2.ravel())
    assert np.array_equal(res_api3.values, res_np3.ravel())
