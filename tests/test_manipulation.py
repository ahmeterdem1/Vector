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

def test_permute_axes(get_arrays):
    api2 = get_arrays["api2"]
    api3 = get_arrays["api3"]

    np2 = get_arrays["np2"]
    np3 = get_arrays["np3"]

    res_api2 = api.permute_dims(api2, (1, 0))
    res_api3 = api.permute_dims(api3, (0, 2, 1))
    res_api4 = api.permute_dims(api3, (1, 0, 2))
    res_np2 = np.swapaxes(np2, 1, 0)
    res_np3 = np.swapaxes(np3, 1, 2)
    res_np4 = np.transpose(np3, (1, 0, 2))

    assert res_api2.shape == res_np2.shape
    assert res_api3.shape == res_np3.shape
    assert res_api4.shape == res_np4.shape
    assert np.allclose(res_api2.values, res_np2.ravel())
    assert np.allclose(res_api3.values, res_np3.ravel())
    assert np.allclose(res_api4.values, res_np4.ravel())

def test_flip(get_arrays):

    api2 = get_arrays["api2"]
    api3 = get_arrays["api3"]

    np2 = get_arrays["np2"]
    np3 = get_arrays["np3"]

    res_api2_0 = api.flip(api2, axis=None)
    res_api2 = api.flip(api2, axis=1)
    res_api3 = api.flip(api3, axis=1)
    res_api3_1 = api.flip(api3, axis=(1, 2))

    res_np2_0 = np.flip(np2, axis=None)
    res_np2 = np.flip(np2, axis=1)
    res_np3 = np.flip(np3, axis=1)
    res_np3_1 = np.flip(np3, axis=(1, 2))

    assert res_api2_0.shape == res_np2_0.shape
    assert res_api2.shape == res_np2.shape
    assert res_api3.shape == res_np3.shape
    assert res_api3_1.shape == res_np3_1.shape
    assert np.allclose(res_api2_0.values, res_np2_0.ravel())
    assert np.allclose(res_api2.values, res_np2.ravel())
    assert np.allclose(res_api3.values, res_np3.ravel())
    assert np.allclose(res_api3_1.values, res_np3_1.ravel())

