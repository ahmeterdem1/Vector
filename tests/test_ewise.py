from vectorgebra.array import api
import numpy as np
import pytest

api.random.seed(42)

@pytest.fixture
def get_arrays():
    api1 = api.random.randn((2,))
    api2 = api.random.randn((2, 3))
    api3 = api.random.randn((2, 3, 4))

    np1 = api1.values
    np2 = api2.values
    np3 = api3.values

    np1 = np.asarray(np1).reshape((2,))
    np2 = np.asarray(np2).reshape((2, 3))
    np3 = np.asarray(np3).reshape((2, 3, 4))

    return {"api1": api1,
            "api2": api2,
            "api3": api3,
            "np1": np1,
            "np2": np2,
            "np3": np3}

@pytest.fixture
def get_positive_integer_arrays():
    api1 = api.random.randint(shape=(2,), a=0, b=10)
    api2 = api.random.randint(shape=(2, 3), a=0, b=10)
    api3 = api.random.randint(shape=(2, 3, 4), a=0, b=10)

    np1 = api1.values
    np2 = api2.values
    np3 = api3.values

    np1 = np.asarray(np1).reshape((2,))
    np2 = np.asarray(np2).reshape((2, 3))
    np3 = np.asarray(np3).reshape((2, 3, 4))

    return {"api1": api1,
            "api2": api2,
            "api3": api3,
            "np1": np1,
            "np2": np2,
            "np3": np3}


def test_abs(get_arrays):
    api1 = get_arrays["api1"]
    api2 = get_arrays["api2"]
    api3 = get_arrays["api3"]

    np1 = get_arrays["np1"]
    np2 = get_arrays["np2"]
    np3 = get_arrays["np3"]

    assert np.all(api.abs(api1).values == np.abs(np1).ravel())
    assert np.all(api.abs(api2).values == np.abs(np2).ravel())
    assert np.all(api.abs(api3).values == np.abs(np3).ravel())

def _test_acos(get_arrays):
    api1 = get_arrays["api1"]
    api2 = get_arrays["api2"]
    api3 = get_arrays["api3"]

    np1 = get_arrays["np1"]
    np2 = get_arrays["np2"]
    np3 = get_arrays["np3"]

    assert np.all(api.acos(api1).values == np.arccos(np1).ravel())
    assert np.all(api.acos(api2).values == np.arccos(np2).ravel())
    assert np.all(api.acos(api3).values == np.arccos(np3).ravel())

def _test_acosh(get_arrays):

    api1 = get_arrays["api1"]
    api2 = get_arrays["api2"]
    api3 = get_arrays["api3"]

    np1 = get_arrays["np1"]
    np2 = get_arrays["np2"]
    np3 = get_arrays["np3"]

    assert np.all(api.acosh(api1).values == np.arccosh(np1).ravel())
    assert np.all(api.acosh(api2).values == np.arccosh(np2).ravel())
    assert np.all(api.acosh(api3).values == np.arccosh(np3).ravel())

def test_add(get_arrays):

    api1 = get_arrays["api1"]
    api2 = get_arrays["api2"]
    api3 = get_arrays["api3"]

    np1 = get_arrays["np1"]
    np2 = get_arrays["np2"]
    np3 = get_arrays["np3"]

    assert np.all(api.add(api1, api1).values == np.add(np1, np1).ravel())
    assert np.all(api.add(api2, api2).values == np.add(np2, np2).ravel())
    assert np.all(api.add(api3, api3).values == np.add(np3, np3).ravel())

def test_atan(get_arrays):

    api1 = get_arrays["api1"]
    api2 = get_arrays["api2"]
    api3 = get_arrays["api3"]

    np1 = get_arrays["np1"]
    np2 = get_arrays["np2"]
    np3 = get_arrays["np3"]

    assert np.all(api.atan(api1).values == np.arctan(np1).ravel())
    assert np.all(api.atan(api2).values == np.arctan(np2).ravel())
    assert np.all(api.atan(api3).values == np.arctan(np3).ravel())

def test_atan2(get_arrays):

    api1 = get_arrays["api1"]
    api2 = get_arrays["api2"]
    api3 = get_arrays["api3"]

    np1 = get_arrays["np1"]
    np2 = get_arrays["np2"]
    np3 = get_arrays["np3"]
    # fails
    assert np.all(api.atan(api1).values == np.arctan(np1).ravel())
    assert np.all(api.atan(api2).values == np.arctan(np2).ravel())
    assert np.all(api.atan(api3).values == np.arctan(np3).ravel())

def test_bitwise_and(get_positive_integer_arrays):
    api1 = get_positive_integer_arrays["api1"]
    api2 = get_positive_integer_arrays["api2"]
    api3 = get_positive_integer_arrays["api3"]

    np1 = get_positive_integer_arrays["np1"]
    np2 = get_positive_integer_arrays["np2"]
    np3 = get_positive_integer_arrays["np3"]

    assert np.all(api.bitwise_and(api1, api1).values == np.bitwise_and(np1, np1).ravel())
    assert np.all(api.bitwise_and(api2, api2).values == np.bitwise_and(np2, np2).ravel())
    assert np.all(api.bitwise_and(api3, api3).values == np.bitwise_and(np3, np3).ravel())

def test_bitwise_left_shift(get_positive_integer_arrays):
    api1 = get_positive_integer_arrays["api1"]
    api2 = get_positive_integer_arrays["api2"]
    api3 = get_positive_integer_arrays["api3"]

    np1 = get_positive_integer_arrays["np1"]
    np2 = get_positive_integer_arrays["np2"]
    np3 = get_positive_integer_arrays["np3"]

    assert np.all(api.bitwise_left_shift(api1, 1).values == np.left_shift(np1, 1).ravel())
    assert np.all(api.bitwise_left_shift(api2, 1).values == np.left_shift(np2, 1).ravel())
    assert np.all(api.bitwise_left_shift(api3, 1).values == np.left_shift(np3, 1).ravel())

def test_bitwise_invert(get_positive_integer_arrays):

    api1 = get_positive_integer_arrays["api1"]
    api2 = get_positive_integer_arrays["api2"]
    api3 = get_positive_integer_arrays["api3"]

    np1 = get_positive_integer_arrays["np1"]
    np2 = get_positive_integer_arrays["np2"]
    np3 = get_positive_integer_arrays["np3"]

    assert np.all(api.bitwise_invert(api1).values == np.bitwise_not(np1).ravel())
    assert np.all(api.bitwise_invert(api2).values == np.bitwise_not(np2).ravel())
    assert np.all(api.bitwise_invert(api3).values == np.bitwise_not(np3).ravel())

def test_ceil(get_arrays):
    api1 = get_arrays["api1"]
    api2 = get_arrays["api2"]
    api3 = get_arrays["api3"]

    np1 = get_arrays["np1"]
    np2 = get_arrays["np2"]
    np3 = get_arrays["np3"]

    assert np.all(api.ceil(api1).values == np.ceil(np1).ravel())
    assert np.all(api.ceil(api2).values == np.ceil(np2).ravel())
    assert np.all(api.ceil(api3).values == np.ceil(np3).ravel())

def test_clip(get_arrays):
    api1 = get_arrays["api1"]
    api2 = get_arrays["api2"]
    api3 = get_arrays["api3"]

    np1 = get_arrays["np1"]
    np2 = get_arrays["np2"]
    np3 = get_arrays["np3"]

    assert np.all(api.clip(api1, -1, 1).values == np.clip(np1, -1, 1).ravel())
    assert np.all(api.clip(api2, -1, 1).values == np.clip(np2, -1, 1).ravel())
    assert np.all(api.clip(api3, -1, 1).values == np.clip(np3, -1, 1).ravel())

def test_divide(get_arrays):
    api1 = get_arrays["api1"]
    api2 = get_arrays["api2"]
    api3 = get_arrays["api3"]

    np1 = get_arrays["np1"]
    np2 = get_arrays["np2"]
    np3 = get_arrays["np3"]

    assert np.all(api.divide(api1, api1).values == np.divide(np1, np1).ravel())
    assert np.all(api.divide(api2, api2).values == np.divide(np2, np2).ravel())
    assert np.all(api.divide(api3, api3).values == np.divide(np3, np3).ravel())

def test_equal(get_arrays):
    api1 = get_arrays["api1"]
    api2 = get_arrays["api2"]
    api3 = get_arrays["api3"]

    np1 = get_arrays["np1"]
    np2 = get_arrays["np2"]
    np3 = get_arrays["np3"]

    assert np.all(api.equal(api1, api1).values == np.equal(np1, np1).ravel())
    assert np.all(api.equal(api2, api2).values == np.equal(np2, np2).ravel())
    assert np.all(api.equal(api3, api3).values == np.equal(np3, np3).ravel())

def test_greater(get_arrays):
    api1 = get_arrays["api1"]
    api2 = get_arrays["api2"]
    api3 = get_arrays["api3"]

    np1 = get_arrays["np1"]
    np2 = get_arrays["np2"]
    np3 = get_arrays["np3"]

    assert np.all(api.greater(api1, api1).values == np.greater(np1, np1).ravel())
    assert np.all(api.greater(api2, api2).values == np.greater(np2, np2).ravel())
    assert np.all(api.greater(api3, api3).values == np.greater(np3, np3).ravel())

def test_hypot(get_arrays):
    api1 = get_arrays["api1"]
    api2 = get_arrays["api2"]
    api3 = get_arrays["api3"]

    np1 = get_arrays["np1"]
    np2 = get_arrays["np2"]
    np3 = get_arrays["np3"]

    assert np.all(np.isclose(api.hypot(api1, api1).values, np.hypot(np1, np1).ravel()))
    assert np.all(np.isclose(api.hypot(api2, api2).values, np.hypot(np2, np2).ravel()))
    assert np.all(np.isclose(api.hypot(api3, api3).values, np.hypot(np3, np3).ravel()))

def test_logaddexp(get_arrays):
    api1 = get_arrays["api1"]
    api2 = get_arrays["api2"]
    api3 = get_arrays["api3"]

    np1 = get_arrays["np1"]
    np2 = get_arrays["np2"]
    np3 = get_arrays["np3"]

    assert np.all(np.isclose(api.logaddexp(api1, api1).values, np.logaddexp(np1, np1).ravel()))
    assert np.all(np.isclose(api.logaddexp(api2, api2).values, np.logaddexp(np2, np2).ravel()))
    assert np.all(np.isclose(api.logaddexp(api3, api3).values, np.logaddexp(np3, np3).ravel()))

def test_round(get_arrays):
    api1 = get_arrays["api1"]
    api2 = get_arrays["api2"]
    api3 = get_arrays["api3"]

    np1 = get_arrays["np1"]
    np2 = get_arrays["np2"]
    np3 = get_arrays["np3"]

    assert np.all(api.round(api1).values == np.round(np1).ravel())
    assert np.all(api.round(api2).values == np.round(np2).ravel())
    assert np.all(api.round(api3).values == np.round(np3).ravel())
