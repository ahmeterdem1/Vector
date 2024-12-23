"""
    Functions referenced by: https://data-apis.org/array-api/latest/API_specification/statistical_functions.html
"""

from ..ndarray import Array, ArgTypeError
from typing import Type, Union, Tuple
from builtins import sum as __builtinSum, max as __builtinMax, min as __builtinMin
from itertools import product as __product
from statistics import variance as __variance, stdev as __stdev, mean as __mean
from math import prod as __prod

def cumulative_sum():
    raise NotImplementedError()

def max(x: Array, axis: Union[int, Tuple[int]] = None, dtype: Type = None, keepdims: bool = False) -> Array:
    """
        Calculate the maximum value of the given array, along given axis/axes.

        Args:
            x (Array): The array to perform max operation on.

            axis: The axis to calculate the max along. There may be multiple axes
                as a tuple of integers. If left as None, max is calculated over
                all of the array. The default is None.

            dtype (Type): The data type of the array that will be returned. If left
                as None, dtype is inferred. The default is None.

            keepdims (bool): If True, reduced axes are included in the shape of the
                returned array. The default is False.

        Returns:
            Array: The generated array after the max operation.

        Raises:
            ArgTypeError: If given axis does not conform to allowed format.

    """
    res = Array()
    res.device = None

    if axis is None:
        res.values = [__builtinMax(x.values)]
        res.dtype = type(res.values[0]) if dtype is None else dtype
        res.size = 1
        res.ndim = 0
        res.shape = (0,)
        return res

    if isinstance(axis, int):
        res.dtype = x.dtype if dtype is None else dtype
        res.shape = tuple([x.shape[i] for i in range(x.ndim) if i != axis]) if not keepdims else tuple(
            [x.shape[i] if i != axis else 1 for i in range(x.ndim)])

        temp_shape = [range(n) if i != axis else [slice(0, n, 1)] for i, n in enumerate(x.shape)]
        res.size = x.size // x.shape[axis]
        vals = [None] * res.size
        for i, produced_shape in enumerate(__product(*temp_shape)):
            vals[i] = __builtinMax(x[*produced_shape].values)

        res.values = vals
        return res

    if isinstance(axis, tuple):
        res.dtype = x.dtype if dtype is None else dtype
        res.shape = tuple([x.shape[i] for i in range(x.ndim) if i not in axis]) if not keepdims else tuple(
            [x.shape[i] if i not in axis else 1 for i in range(x.ndim)])
        res.ndim = len(res.shape)
        temp_shape = [range(n) if i not in axis else [slice(0, n, 1)] for i, n in enumerate(x.shape)]
        res.size = x.size
        for a in axis:
            res.size //= x.shape[a]
        vals = [None] * res.size
        for i, produced_shape in enumerate(__product(*temp_shape)):
            vals[i] = __builtinMax(x[*produced_shape].values)

        res.values = vals
        return res

    raise ArgTypeError("Argument 'axis' must be one of [None, int, tuple[int]]")

def mean(x: Array, axis: Union[int, Tuple[int]] = None, dtype: Type = None, keepdims: bool = False):
    """
        Calculate the mean of the given array, along given axis/axes.

        Args:
            x (Array): The array to calculate mean on.

            axis: The axis to calculate the mean along. There may be multiple axes
                as a tuple of integers. If left as None, mean is calculated over
                all of the array. The default is None.

            dtype (Type): The data type of the array that will be returned. If left
                as None, dtype is inferred. The default is None.

            keepdims (bool): If True, reduced axes are included in the shape of the
                returned array. The default is False.

        Returns:
            Array: The generated array after the mean calculation.

        Raises:
            ArgTypeError: If given axis does not conform to allowed format.

    """

    res = Array()
    res.device = None

    if axis is None:
        res.values = [__mean(x.values)]
        res.dtype = type(res.values[0]) if dtype is None else dtype
        res.size = 1
        res.ndim = 0
        res.shape = (0,)
        return res

    if isinstance(axis, int):
        res.dtype = x.dtype if dtype is None else dtype
        res.shape = tuple([x.shape[i] for i in range(x.ndim) if i != axis]) if not keepdims else tuple(
            [x.shape[i] if i != axis else 1 for i in range(x.ndim)])

        temp_shape = [range(n) if i != axis else [slice(0, n, 1)] for i, n in enumerate(x.shape)]
        res.size = x.size // x.shape[axis]
        vals = [None] * res.size
        for i, produced_shape in enumerate(__product(*temp_shape)):
            vals[i] = __mean(x[*produced_shape].values)

        res.values = vals
        return res

    if isinstance(axis, tuple):
        res.dtype = x.dtype if dtype is None else dtype
        res.shape = tuple([x.shape[i] for i in range(x.ndim) if i not in axis]) if not keepdims else tuple(
            [x.shape[i] if i not in axis else 1 for i in range(x.ndim)])
        res.ndim = len(res.shape)
        temp_shape = [range(n) if i not in axis else [slice(0, n, 1)] for i, n in enumerate(x.shape)]
        res.size = x.size
        for a in axis:
            res.size //= x.shape[a]
        vals = [None] * res.size
        for i, produced_shape in enumerate(__product(*temp_shape)):
            vals[i] = __mean(x[*produced_shape].values)

        res.values = vals
        return res

    raise ArgTypeError("Argument 'axis' must be one of [None, int, tuple[int]]")

def min(x: Array, axis: Union[int, Tuple[int]] = None, dtype: Type = None, keepdims: bool = False) -> Array:
    """
        Calculate the minimum value of the given array, along given axis/axes.

        Args:
            x (Array): The array to perform min operation on.

            axis: The axis to calculate the min along. There may be multiple axes
                as a tuple of integers. If left as None, min is calculated over
                all of the array. The default is None.

            dtype (Type): The data type of the array that will be returned. If left
                as None, dtype is inferred. The default is None.

            keepdims (bool): If True, reduced axes are included in the shape of the
                returned array. The default is False.

        Returns:
            Array: The generated array after the min operation.

        Raises:
            ArgTypeError: If given axis does not conform to allowed format.

    """

    res = Array()
    res.device = None

    if axis is None:
        res.values = [__builtinMin(x.values)]
        res.dtype = type(res.values[0]) if dtype is None else dtype
        res.size = 1
        res.ndim = 0
        res.shape = (0,)
        return res

    if isinstance(axis, int):
        res.dtype = x.dtype if dtype is None else dtype
        res.shape = tuple([x.shape[i] for i in range(x.ndim) if i != axis]) if not keepdims else tuple(
            [x.shape[i] if i != axis else 1 for i in range(x.ndim)])

        temp_shape = [range(n) if i != axis else [slice(0, n, 1)] for i, n in enumerate(x.shape)]
        res.size = x.size // x.shape[axis]
        vals = [None] * res.size
        for i, produced_shape in enumerate(__product(*temp_shape)):
            vals[i] = __builtinMin(x[*produced_shape].values)

        res.values = vals
        return res

    if isinstance(axis, tuple):
        res.dtype = x.dtype if dtype is None else dtype
        res.shape = tuple([x.shape[i] for i in range(x.ndim) if i not in axis]) if not keepdims else tuple(
            [x.shape[i] if i not in axis else 1 for i in range(x.ndim)])
        res.ndim = len(res.shape)
        temp_shape = [range(n) if i not in axis else [slice(0, n, 1)] for i, n in enumerate(x.shape)]
        res.size = x.size
        for a in axis:
            res.size //= x.shape[a]
        vals = [None] * res.size
        for i, produced_shape in enumerate(__product(*temp_shape)):
            vals[i] = __builtinMin(x[*produced_shape].values)

        res.values = vals
        return res

    raise ArgTypeError("Argument 'axis' must be one of [None, int, tuple[int]]")

def prod(x: Array, axis: Union[int, Tuple[int]] = None, dtype: Type = None, keepdims: bool = False):
    """
        Calculate the product of values in the given array, along given axis/axes.

        Args:
            x (Array): The array to calculate the elementary product.

            axis: The axis to calculate the product along. There may be multiple axes
                as a tuple of integers. If left as None, product is calculated over
                all of the array. The default is None.

            dtype (Type): The data type of the array that will be returned. If left
                as None, dtype is inferred. The default is None.

            keepdims (bool): If True, reduced axes are included in the shape of the
                returned array. The default is False.

        Returns:
            Array: The generated array after the product.

        Raises:
            ArgTypeError: If given axis does not conform to allowed format.

    """

    res = Array()
    res.device = None

    if axis is None:
        res.values = [__prod(x.values)]
        res.dtype = type(res.values[0]) if dtype is None else dtype
        res.size = 1
        res.ndim = 0
        res.shape = (0,)
        return res

    if isinstance(axis, int):
        res.dtype = x.dtype if dtype is None else dtype
        res.shape = tuple([x.shape[i] for i in range(x.ndim) if i != axis]) if not keepdims else tuple(
            [x.shape[i] if i != axis else 1 for i in range(x.ndim)])

        temp_shape = [range(n) if i != axis else [slice(0, n, 1)] for i, n in enumerate(x.shape)]
        res.size = x.size // x.shape[axis]
        vals = [None] * res.size
        for i, produced_shape in enumerate(__product(*temp_shape)):
            vals[i] = __prod(x[*produced_shape].values)

        res.values = vals
        return res

    if isinstance(axis, tuple):
        res.dtype = x.dtype if dtype is None else dtype
        res.shape = tuple([x.shape[i] for i in range(x.ndim) if i not in axis]) if not keepdims else tuple(
            [x.shape[i] if i not in axis else 1 for i in range(x.ndim)])
        res.ndim = len(res.shape)
        temp_shape = [range(n) if i not in axis else [slice(0, n, 1)] for i, n in enumerate(x.shape)]
        res.size = x.size
        for a in axis:
            res.size //= x.shape[a]
        vals = [None] * res.size
        for i, produced_shape in enumerate(__product(*temp_shape)):
            vals[i] = __prod(x[*produced_shape].values)

        res.values = vals
        return res

    raise ArgTypeError("Argument 'axis' must be one of [None, int, tuple[int]]")

def std(x: Array, axis: Union[int, Tuple[int]] = None, dtype: Type = None, keepdims: bool = False):
    """
        Calculate the standard deviation of the given array, along given axis/axes.

        Args:
            x (Array): The array to calculate standard deviation on.

            axis: The axis to calculate the standard deviation along. There may be multiple axes
                as a tuple of integers. If left as None, standard deviation is calculated over
                all of the array. The default is None.

            dtype (Type): The data type of the array that will be returned. If left
                as None, dtype is inferred. The default is None.

            keepdims (bool): If True, reduced axes are included in the shape of the
                returned array. The default is False.

        Returns:
            Array: The generated array after the standard deviation calculation.

        Raises:
            ArgTypeError: If given axis does not conform to allowed format.

    """
    res = Array()
    res.device = None

    if axis is None:
        res.values = [__stdev(x.values)]
        res.dtype = type(res.values[0]) if dtype is None else dtype
        res.size = 1
        res.ndim = 0
        res.shape = (0,)
        return res

    if isinstance(axis, int):
        res.dtype = x.dtype if dtype is None else dtype
        res.shape = tuple([x.shape[i] for i in range(x.ndim) if i != axis]) if not keepdims else tuple(
            [x.shape[i] if i != axis else 1 for i in range(x.ndim)])

        temp_shape = [range(n) if i != axis else [slice(0, n, 1)] for i, n in enumerate(x.shape)]
        res.size = x.size // x.shape[axis]
        vals = [None] * res.size
        for i, produced_shape in enumerate(__product(*temp_shape)):
            vals[i] = __stdev(x[*produced_shape].values)

        res.values = vals
        return res

    if isinstance(axis, tuple):
        res.dtype = x.dtype if dtype is None else dtype
        res.shape = tuple([x.shape[i] for i in range(x.ndim) if i not in axis]) if not keepdims else tuple(
            [x.shape[i] if i not in axis else 1 for i in range(x.ndim)])
        res.ndim = len(res.shape)
        temp_shape = [range(n) if i not in axis else [slice(0, n, 1)] for i, n in enumerate(x.shape)]
        res.size = x.size
        for a in axis:
            res.size //= x.shape[a]
        vals = [None] * res.size
        for i, produced_shape in enumerate(__product(*temp_shape)):
            vals[i] = __stdev(x[*produced_shape].values)

        res.values = vals
        return res

    raise ArgTypeError("Argument 'axis' must be one of [None, int, tuple[int]]")

def sum(x: Array, axis: Union[int, Tuple[int]] = None, dtype: Type = None, keepdims: bool = False) -> Array:
    """
        Calculate the sum of the given array, along given axis/axes.

        Args:
            x (Array): The array to calculate sum on.

            axis: The axis to calculate the sum along. There may be multiple axes
                as a tuple of integers. If left as None, sum is calculated over
                all of the array. The default is None.

            dtype (Type): The data type of the array that will be returned. If left
                as None, dtype is inferred. The default is None.

            keepdims (bool): If True, reduced axes are included in the shape of the
                returned array. The default is False.

        Returns:
            Array: The generated array after the sum calculation.

        Raises:
            ArgTypeError: If given axis does not conform to allowed format.

    """

    res = Array()
    res.device = None

    if axis is None:
        res.values = [__builtinSum(x.values)]
        res.dtype = type(res.values[0]) if dtype is None else dtype
        res.size = 1
        res.ndim = 0
        res.shape = (0,)
        return res

    if isinstance(axis, int):
        res.dtype = x.dtype if dtype is None else dtype
        res.shape = tuple([x.shape[i] for i in range(x.ndim) if i != axis]) if not keepdims else tuple(
            [x.shape[i] if i != axis else 1 for i in range(x.ndim)])

        temp_shape = [range(n) if i != axis else [slice(0, n, 1)] for i, n in enumerate(x.shape)]
        res.size = x.size // x.shape[axis]
        vals = [None] * res.size
        for i, produced_shape in enumerate(__product(*temp_shape)):
            vals[i] = __builtinSum(x[*produced_shape].values)

        res.values = vals
        return res

    if isinstance(axis, tuple):
        res.dtype = x.dtype if dtype is None else dtype
        res.shape = tuple([x.shape[i] for i in range(x.ndim) if i not in axis]) if not keepdims else tuple(
            [x.shape[i] if i not in axis else 1 for i in range(x.ndim)])
        res.ndim = len(res.shape)
        temp_shape = [range(n) if i not in axis else [slice(0, n, 1)] for i, n in enumerate(x.shape)]
        res.size = x.size
        for a in axis:
            res.size //= x.shape[a]
        vals = [None] * res.size
        for i, produced_shape in enumerate(__product(*temp_shape)):
            vals[i] = __builtinSum(x[*produced_shape].values)

        res.values = vals
        return res

    raise ArgTypeError("Argument 'axis' must be one of [None, int, tuple[int]]")

def var(x: Array, axis: Union[int, Tuple[int]] = None, dtype: Type = None, keepdims: bool = False):
    """
        Calculate the variance of the given array, along given axis/axes.

        Args:
            x (Array): The array to calculate variance on.

            axis: The axis to calculate the variance along. There may be multiple axes
                as a tuple of integers. If left as None, variance is calculated over
                all of the array. The default is None.

            dtype (Type): The data type of the array that will be returned. If left
                as None, dtype is inferred. The default is None.

            keepdims (bool): If True, reduced axes are included in the shape of the
                returned array. The default is False.

        Returns:
            Array: The generated array after the variance calculation.

        Raises:
            ArgTypeError: If given axis does not conform to allowed format.

    """

    res = Array()
    res.device = None

    if axis is None:
        res.values = [__variance(x.values)]
        res.dtype = type(res.values[0]) if dtype is None else dtype
        res.size = 1
        res.ndim = 0
        res.shape = (0,)
        return res

    if isinstance(axis, int):
        res.dtype = x.dtype if dtype is None else dtype
        res.shape = tuple([x.shape[i] for i in range(x.ndim) if i != axis]) if not keepdims else tuple(
            [x.shape[i] if i != axis else 1 for i in range(x.ndim)])

        temp_shape = [range(n) if i != axis else [slice(0, n, 1)] for i, n in enumerate(x.shape)]
        res.size = x.size // x.shape[axis]
        vals = [None] * res.size
        for i, produced_shape in enumerate(__product(*temp_shape)):
            vals[i] = __variance(x[*produced_shape].values)

        res.values = vals
        return res

    if isinstance(axis, tuple):
        res.dtype = x.dtype if dtype is None else dtype
        res.shape = tuple([x.shape[i] for i in range(x.ndim) if i not in axis]) if not keepdims else tuple(
            [x.shape[i] if i not in axis else 1 for i in range(x.ndim)])
        res.ndim = len(res.shape)
        temp_shape = [range(n) if i not in axis else [slice(0, n, 1)] for i, n in enumerate(x.shape)]
        res.size = x.size
        for a in axis:
            res.size //= x.shape[a]
        vals = [None] * res.size
        for i, produced_shape in enumerate(__product(*temp_shape)):
            vals[i] = __variance(x[*produced_shape].values)

        res.values = vals
        return res

    raise ArgTypeError("Argument 'axis' must be one of [None, int, tuple[int]]")

