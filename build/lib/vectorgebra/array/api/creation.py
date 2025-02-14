"""
    Functions referenced by https://data-apis.org/array-api/latest/API_specification/creation_functions.html
"""

from .dtype import BASIC_ITERABLE
from ..ndarray import Array
from .. import Range, DimensionError
from typing import Union, Type
from decimal import Decimal
from copy import deepcopy, copy

def arange(start: Union[int, float, Decimal],
           stop: Union[int, float, Decimal] = None,
           step: Union[int, float, Decimal] = 1,
           dtype: Type = None, device=None) -> Array:
    """
        Returns one dimensional array of dtype
        elements in range defined by start, stop
        and step.

        Args:
            start (Union[int, float, Decimal]): The start of the range.
                If rest of the arguments are omitted, this will act as
                the end of the range, and (0, start, 1) range will be
                returned.

            stop (Union[int, float, Decimal]): The end of the range.
                The default value is None, therefore it may be omitted
                for argument "start" to function as "stop".

            step (Union[int, float, Decimal]): Step of the range.

            dtype (Union[int, float, Decimal]): An allowed dtype by
                Vectorgebra.

            device (Union[int, float, Decimal]): Currently, not implemented.
                Exists for compatibility with Array API.

        Returns:
            The range as a 1 dimensional Array object of dtype.

        Raises:
            ArgTypeError: If start and stop are both given, but not numerical.

            RangeError: If start and stop are equal, or the direction of the range
                is ambiguous.

            (See vectorgebra.math.functions.Range)
    """
    res = Array()
    res.dtype = dtype if dtype is not None else type(step)
    res.values = res.values = [res.dtype(k) for k in Range(start, stop, step)] if stop is not None else [res.dtype(k) for k in Range(0, start, step)]
    res.device = device
    res.shape = (len(res.values),)
    res.ndim = 1
    res.size = res.shape[0]
    return res

def asarray(obj, dtype: Type = None, device=None, copy: bool = False) -> Array:
    """
        Returns arrayified version of obj. This implementation slightly
        deviates from original Array API for the "copy" argument.

        Args:
            obj: Actual Python object to be casted as array.

            dtype (Type): Data type to cast to. Defaults to None, if
                left as such, dtype is inferred.

            device: Exist for compatibility with Array API, not implemented.

            copy (bool): If True, given object is deepcopied. If false, it
                is not deepcopied, explicitly. An object is directly given
                to the Array constructor, where the data passes through
                Array.flatten static method. This method implicitly copies
                the given object.

        Returns:
             Array: Object casted as array

    """

    return Array(obj if not copy else deepcopy(obj), dtype, device)

def empty(shape: BASIC_ITERABLE, dtype: Type = None, device=None) -> Array:
    """
        Returns a Python style uninitialized Array given the shape and
        dtype. No matter the dtype, for it to count as "uninitialized",
        *values* attribute of the Array is filled with None.

        Args:
             shape (Union[list, tuple]): Shape of the array

             dtype (Type): Data type for the array to have. Defaults to None.
                If left as such, returned array will have a dtype of base Python
                float.

            device: Exist for compatibility with Array API, not implemented.

        Returns:
            Array: Uninitialized array with given shape and dtype attributes.
    """
    res = Array()
    n = 1
    for k in shape:
        n *= k
    res.values = [None] * n
    res.dtype = dtype if dtype is not None else float
    res.device = device
    res.shape = copy(shape)
    res.ndim = len(res.shape)
    res.size = n
    return res

def empty_like(x: Array, dtype: Type = None, device=None) -> Array:
    """
        Create an uninitialized array with the same shape as
        given array.

        Args:
             x (Array): Array object for the shape to be inferred
                from.

            dtype (Type): Data type for the array to have. Defaults to
                None. If left as such, returned array will have a dtype
                of base Python float.

            device: Exist for compatibility with Array API, not implemented.

        Returns:
            Array: Uninitialized array with given shape and dtype attributes.
    """
    res = Array()
    res.values = [None] * x.size
    res.dtype = dtype if dtype is not None else float
    res.device = device
    res.shape = copy(x.shape)
    res.ndim = x.ndim
    res.size = x.size
    return res

def eye(n_rows: int, n_cols: int, k: int = 0, dtype: Type = None, device=None) -> Array:
    """
        Create an identity matrix, given dimensions and diagonal offset.

        Args:
            n_rows (int): Number of rows.

            n_cols (int): Number of columns.

            k (int): Diagonal offset. Defaults to 0. Positive values mean
                upper diagonal, and negative values mean lower diagonal.

            dtype (Type): Data type for the array to have. Defaults to
                None. If left as such, returned array will have a dtype
                of base Python float.

            device: Exist for compatibility with Array API, not implemented.

        Returns:
            Array: Identity matrix as 2 dimensional Array object, with given
                diagonal offset.
    """

    res = Array()
    res.size = n_rows * n_cols
    res.dtype = dtype if dtype is not None else float
    res.values = [res.dtype(0)] * res.size
    offset: int
    ONE = res.dtype(1)
    for i in range(n_rows):
        offset = i + k
        if 0 <= offset < n_cols:
            res.values[i * n_rows + offset] = ONE
    res.shape = (n_rows, n_cols,)
    res.ndim = 2
    res.device = device
    return res



def from_dlpack():
    raise NotImplementedError()

def full(shape: BASIC_ITERABLE, fill_value, dtype: Type = None, device=None) -> Array:
    """
        Create a filled array given shape and the value to fill the array with.

        Args:
            shape (Union[list, tuple]): Shape of the array

            fill_value: The value to fill the array with. When filling,
                will be casted to given dtype.

            dtype (Type): Data type to cast to. Defaults to None, if
                left as such, dtype is inferred.

            device: Exist for compatibility with Array API, not implemented.

        Returns:
            Array: Array filled with proper value and data type, with proper shape.
    """

    res = Array()
    n = 1
    for k in shape:
        n *= k
    res.dtype = dtype if dtype is not None else float
    res.values = [res.dtype(fill_value)] * n
    res.device = device
    res.shape = copy(shape)
    res.ndim = len(shape)
    res.size = n
    return res

def full_like(x: Array, fill_value, dtype: Type = None, device=None):
    """
        Create a full array with the same shape as given array, and
        with given fill value.

        Args:
            x (Array): Array object for the shape to be inferred
                from.

            fill_value: The value to fill the array with. When filling,
                will be casted to given dtype.

            dtype (Type): Data type for the array to have. Defaults to
                None. If left as such, returned array will have a dtype
                of base Python float.

            device: Exist for compatibility with Array API, not implemented.

        Returns:
            Array: Full array with given shape and dtype attributes, with the
                fill value.
        """
    res = Array()
    res.dtype = dtype if dtype is not None else float
    res.values = [res.dtype(fill_value)] * x.size
    res.device = device
    res.shape = copy(x.shape)
    res.ndim = x.ndim
    res.size = x.size
    return res

def linspace(start, stop, num: int, dtype: Type = None, device=None, endpoint: bool = True) -> Array:
    """
        Create evenly spaced numbers as an Array, over the given interval.
        This function does not implement complex ranges.

        However, vectorgebra.math.complex, has an equivariant complex numbered
        range implementation.

        This function, due to floating point errors, may not be %100 consistent.

        Args:
            start: The start of the range.

            stop: The end of the range.

            num (int): Number of elements to be in the range.

            dtype (Type): Data type for the array to have. Defaults to
                None. If left as such, returned array will have a dtype
                of base Python float.

            device: Exist for compatibility with Array API, not implemented.

            endpoint (bool): The choice to include "stop" in the Array or not.
                Defaults to True.

        Returns:
            Array: The generated value-array with vectorgebra.math.functions.Range.

    """
    dR = (stop - start) / (num - 1)
    res = Array()
    res.dtype = dtype if dtype is not None else float
    if endpoint:
        res.values = [res.dtype(k) for k in Range(start, stop + dR / 2, dR)]
    else:
        res.values = [res.dtype(k) for k in Range(start, stop, dR)]
    res.device = device
    res.shape = (len(res.values),)
    res.ndim = 1
    res.size = res.shape[0]
    return res


def meshgrid(*arrays, indexing: str = "xy"):
    pass

def ones(shape: BASIC_ITERABLE, dtype: Type = None, device=None):
    """
        Create a 1-filled array given shape.

        Args:
            shape (Union[list, tuple]): Shape of the array

            dtype (Type): Data type to cast to. Defaults to None, if
                left as such, dtype is inferred.

            device: Exist for compatibility with Array API, not implemented.

        Returns:
            Array: Array filled with 1 and data type, with proper shape.
    """

    res = Array()
    n = 1
    for k in shape:
        n *= k
    res.dtype = dtype if dtype is not None else float
    res.values = [res.dtype(1)] * n
    res.device = device
    res.shape = copy(shape)
    res.ndim = len(shape)
    res.size = n
    return res

def ones_like(x: Array, dtype: Type = None, device=None):
    """
        Create a 1-filled array with the same shape as given array.

        Args:
            x (Array): Array object for the shape to be inferred
                from.

            dtype (Type): Data type for the array to have. Defaults to
                None. If left as such, returned array will have a dtype
                of base Python float.

            device: Exist for compatibility with Array API, not implemented.

        Returns:
            Array: 1-filled Array with given shape and dtype attributes..
    """
    res = Array()
    res.dtype = dtype if dtype is not None else float
    res.values = [res.dtype(1)] * x.size
    res.device = device
    res.shape = copy(x.shape)
    res.ndim = x.ndim
    res.size = x.size
    return res

def tril(x: Array, k: int = 0):
    """
        Return the lower triangular matrix, of a given matrix or a stack of matrices.

        Args:
            x (Array): Array having a shape (..., M, N).

            k (int): Diagonal offset to calculate the lower portion of.

        Returns:
             Array: Lower triangular matrices generated from x. All other attributes
                than *values* are numerically the same.

        Raises:
            DimensionError: If ndim of x is lower than 2.
    """
    if x.ndim > 2:
        new_values = []
        for i in range(x.shape[0]):
            new_values.extend(tril(x[i], k).values)  # Not the best implementation
        res = Array()
        res.shape = copy(x.shape)
        res.values = new_values
        res.ndim = x.ndim
        res.size = x.size
        res.dtype = x.dtype
        res.device = x.device
        return res
    elif x.ndim == 2:

        offset: int
        c_values = copy(x.values)
        ZERO = x.dtype(0)
        for i in range(x.shape[0]):
            offset = i + k + 1
            for j in range(offset, x.shape[1]):
                c_values[i * x.shape[0] + j] = ZERO
        res = Array()
        res.shape = copy(x.shape)
        res.values = c_values
        res.ndim = x.ndim
        res.size = x.size
        res.dtype = x.dtype
        res.device = x.device
        return res
    else:
        raise DimensionError(0)

def triu(x: Array, k: int = 0):
    """
            Return the lower triangular matrix, of a given matrix or a stack of matrices.

            Args:
                x (Array): Array having a shape (..., M, N).

                k (int): Diagonal offset to calculate the lower portion of.

            Returns:
                 Array: Lower triangular matrices generated from x. All other attributes
                    than *values* are numerically the same.

            Raises:
                DimensionError: If ndim of x is lower than 2.
        """
    if x.ndim > 2:
        new_values = []
        for i in range(x.shape[0]):
            new_values.extend(tril(x[i], k).values)  # Not the best implementation
        res = Array()
        res.shape = copy(x.shape)
        res.values = new_values
        res.ndim = x.ndim
        res.size = x.size
        res.dtype = x.dtype
        res.device = x.device
        return res
    elif x.ndim == 2:

        offset: int
        c_values = copy(x.values)
        ZERO = x.dtype(0)
        for i in range(x.shape[0]):
            offset = i + k
            for j in range(offset):
                c_values[i * x.shape[0] +  j] = ZERO
        res = Array()
        res.shape = copy(x.shape)
        res.values = c_values
        res.ndim = x.ndim
        res.size = x.size
        res.dtype = x.dtype
        res.device = x.device
        return res
    else:
        raise DimensionError(0)

def zeros(shape: BASIC_ITERABLE, dtype: Type = None, device=None):
    """
        Create a 0-filled array given shape and the value to fill the array with.

        Args:
            shape (Union[list, tuple]): Shape of the array

            dtype (Type): Data type to cast to. Defaults to None, if
                left as such, dtype is inferred.

            device: Exist for compatibility with Array API, not implemented.

            Returns:
                Array: Array 0-filled with proper data type, with proper shape.
        """

    res = Array()
    n = 1
    for k in shape:
        n *= k
    res.dtype = dtype if dtype is not None else float
    res.values = [res.dtype(0)] * n
    res.device = device
    res.shape = copy(shape)
    res.ndim = len(shape)
    res.size = n
    return res

def zeros_like(x: Array, dtype: Type = None, device=None):
    """
        Create 0-filled array with the same shape as given array.

        Args:
            x (Array): Array object for the shape to be inferred
                from.

            dtype (Type): Data type for the array to have. Defaults to
                None. If left as such, returned array will have a dtype
                of base Python float.

            device: Exist for compatibility with Array API, not implemented.

        Returns:
            Array: 0-filled array with given shape and dtype attributes.
    """
    res = Array()
    res.dtype = dtype if dtype is not None else float
    res.values = [res.dtype(0)] * x.size
    res.device = device
    res.shape = copy(x.shape)
    res.ndim = x.ndim
    res.size = x.size
    return res
