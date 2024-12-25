"""
    Functions referenced by: https://data-apis.org/array-api/latest/API_specification/searching_functions.html
"""

from ..ndarray import Array
from .utility import *
from builtins import max as __builtinMax, min as __builtinMin

def argmax(x: Array, axis: Union[int, Tuple[int]] = None, keepdims: bool = False) -> Array:
    """
        Get the index of the maximal element(s), along the given axis.

        Args:
            x (Array): The array to find maximal element indexes

            axis: The axis/axes to find maximal indexes along. If None, maximal
                index among all elements in the array is returned as a singular
                element array. The default is None.

            keepdims (bool): If True, reduced axes are included in the shape of the
                returned array. The default is False.

        Returns:
            Array: The array consisting of maximal indexes along axes.
    """
    res = Array()
    res.dtype = None
    vals = []
    temp: list

    if axis is None:
        vals.append(x.values.index(__builtinMax(x.values)))
        res.size = 1
        res.ndim = 0
        res.shape = (0,)
    elif isinstance(axis, int):
        for shape in axis_query_(x.shape, axis):
            temp = x[*shape].values
            vals.append(temp.index(__builtinMax(temp)))
        res.shape = tuple([x.shape[i] for i in range(x.ndim) if i != axis]) if not keepdims else tuple(
            [x.shape[i] if i != axis else 1 for i in range(x.ndim)])
        res.size = x.size // x.shape[axis]
        res.ndim = len(res.shape)
        res.values = vals
    elif isinstance(axis, tuple):
        for shape in axis_query_(x.shape, axis):
            temp = x[*shape].values
            vals.append(temp.index(__builtinMax(temp)))
        res.shape = tuple([x.shape[i] for i in range(x.ndim) if i not in axis]) if not keepdims else tuple(
            [x.shape[i] if i not in axis else 1 for i in range(x.ndim)])
        res.size = x.size // x.shape[axis]
        res.ndim = len(res.shape)
        res.values = vals

    return res

def argmin(x: Array, axis: Union[int, Tuple[int]] = None, keepdims: bool = False) -> Array:
    """
        Get the index of the minimal element(s), along the given axis.

        Args:
            x (Array): The array to find minimal element indexes

            axis: The axis/axes to find minimal indexes along. If None, minimal
                index among all elements in the array is returned as a singular
                element array. The default is None.

            keepdims (bool): If True, reduced axes are included in the shape of the
                returned array. The default is False.

        Returns:
            Array: The array consisting of minimal indexes along axes.
    """
    res = Array()
    res.dtype = None
    vals = []
    temp: list

    if axis is None:
        vals.append(x.values.index(__builtinMin(x.values)))
        res.size = 1
        res.ndim = 0
        res.shape = (0,)
    elif isinstance(axis, int):
        for shape in axis_query_(x.shape, axis):
            temp = x[*shape].values
            vals.append(temp.index(__builtinMin(temp)))
        res.shape = tuple([x.shape[i] for i in range(x.ndim) if i != axis]) if not keepdims else tuple(
            [x.shape[i] if i != axis else 1 for i in range(x.ndim)])
        res.size = x.size // x.shape[axis]
        res.ndim = len(res.shape)
        res.values = vals
    elif isinstance(axis, tuple):
        for shape in axis_query_(x.shape, axis):
            temp = x[*shape].values
            vals.append(temp.index(__builtinMin(temp)))
        res.shape = tuple([x.shape[i] for i in range(x.ndim) if i not in axis]) if not keepdims else tuple(
            [x.shape[i] if i not in axis else 1 for i in range(x.ndim)])
        res.size = x.size // x.shape[axis]
        res.ndim = len(res.shape)
        res.values = vals

    return res

def nonzero(x: Array) -> Array:
    """
        Returns the indices of all nonzero values in the given array,
        as an Array.
    """
    return Array([get_index_(x.shape, i) for i, val in enumerate(x.values) if val])

def nonzero_(x: Array) -> Array:
    """
        Returns the internal indices of all nonzero values in the given array,
        as an Array.
    """
    return Array([i for i, val in enumerate(x.values) if val])

def searchsorted(x1: Array, x2: Array, side: str, sorter: Array = None):
    raise NotImplementedError()

def where(condition: Array, x1: Array, x2: Array) -> Array:
    """
        Returns an array of elements from *x1* and *x2*, chosen based on the
        corresponding condition from *condition*. All 3 arrays must be broadcastable.

        Args:
            condition (Array): Boolean array to be the choice basis between x1 and x2.

            x1 (Array): Array of elements to be picked if the accordingly indexed condition
                is True.

            x2 (Array): Array of elements to be picked if the accordingly indexed condition
                is False.

        Returns:
            Array: Combination of x1 and x2 based on condition, broadcasted to common shape.

        Raises:
            DimensionError: If any of the arrays are not broadcastable.

    """
    res = Array()
    res.dtype = None

    if condition.shape == x1.shape == x2.shape:
        res.shape = condition.shape
        res.ndim = condition.ndim
        res.size = condition.size
        res.values = [x1.values[i] if condition.values[i] else x2.values[i] for i in range(res.size)]
        return res
    if condition.shape == x1.shape:
        common_shape = Array.broadcast(condition, x2)
        c_condition = condition
        c_x1 = x1
        c_x2 = x2

        if common_shape != condition.shape:
            c_condition = condition.copy().broadcast_to(common_shape)
        if common_shape != x1.shape:
            c_x1 = x1.copy().broadcast_to(common_shape)
        if common_shape != x2.shape:
            c_x2 = x2.copy().broadcast_to(common_shape)

        # It is now assured that each 3 of arrays are in the common shape
        res.shape = c_condition.shape
        res.ndim = c_condition.ndim
        res.size = c_condition.size
        res.values = [c_x1.values[i] if c_condition.values[i] else c_x2.values[i] for i in range(res.size)]
        return res

    if condition.shape == x2.shape:
        common_shape = Array.broadcast(condition, x1)
        c_condition = condition
        c_x1 = x1
        c_x2 = x2

        if common_shape != condition.shape:
            c_condition = condition.copy().broadcast_to(common_shape)
        if common_shape != x1.shape:
            c_x1 = x1.copy().broadcast_to(common_shape)
        if common_shape != x2.shape:
            c_x2 = x2.copy().broadcast_to(common_shape)

        # It is now assured that each 3 of arrays are in the common shape
        res.shape = c_condition.shape
        res.ndim = c_condition.ndim
        res.size = c_condition.size
        res.values = [c_x1.values[i] if c_condition.values[i] else c_x2.values[i] for i in range(res.size)]
        return res
    if x1.shape == x2.shape:
        common_shape = Array.broadcast(condition, x2)
        c_condition = condition
        c_x1 = x1
        c_x2 = x2

        if common_shape != condition.shape:
            c_condition = condition.copy().broadcast_to(common_shape)
        if common_shape != x1.shape:
            c_x1 = x1.copy().broadcast_to(common_shape)
        if common_shape != x2.shape:
            c_x2 = x2.copy().broadcast_to(common_shape)

        # It is now assured that each 3 of arrays are in the common shape
        res.shape = c_condition.shape
        res.ndim = c_condition.ndim
        res.size = c_condition.size
        res.values = [c_x1.values[i] if c_condition.values[i] else c_x2.values[i] for i in range(res.size)]
        return res
    else:
        # The optimal broadcasting order here is probably an NP hard problem.
        # We assume, "condition" is likely to be in the broadcasted shapes of x1 and x2
        common_shape = Array.broadcast(x1, x2)
        c_condition = condition
        c_x1 = x1
        c_x2 = x2

        if common_shape != x1.shape:
            c_x1 = x1.copy().broadcast_to(common_shape)
        if common_shape != x2.shape:
            c_x2 = x2.copy().broadcast_to(common_shape)

        # It is now assured that x1 and x2 are in the same shape

        common_shape = Array.broadcast(condition, c_x1)

        if common_shape != condition.shape:
            c_condition = condition.copy().broadcast_to(common_shape)
        if common_shape != x1.shape:
            c_x1 = x1.copy().broadcast_to(common_shape)
        if common_shape != x2.shape:
            c_x2 = x2.copy().broadcast_to(common_shape)

        # It is now assured that each 3 of arrays are in the common shape
        res.shape = c_condition.shape
        res.ndim = c_condition.ndim
        res.size = c_condition.size
        res.values = [c_x1.values[i] if c_condition.values[i] else c_x2.values[i] for i in range(res.size)]
        return res

