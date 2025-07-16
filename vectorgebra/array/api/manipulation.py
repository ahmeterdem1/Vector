from ..ndarray import Array
from .. import ArgTypeError, DimensionError
from .utility import index_carry_, get_reversed_index, get_index_
from typing import Union, Tuple, List
from math import prod as __prod

def broadcast_arrays():
    pass

def broadcast_to(x: Array, shape: Union[list, tuple]) -> Array:
    """
        Broadcast the array to given shape. Does not
        modify the given array, broadcasts and returns a copy
        of it.

        Args:
             x (Array): The array to be broadcasted.

             shape (list | tuple): The shape to broadcast the
                array to.

        Returns:
            The array that is broadcasted to given shape.
    """
    c_x = x.copy()
    c_x.broadcast_to(shape)
    return c_x

def concat(arrays: Union[Tuple[Array], List[Array]], axis: int = 0) -> Array:
    """
        Concatenates the values within the given list/tuple of arrays,
        along an already existing axis. ndim of all given arrays must
        be equal, arrays are not broadcasted.

        Args:
            arrays: The list/tuple of Arrays, cannot be empty

            axis (int | None): The axis to conatenate along.
                Defaults to 0. If None, concatenates the
                flattened arrays.

        Returns:
            The concatenated Array object.

        Raises:
            ArgTypeError: If the given array list is empty

            DimensionError: If the given arrays have non-matching
                shapes, except for the selected axis.
    """
    if axis is None:
        res = Array()
        res.size = sum([arr.size for arr in arrays])
        res.ndim = 1
        res.values = []
        _dtype = arrays[0].dtype

        for arr in arrays:
            res.values.extend(arr.values.copy())
            _dtype = type(_dtype() + arr.dtype())

        res.shape = (res.size,)
        res.device = arrays[0].device
        res.dtype = _dtype
        return res

    try:
        _initial = arrays[0]
    except IndexError:
        raise ArgTypeError("Array list/tuple cannot be empty.")

    target_shape = list(_initial.shape)
    res = Array()
    res.ndim = _initial.ndim
    res.size = sum([arr.size for arr in arrays])
    res.values = [None] * res.size
    res.device = _initial.device  # Be careful here in the future

    _dtype = _initial.dtype

    for arr in arrays:

        if arr.ndim > _initial.ndim:
            raise DimensionError(0)

        for i in range(arr.ndim):

            if i != axis and arr.shape[i] != _initial.shape[i]:
                raise DimensionError(0)

            target_shape[i] += arr.shape[i]
        _dtype = type(_dtype() + arr.dtype())  # Promote types

    res.dtype = _dtype
    # TODO: Fix this
    _index: tuple
    _size_until_axe = 0
    for arr in arrays:
        for i, val in arr.values:
            res.values[i + _size_until_axe] = val

        _size_until_axe += __prod(arr.shape[axis:])

    return res

def expand_dims(x: Array, axis: int = 0) -> Array:
    """
        Increases the ndim of given array by 1, with inserting
        a 1 dimensional axis into the given index as *axis*.
        Does not modify the given array.

        Args:
             x (Array): The array to be expanded.

             axis (int): The index to insert an axis into.
                Defaults to 0.

        Returns:
            The expanded Array object.
    """
    shape = tuple(list(x.shape).insert(axis, 1))  # Might raise index error here
    c_x = x.copy()
    c_x.shape = shape
    return c_x

def flip():
    pass

def moveaxis(x: Array,
             source: Union[Tuple[int], List[int]],
             destination: Union[Tuple[int], List[int]]) -> Array:
    pass

def permute_dims(x: Array, axes: Union[Tuple[int], List[int]]) -> Array:
    res = x.copy()
    res.shape = [x.shape[i] for i in axes]
    res.size = __prod(res.shape)
    res.ndim = x.ndim
    res.dtype = x.dtype
    res.device = x.device

    for i, val in x.values:
        res.values[index_carry_(x.shape, res.shape, i)] = val

    return res


def repeat():
    pass

def reshape(x: Array,
            shape: Union[Tuple[int], List[int]],
            copy: bool = None) -> Array:
    c_x = x
    if copy:
        c_x = x.copy()

    if __prod(shape) != c_x.size:
        raise DimensionError(0)

    c_x.shape = tuple(shape)
    return c_x

def roll():
    pass

def squeeze():
    pass

def stack():
    pass

def tile():
    pass

def unstack():
    pass
