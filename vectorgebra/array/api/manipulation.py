from ..ndarray import Array
from .. import ArgTypeError, DimensionError
from .utility import index_carry_, get_reversed_index, get_index_, axis_query_, axis_index_
from typing import Union, Tuple, List
from math import prod as __prod
from itertools import product as __product

def broadcast_arrays(*arrays) -> List[Tuple]:
    """
        Broadcasts the given arrays to a common shape, returning
        a list of broadcasted arrays. The common shape is determined
        by broadcasting the shapes of all given arrays.

        Args:
            *arrays: The arrays to be broadcasted. Must not be empty.

        Returns:
            A list of broadcasted arrays with a common shape.

        Notes:
            For some reason, Python Array API specifies that this function
            returns a "list" of arrays. For other functions in the manipulation
            module, they specify a "tuple". We follow the API specification no
            matter what. Thus we return a list here.
    """
    common_shape = None
    for arr in arrays:
        if common_shape is None:
            common_shape = arr.shape
        else:
            common_shape = Array.broadcast_shapes(common_shape, arr.shape)

    return [arr.broadcast_to(common_shape) for arr in arrays]


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
            _dtype = type(_dtype() + arr.dtype())  # We make Python to decide on the type lol

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

        if arr.ndim != _initial.ndim:
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
    N = 0  # size until axis, including the axis (append to axe)
    for arr in arrays:
        for i, val in enumerate(arr.values):
            _index = get_index_(arr.shape, i)
            get_reversed_index()
            #_index = index_carry_(_initial.shape, target_shape, _index)
            #res.values[_size_until_axe + i] = val

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

def flip(x: Array, axis: Union[int, Tuple[int], List[int]] = None) -> Array:
    """
        Flips the ordering of the values of the array along the specified axis or axes.
        If no axis is specified, the array is reversed completely.

        Args:
            x (Array): The array to be flipped.

            axis (int | tuple | list | None): The axis or axes to flip along.
                If None, the entire array is reversed. Defaults to None.

        Returns:
            The flipped Array object.
    """
    c_x = x.copy()

    if axis is None:
        c_x.values.reverse()

    elif isinstance(axis, int):
        shape = c_x.shape
        n = shape[axis]
        vals: list
        index_: int

        for shape_ in axis_query_(c_x.shape, axis):
            values = c_x[shape_].values  # This will have length n
            productable_shape_ = [[k] for k in shape_]
            productable_shape_[axis] = list(range(n))  # This will be a list of lists, each containing one element

            for i, tuple_index in enumerate(__product(*productable_shape_)):
                c_x.values[get_reversed_index(shape, tuple_index)] = values[n - i - 1]

    elif isinstance(axis, Union[list, tuple]):
        shape = c_x.shape
        for ax in axis:
            n = shape[ax]
            vals: list
            index_: int
            for shape_ in axis_query_(c_x.shape, ax):
                values = c_x[shape_].values
                productable_shape_ = [[k] for k in shape_]
                productable_shape_[ax] = list(range(n))  # This will be a list of lists, each containing one element

                for i, tuple_index in enumerate(__product(*productable_shape_)):
                    c_x.values[get_reversed_index(shape, tuple_index)] = values[n - i - 1]

    return c_x

def moveaxis(x: Array,
             source: Union[Tuple[int], List[int]],
             destination: Union[Tuple[int], List[int]]) -> Array:
    pass

def permute_dims(x: Array, axes: Union[Tuple[int], List[int]]) -> Array:

    """
        Permutes the dimensions of the given array according to the specified axes.

        Args:
            x (Array): The array to be permuted.

            axes (tuple | list): The new order of the axes. Must be a permutation
                of the original axes indices.

        Returns:
            The permuted Array object.

    """

    if len(axes) != x.ndim:
        raise DimensionError(0)

    res = x.copy()
    res.shape = tuple([x.shape[i] for i in axes])
    res.size = __prod(res.shape)
    res.ndim = x.ndim
    res.dtype = x.dtype
    res.device = x.device

    res.values = [x.values[get_reversed_index(x.shape, shape_)]
                  for shape_ in axis_query_(x.shape, axis=None, priority=axes)]

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

def squeeze(x: Array, axis: Union[int, Tuple[int], List[int]]) -> Array:
    c_x = x.copy()
    shape = list(x.shape)
    if isinstance(axis, int):
        if shape[axis] != 1:
            raise ValueError("Selected axis must be singleton.")
        shape.pop(axis)
        c_x.shape = tuple(shape)
    elif isinstance(axis, Union[list, tuple]):
        for ax in axis:
            if shape[ax] != 1:
                raise ValueError("Selected axis must be singleton.")
            shape.pop(ax)
        c_x.shape = tuple(shape)
    else:
        raise TypeError("Argument axis must be int, list or tuple.")

    return c_x

def stack():
    pass

def tile():
    pass

def unstack(x: Array, axis: int = 0) -> Tuple[Array]:
    """
        Splits the given array into multiple arrays along the
        given axis, returning a tuple of them.

        Args:
            x (Array): The array to be unstacked.

            axis (int): The axis to unstack along. Defaults to 0.

        Returns:
            A tuple of unstacked arrays.
    """
    return tuple(x[tuple(shape_)] for shape_ in axis_query_(x.shape, axis))

