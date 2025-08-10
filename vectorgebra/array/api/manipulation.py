from ..ndarray import Array
from .. import ArgTypeError, DimensionError
from .utility import get_reversed_index, get_index_, axis_query_, shift_array_
from typing import Union, Tuple, List
from builtins import sum as __builtinSum
from math import prod as __prod
from itertools import product as __product, chain as __chain, repeat as __repeat

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
    res.device = _initial.device  # Be careful here in the future

    _dtype = _initial.dtype

    for arr in arrays[1:]:

        if arr.ndim != _initial.ndim:
            raise DimensionError(0)

        for i in range(arr.ndim):

            if i != axis and arr.shape[i] != _initial.shape[i]:
                raise DimensionError(0)

        _dtype = type(_dtype() + arr.dtype())  # Promote types
        target_shape[axis] += arr.shape[axis]  # Add the size of the axis

    res.size = __prod(target_shape)
    res.values = [None] * res.size
    res.dtype = _dtype
    tuple_index: tuple
    N = 0  # size until axis, including the axis (append to axe)
    for arr in arrays:
        for i, val in enumerate(arr.values):
            tuple_index = list(get_index_(arr.shape, i))
            tuple_index[axis] += N
            res.values[get_reversed_index(target_shape, tuple_index)] = val
        N += arr.shape[axis]

    res.shape = tuple(target_shape)
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

    to_insert = list(x.shape)
    to_insert.insert(axis, 1)
    shape = tuple(to_insert)  # Might raise index error here
    c_x = x.copy()
    c_x.shape = shape
    return c_x

def unsqueeze(x: Array, axis: int = 0) -> Array:
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
    to_insert = list(x.shape)
    to_insert.insert(axis, 1)  # Insert a 1 at the given axis
    shape = tuple(to_insert)  # Might raise index error here
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
             source: Union[int, Tuple[int], List[int]],
             destination: Union[int, Tuple[int], List[int]]) -> Array:

    """
        Moves the specified axes of the array to new positions.
        If source is an int, it moves that axis to the destination index.
        If source is a list or tuple, it moves all specified axes to the corresponding
        destination indices.

        This function simply calculates a permutation map based on given source/destination,
        and then calls permute_dims with that permutation.

        Args:
            x (Array): The array to be manipulated.

            source (int | list | tuple): The axis or axes to move.
                If an int, moves that axis to the destination index.
                If a list or tuple, moves all specified axes to the corresponding
                destination indices.

            destination (int | list | tuple): The new position(s) for the specified axes.
                Must match the length of source if source is a list or tuple.

        Returns:
            The Array object with the specified axes moved.

        Raises:
            ArgTypeError: If source is an int and destination is not an int,
                or if source is a list/tuple and destination is not a list/tuple.
            DimensionError: If the destination does not match the length of source
                when source is a list/tuple.
    """

    if isinstance(source, int):
        if not isinstance(destination, int):
            raise ArgTypeError("If source is an int, destination must also be an int.")
        if source < 0:
            source += x.ndim
        if destination < 0:
            destination += x.ndim
        permutation = list(range(x.ndim))
        permutation.insert(destination, permutation.pop(source))  # Move the axis
        return permute_dims(x, permutation)
    elif isinstance(source, Union[list, tuple]):
        if len(source) != len(set(source)):
            raise DimensionError(0)

        source = [s if s >= 0 else s + x.ndim for i, s in enumerate(source)]
        destination = [d if d >= 0 else d + x.ndim for i, d in enumerate(destination)]
        final_permutation = [None] * x.ndim

        for i, s in enumerate(source):
            final_permutation[destination[i]] = s

        not_source = [i for i in range(x.ndim) if i not in source]

        for i in range(x.ndim):
            if final_permutation[i] is None:
                final_permutation[i] = not_source.pop(0)

        return permute_dims(x, final_permutation)


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

def repeat(x: Array, repeats: Union[int, Array], axis: int = None) -> Array:

    """
        Repeats the values of the array along the specified axis or axes.
        If no axis is specified, the array is flattened and repeated.

        Args:
            x (Array): The array to be repeated.

            repeats (int | Array): The number of times to repeat the values.
                If an int, repeats all values that many times.
                If an Array, repeats each value according to the corresponding
                value in the Array.

            axis (int | None): The axis to repeat along. If None, flattens
                the array and repeats all values. Defaults to None.

        Returns:
            The repeated Array object.

        Raises:
            ArgTypeError: If repeats is not an int or an Array, or axis is not an int or None.
            DimensionError: If the size of the repeats Array does not match
                the size of the axis being repeated.
    """

    res = Array()
    res.ndim = x.ndim
    res.dtype = x.dtype
    res.device = x.device

    if axis is None:
        if isinstance(repeats, int):
            res.size = x.size * repeats
            res.shape = (res.size,)
            res.values = list(__chain.from_iterable(__repeat(v, repeats) for v in x.values))
        elif isinstance(repeats, Array):
            if repeats.size == 1:
                repeats = repeats.values[0]
                res.size = x.size * repeats
                res.shape = (res.size,)
                res.values = list(__chain.from_iterable(__repeat(v, repeats) for v in x.values))
            else:
                res.size = x.size * repeats.sum().values[0]
                res.values = list(__chain.from_iterable(__repeat(v, r) for v, r in zip(x.values, repeats.values)))
                repeats.reshape(x.shape)
                res.shape = (res.size,)
        else:
            raise ArgTypeError("'repeats' must be either int or Array")
    elif isinstance(axis, int):
        if isinstance(repeats, int):
            res.shape = tuple([x.shape[i] if i != axis else x.shape[i] * repeats for i in range(x.ndim)])
            res.size = __prod(res.shape)
            res.values = [None] * res.size
            temp_shape: list
            x_val: x.dtype

            for i, shape_ in enumerate(axis_query_(x.shape)):
                temp_shape = shape_.copy()
                temp_shape[axis] = temp_shape[axis] * repeats
                x_val = x.values[get_reversed_index(x.shape, shape_)]
                for j in range(repeats):
                    res.values[get_reversed_index(res.shape, temp_shape)] = x_val
                    temp_shape[axis] += 1

        elif isinstance(repeats, Array):
            if repeats.size == 1:
                repeats = repeats.values[0]
                res.shape = tuple([x.shape[i] if i != axis else x.shape[i] * repeats for i in range(x.ndim)])
                res.size = __prod(res.shape)
                res.values = [None] * res.size
                temp_shape: list
                x_val: x.dtype

                for i, shape_ in enumerate(axis_query_(x.shape)):
                    temp_shape = shape_.copy()
                    temp_shape[axis] = temp_shape[axis] * repeats
                    x_val = x.values[get_reversed_index(x.shape, shape_)]
                    for j in range(repeats):
                        res.values[get_reversed_index(res.shape, temp_shape)] = x_val
                        temp_shape[axis] += 1
            else:
                if repeats.shape[0] != x.shape[axis]:
                    raise DimensionError(0)
                res.shape = tuple([x.shape[i] if i != axis else repeats.sum().values[0] for i in range(x.ndim)])
                res.size = __prod(res.shape)
                res.values = [None] * res.size
                temp_shape: list
                offset: int
                repeat_values = repeats.values

                for i, shape_ in enumerate(axis_query_(x.shape)):
                    temp_shape = shape_.copy()
                    temp_shape[axis] = __builtinSum(repeat_values[:shape_[axis]])
                    x_val = x.values[get_reversed_index(x.shape, shape_)]
                    for j in range(repeat_values[shape_[axis]]):
                        res.values[get_reversed_index(res.shape, temp_shape)] = x_val
                        temp_shape[axis] += 1
        else:
            raise ArgTypeError("'repeats' must be either int or Array")
    else:
        raise ArgTypeError("Axis must be int or None.")

    return res

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

def roll(x: Array,
         shift: Union[int, Tuple[int], List[int]],
         axis: Union[int, Tuple[int]] = None) -> Array:
    """
        Rolls the values of the array along the specified axis or axes.
        If no axis is specified, the array is rolled completely.

        Args:
            x (Array): The array to be rolled.

            shift (int | tuple | list): The number of positions to roll the values.
                If a single integer is given, rolls along the specified axis.
                If a tuple or list is given, rolls along multiple axes.

            axis (int | tuple | list | None): The axis or axes to roll along.
                If None, the entire array is rolled. Defaults to None.

        Returns:
            The rolled Array object.

        Raises:
            ArgTypeError: If shift is a list or tuple and axis is not also a list or tuple.
            DimensionError: If the length of shift does not match the number of axes specified.
    """

    c_x = x.copy()

    if axis is None:
        axis = list(range(c_x.ndim))

    if isinstance(shift, int):

        if isinstance(axis, int):
            shape = c_x.shape
            n = shape[axis]
            vals: list
            index_: int

            for shape_ in axis_query_(c_x.shape, axis):
                values = shift_array_(c_x[shape_].values, shift)  # This will have length n
                productable_shape_ = [[k] for k in shape_]
                productable_shape_[axis] = list(range(n))  # This will be a list of lists, each containing one element

                for i, tuple_index in enumerate(__product(*productable_shape_)):
                    c_x.values[get_reversed_index(shape, tuple_index)] = values[i]
        elif isinstance(axis, Union[list, tuple]):
            shape = c_x.shape
            for ax in axis:
                n = shape[ax]
                vals: list
                index_: int
                for shape_ in axis_query_(c_x.shape, ax):
                    values = shift_array_(c_x[shape_].values, shift)
                    productable_shape_ = [[k] for k in shape_]
                    productable_shape_[ax] = list(range(n))  # This will be a list of lists, each containing one element

                    for i, tuple_index in enumerate(__product(*productable_shape_)):
                        c_x.values[get_reversed_index(shape, tuple_index)] = values[i]

    elif isinstance(shift, Union[list, tuple]):
        if isinstance(axis, int):
            raise ArgTypeError("If shift is a list or tuple, axis must also be a list or tuple.")
        if len(shift) != len(axis):  # Then axis must also be a list or tuple
            raise DimensionError(0)

        shape = c_x.shape
        for i, ax in enumerate(axis):
            n = shape[ax]
            vals: list
            index_: int
            for shape_ in axis_query_(c_x.shape, ax):
                values = shift_array_(c_x[shape_].values, shift[i])
                productable_shape_ = [[k] for k in shape_]
                productable_shape_[ax] = list(range(n))  # This will be a list of lists, each containing one element

                for j, tuple_index in enumerate(__product(*productable_shape_)):
                    c_x.values[get_reversed_index(shape, tuple_index)] = values[j]

    return c_x


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

def stack(arrays: Union[Tuple[Array], List[Array]], axis: int = 0) -> Array:

    """
        Stacks the values within the given list/tuple of arrays, along the
        specified axis. The shape of all given arrays must match. This
        operation simply inserts an axis of length "1" at the position
        defined by "axis", and then concatenates the arrays.

        Args:
            arrays: The list/tuple of Arrays, cannot be empty

            axis (int): The axis to stack along. Defaults to 0.

        Returns:
            The stacked Array object.

        Raises:
            ArgTypeError: If the given array list is empty

            DimensionError: If the given arrays have non-matching shapes.
    """

    try:
        _initial = arrays[0]
    except IndexError:
        raise ArgTypeError("Array list/tuple cannot be empty.")

    for arr in arrays[1:]:
        if arr.shape != _initial.shape:
            raise DimensionError(0)

    arrays = [expand_dims(arr, axis=axis) for arr in arrays]

    target_shape = list(arrays[0].shape)
    res = Array()
    res.ndim = _initial.ndim
    res.device = _initial.device  # Be careful here in the future

    _dtype = _initial.dtype

    for arr in arrays[1:]:
        _dtype = type(_dtype() + arr.dtype())  # Promote types

    target_shape[axis] = len(arrays)

    res.size = __prod(target_shape)
    res.values = [None] * res.size
    res.dtype = _dtype
    tuple_index: tuple
    N = 0  # size until axis, including the axis (append to axe)
    for arr in arrays:
        for i, val in enumerate(arr.values):
            tuple_index = list(get_index_(arr.shape, i))
            tuple_index[axis] += N
            res.values[get_reversed_index(target_shape, tuple_index)] = val
        N += arr.shape[axis]

    res.shape = tuple(target_shape)
    return res

def tile(x: Array, repetitions: Tuple[int]) -> Array:

    N = x.ndim
    M = len(repetitions)
    c_x = x.copy()  # This function therefore utilizes more RAM

    if N > M:
        diff = N - M
        repetitions = tuple(*[1 for k in range(diff)], *repetitions)
    elif N < M:
        diff = M - N
        c_x.shape = tuple(*[1 for k in range(diff)], *x.shape)

    for i in range(len(repetitions)):
        if repetitions[i] == 1:
            continue

        c_x = concat([c_x] * repetitions[i], axis=i)

    return c_x

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
    return tuple(x[shape_] for shape_ in axis_query_(x.shape, axis))

