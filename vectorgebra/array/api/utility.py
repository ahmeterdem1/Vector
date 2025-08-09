from ..ndarray import Array, ArgTypeError
from typing import Tuple, Union, List
from itertools import product as __product
from math import prod as __prod

def all(x: Array, axis: Union[int, Tuple[int]] = None, keepdims: bool = False):
    """
        Tests whether all input array elements evaluate to True along a specified axis.

        Args:
            x (Array): The array to perform "all" operation on.

            axis: The axis to calculate the truth value along. There may be multiple axes
                as a tuple of integers. If left as None, truth value is calculated over
                all of the array. The default is None.

            keepdims (bool): If True, reduced axes are included in the shape of the
                returned array. The default is False.

        Returns:
            The array containing the truth values from the "all" operation.

        Raises:
            ArgTypeError: If given axis does not conform to allowed format.
    """

    res = Array()
    values = [x[shape_].all() for shape_ in axis_query_(x.shape, axis)] if axis is not None else [all(x.values)]
    res.values = values
    res.size = len(values)
    res.dtype = bool

    if axis is None:
        res.shape = (1,) * x.ndim if keepdims else (0,)
    elif isinstance(axis, int):
        res.shape = tuple([x.shape[i] for i in range(x.ndim) if i != axis]) if not keepdims else tuple(
            [x.shape[i] if i != axis else 1 for i in range(x.ndim)])
    elif isinstance(axis, tuple):
        res.shape = tuple([x.shape[i] for i in range(x.ndim) if i not in axis]) if not keepdims else tuple(
            [x.shape[i] if i not in axis else 1 for i in range(x.ndim)])
    else:
        raise ArgTypeError("Argument 'axis' must be one of [None, int, tuple[int]]")

    res.ndim = len(res.shape)
    return res

def any(x: Array, axis: Union[int, Tuple[int]] = None, keepdims: bool = False):
    """
        Tests whether "any" input array elements evaluate to True along a specified axis.

        Args:
            x (Array): The array to perform "any" operation on.

            axis: The axis to calculate the truth value along. There may be multiple axes
                as a tuple of integers. If left as None, truth value is calculated over
                all of the array. The default is None.

            keepdims (bool): If True, reduced axes are included in the shape of the
                returned array. The default is False.

        Returns:
            The array containing the truth values from the "any" operation.

        Raises:
            ArgTypeError: If given axis does not conform to allowed format.
    """

    res = Array()
    values = [x[shape_].any() for shape_ in axis_query_(x.shape, axis)] if axis is not None else [any(x.values)]
    res.values = values
    res.size = len(values)
    res.dtype = bool

    if axis is None:
        res.shape = (1,) * x.ndim if keepdims else (0,)
    elif isinstance(axis, int):
        res.shape = tuple([x.shape[i] for i in range(x.ndim) if i != axis]) if not keepdims else tuple(
            [x.shape[i] if i != axis else 1 for i in range(x.ndim)])
    elif isinstance(axis, tuple):
        res.shape = tuple([x.shape[i] for i in range(x.ndim) if i not in axis]) if not keepdims else tuple(
            [x.shape[i] if i not in axis else 1 for i in range(x.ndim)])
    else:
        raise ArgTypeError("Argument 'axis' must be one of [None, int, tuple[int]]")

    res.ndim = len(res.shape)
    return res

def permute_array_(array: Union[list, tuple], permutation: Union[list, tuple]) -> list:
    """
        Permutes the given array according to the given permutation.

        Args:
            array (list or tuple): The array to permute.

            permutation (list or tuple): The permutation to apply to the array.
                It should contain indices that correspond to the new order of
                elements in the array.

    """
    return [array[idx] for idx in permutation]

def inverse_permutation_(perm: Union[list, tuple]) -> list:
    inv = [0] * len(perm)
    for i, p in enumerate(perm):
        inv[p] = i
    return inv

def axis_query_(shape: tuple,
                axis: Union[int, Tuple[int], None] = None,
                priority: Union[List[int], Tuple[int]] = None):
    """
        A generator, that indexes the given shape in lexical order,
        while indexing through all of the given axes.

        The algorithm of this function is used in any other function
        that has an "axis" argument. This function is not called from
        them to prevent stack overhead. Rather, it is implemented in
        place.

        Args:
            shape (tuple): The shape to iterate through.

            axis: Axis/axes to index through whilst iterating over the other
                axes. Defaults to None, which means that all axes are indexed.

            priority (list[int]): Optional ordering of axes for iteration priority.
                              Lower indices in this list vary fastest.

        Example:
            shape - (2, 3, 2)
            priority - (0, 1, 2)
            axis - 1

            outputs in order
                - (0, :, 0)
                - (0, :, 1)
                - (1, :, 0)
                - (1, :, 1)
    """

    if priority is None:
        priority = list(range(len(shape)))

    inverse_priority = inverse_permutation_(priority)
    shape = permute_array_(shape, priority)

    if axis is None:
        for shape_ in __product(*[range(n) for n in shape]):
            yield permute_array_(shape_, inverse_priority)
    elif isinstance(axis, int):
        permuted_axis = inverse_priority[axis]
        reduced_shape = [range(n) if i != permuted_axis else [slice(0, n, 1)] for i, n in enumerate(shape)]
        for shape_ in __product(*reduced_shape):
            yield permute_array_(shape_, inverse_priority)
    elif isinstance(axis, tuple):
        permuted_axis = [inverse_priority[ax] for ax in axis]
        reduced_shape = [range(n) if i not in permuted_axis else [slice(0, n, 1)] for i, n in enumerate(shape)]
        for shape_ in __product(*reduced_shape):
            yield permute_array_(shape_, inverse_priority)

def axis_index_(query: Union[list, tuple], axis: int, n: int) -> tuple:
    slice_axe = query[axis]
    start, stop, step = slice_axe.indices(n)

    for i in range(start, stop, step):
        yield tuple(query[j] if j != axis else i for j in range(len(query)))

def index_query_(shape: tuple, key) -> list:
    """
        This function, returns the corresponding indexes to
        a given getitem query. The internal algorithm of this
        function is used by __setitem__ method of Array class.
        But it does not make a call to this function, rather
        implements it separately.

        Args:
            shape (tuple): Shape to retrieve indexes according to.

            key: An index, a tuple of indexes/slices etc. to retrieve
                element indexes from the shape.

        Returns:
            list: The list of corresponding indexes.

        Example:
            array = Array(...) # shape = (2, 3, 2)
            indexes = api.index_query_((2, 3, 2), [slice(2), 2, slice(2)])  # usage
            array[indexes] = [values to assign]  # further usage of the result from this function
    """
    size = __prod(shape)
    Ns = [size // shape[0]]
    for k in shape[1:]:
        Ns.append(Ns[-1] // k)

    indices = []

    start: int
    stop: int
    step: int

    sliced_previously = False

    for i, it in enumerate(key):
        if isinstance(it, int):
            if sliced_previously:  # indices always exist
                temp_indices = []
                for k in range(len(indices)):
                    temp_indices.extend(indices[it * Ns[i] + k * Ns[i - 1]:it * Ns[i] + k * Ns[i - 1] + Ns[i]])
                indices = temp_indices
            elif indices:
                indices = indices[it * Ns[i]:(it + 1) * Ns[i]]
            else:
                indices = list(range(it * Ns[i], (it + 1) * Ns[i]))
        elif isinstance(it, slice):
            start, stop, step = it.indices(shape[i])

            if sliced_previously:
                dN = Ns[i] * (stop - start) // step

                temp_indices = []
                for k in range(shape[i - 1]):
                    temp_indices.extend(indices[start * Ns[i] + k * dN:stop * Ns[i] + k * dN:step])
                indices = temp_indices
            elif indices:
                indices = indices[start * Ns[i]:stop * Ns[i]:step]
            else:
                indices = list(range(start * Ns[i], stop * Ns[i], step))

            sliced_previously = True

    return indices

def get_index_(shape: tuple, index: int) -> tuple:
    indices = []

    for dim in reversed(shape):
        indices.append(index % dim)
        index //= dim

    return tuple(reversed(indices))

def get_reversed_index(shape: tuple, indices: tuple) -> int:
    """
        Returns the integer inner index of a given tuple index
        within the given shape domain.
    """
    size = __prod(shape)

    res: int = 0
    for i, val in enumerate(indices):
        size //= shape[i]  # will be 1 at the last iteration
        res += size * val
    return res

def index_carry_(shape1: tuple, shape2: tuple, index: int) -> int:
    indices = []

    for dim in reversed(shape1):
        indices.append(index % dim)
        index //= dim

    indices.reverse()
    size = __prod(indices)

    res: int = 0
    for i, val in enumerate(indices):
        size //= shape2[i]  # will be 1 at the last iteration
        res += size * val
    return res

def index_map_(shape: tuple, perm: tuple, index: tuple) -> tuple:
    """
        Calculates the integer index of a given tuple index,
        within the new shape that is defined by the given
        permutation of the given shape.

        Args:
            shape (tuple): Shape of the source array.

            perm (tuple): Permutation tuple, containing the
                indexes of each corresponding element of
                the given shape, in the target shape. E.g.
                (0, 2, 1).

            index (tuple): Tuple-index of an element within
                the given shape.

    """
    target_shape = [shape[i] for i in perm]
    target_index = [index[i] for i in perm]
    size = __prod(target_shape)

    res: int = 0
    for i, val in enumerate(target_index):
        size //= target_shape[i]  # will be 1 at the last iteration
        res += size * target_index[val]
    return res


