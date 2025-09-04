from ..ndarray import Array
from typing import Tuple, Union
from itertools import product as __product

def all():
    pass

def any():
    pass

def index_(shape: tuple, axis: Union[int, Tuple[int]]):
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
                axes.

        Example:
            shape - (2, 3, 2)
            axis - 1
            outputs in order
                - (0, :, 0)
                - (0, :, 1)
                - (1, :, 0)
                - (1, :, 1)
    """
    if isinstance(axis, int):
        reduced_shape = [range(n) if i != axis else [slice(0, n, 1)] for i, n in enumerate(shape)]
        for shape_ in __product(*reduced_shape):
            yield shape_
    elif isinstance(axis, tuple):
        reduced_shape = [range(n) if i not in axis else [slice(0, n, 1)] for i, n in enumerate(shape)]
        for shape_ in __product(*reduced_shape):
            yield shape_


