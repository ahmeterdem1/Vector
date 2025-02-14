from ..ndarray import Array
from typing import Union

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

def concat():
    pass

def expand_dims(x: Array, axis: int = 0) -> Array:
    shape = tuple(list(x.shape).insert(axis, 1))  # Might raise index error here
    c_x = x.copy()
    c_x.shape = shape
    return c_x

def flip():
    pass

def moveaxis():
    pass

def permute_dims():
    pass

def repeat():
    pass

def reshape():
    pass

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
