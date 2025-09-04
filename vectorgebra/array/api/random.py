from ..ndarray import Array
import random as __random
from typing import Union

def seed(a):
    """
        Set a seed for random generations.
        Explicitly calls the .seed() function from
        builtin "random" module.
    """
    __random.seed(a)

def randn(shape: Union[list, tuple], device=None) -> Array:
    """
        Generate an Array of random numbers, pulled from normal
        Gaussian distribution.

        This function might be slower than others because
        instead of making a call to random.choices, it uses
        list comprehensions.

        Args:
            shape (list | tuple): Shape of the Array to be generated.

            device: Device of the Array to be generated. (Not Implemented).

        Returns:
            The generated random normal valued Array.
    """
    res = Array()
    res.device = device
    res.dtype = float
    res.shape = tuple(shape)
    res.ndim = len(shape)
    res.size = 1
    for k in shape:
        res.size *= k

    res.values = [__random.gauss(0, 1) for k in range(res.size)]
    return res

def rand(shape: Union[list, tuple], device=None) -> Array:
    """
        Generate an Array of random numbers, within [0, 1) range.

        Args:
            shape (list | tuple): Shape of the Array to be generated.

            device: Device of the Array to be generated. (Not Implemented).

        Returns:
            The generated random valued Array.

    """
    res = Array()
    res.device = device
    res.dtype = float
    res.shape = tuple(shape)
    res.ndim = len(shape)
    res.size = 1
    for k in shape:
        res.size *= k

    res.values = [__random.random() for k in range(res.size)]
    return res

def randint(shape: Union[list, tuple], a: int, b: int,
            device=None) -> Array:
    """
        Generate an Array of random integers, from an interval.

        Args:
             shape (list | tuple): Shape of the Array to be generated.

             a (int): Low end of the interval, included.

             b (int): High end of the interval, excluded.

            device: Device of the Array to be generated. (Not Implemented).

        Returns:
            The generated random valued Array.
    """
    res = Array()
    res.device = device
    res.dtype = float
    res.shape = tuple(shape)
    res.ndim = len(shape)
    res.size = 1
    for k in shape:
        res.size *= k

    res.values = __random.choices(range(a, b), k=res.size)
    return res

def randrange(shape: Union[list, tuple], start: int, stop: int, step: int = 1,
              device=None) -> Array:
    """
        Generate an Array of random integers, from within a range, with given
        stepsize.

        Args:
             shape (list | tuple): Shape of the Array to be generated.

             start (int): Low end of the interval, included.

             stop (int): High end of the interval, excluded.

             step (int): Steps of the range to generate from.

            device: Device of the Array to be generated. (Not Implemented).

        Returns:
            The generated random valued Array.
    """
    res = Array()
    res.device = device
    res.dtype = float
    res.shape = tuple(shape)
    res.ndim = len(shape)
    res.size = 1
    for k in shape:
        res.size *= k

    res.values = __random.choices(range(start, stop, step), k=res.size)
    return res

def uniform(shape: Union[list, tuple], a: float, b: float, device=None) -> Array:
    """
        Generate an Array of random uniform values, from [a, b) interval.

        Args:
            shape (list | tuple): Shape of the Array to be generated.

            a (float): Low end of the interval, included.

            b (float): High end of the interval, excluded.

            device: Device of the Array to be generated. (Not Implemented).

        Returns:
            The generated random valued Array.
    """
    res = Array()
    res.device = device
    res.dtype = float
    res.shape = tuple(shape)
    res.ndim = len(shape)
    res.size = 1
    for k in shape:
        res.size *= k

    res.values = [__random.uniform(a, b) for k in range(res.size)]
    return res

