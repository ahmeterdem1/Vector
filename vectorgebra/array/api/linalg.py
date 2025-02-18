"""
    Linear algebra extension of Array API referenced by: https://data-apis.org/array-api/latest/extensions/linear_algebra_functions.html

    Algorithms provided here are specifically optimized for matrices, arrays having ndim=2.
"""

import math
from ..ndarray import Array
from .. import DimensionError
from operator import mul as _mul
from itertools import product as _product
from typing import Union, Tuple

def cholesky():
    pass

def cross():
    pass

def det(x: Array, epsilon: float = 1e-07) -> Array:
    """
        Calculate the determinant of a matrix, or a structured stack of matrices.

        Args:
            x (Array): Matrix, or matrices to calculate the determinant of.

            epsilon (float): Numerical margin to prevent zero divisions.

        Returns:
            An array of calculated determinants, conforming to the shape
            of the given array.

        Raises:
            DimensionError: If the matrix or matrices are not square.
    """

    if x.ndim > 2:
        dets = []
        query_shapes = [range(x.shape[i]) if i < x.ndim - 2 else [slice(0, x.shape[i], 1)] for i in range(x.ndim)]
        factor: x.dtype

        for shape in _product(*query_shapes):
            c_x = x[*shape].copy()

            M = c_x.shape[1]
            N = c_x.shape[0]

            if N != M:
                raise DimensionError(2)

            for j in range(N):
                for k in range(N):
                    if j == k:
                        continue
                    factor = (c_x.values[j + j * M] + epsilon) / (c_x.values[j + k * M] + epsilon)  # row1 / row2

                    if math.isclose(factor, 0):
                        continue

                    c_x.values[j + k * M:j + k * M + M - j] = [val2 - val1 / factor for val1, val2 in
                                                               zip(c_x.values[j + j * M:j + j * M + M - j],
                                                                   c_x.values[j + k * M:j + k * M + M - j])]
            _det = 1
            for i in range(N):
                _det *= c_x.values[i + i * M]
            dets.append(_det)

        res = Array()
        res.device = x.device
        res.values = dets
        res.ndim = x.ndim - 2
        res.shape = x.shape[:-2]
        res.size = math.prod(res.shape)
        return res

    M = x.shape[1]
    N = x.shape[0]

    if N != M:
        raise DimensionError(2)

    c_x = x.copy()
    factor: x.dtype

    for j in range(N):
        for k in range(N):
            if j == k:
                continue
            factor = (c_x.values[j + j * M] + epsilon) / (c_x.values[j + k * M] + epsilon)  # row1 / row2

            if math.isclose(factor, 0):
                continue

            c_x.values[j + k * M:j + k * M + M - j] = [val2 - val1 / factor for val1, val2 in
                                                       zip(c_x.values[j + j * M:j + j * M + M - j],
                                                           c_x.values[j + k * M:j + k * M + M - j])]

    _det = 1
    for i in range(N):
        _det *= c_x.values[i + i * M]  # diagonals
    return Array([_det])

def diagonal(x: Array, offset: int = 0) -> Array:
    """
        Retrieves the diagonals of the given matrix, or structured stack of matrices.

        Args:
            x (Array): Matrix, or stack of matrices to retrieve the diagonals of

            offset (int): Diagonal offset relative to the main diagonal

        Returns:
            An array containing the retrieved diagonals, conforming to the shape of
            the given matrix or stack of matrices.
    """

    if x.ndim > 2:
        res = Array()
        res.shape = (*x.shape[:-2], max(x.shape[-2], x.shape[-1]))
        res.ndim = x.ndim - 1
        res.values = []
        res.device = x.device

        query_shapes = [range(x.shape[i]) if i < x.ndim - 2 else [slice(0, x.shape[i], 1)] for i in range(x.ndim)]

        M: int
        for shape in _product(*query_shapes):
            x_ = x[*shape]
            M = x_.shape[1]
            res.values.extend(x_.values[i + i * M + offset] for i in range(x_.shape[0]))

        res.size = len(res.values)
        res.dtype = type(res.values[0])
        return res

    res = Array()
    M = x.shape[1]
    res.values = [x.values[i + i * M + offset] for i in range(x.shape[0])]
    res.size = len(res.values)
    res.ndim = 1
    res.shape = (res.size,)
    res.dtype = type(res.values[0])
    res.device = x.device
    return res

def echelon(x: Array, epsilon: float = 1e-07) -> Array:
    """
        Converts the given matrix into (reduced) echelon form.

        Args:
            x (Array): The array to be converted to echelon form.

            epsilon (float): The margin value to prevent zero divisions
                when calculating row coefficients.

        Returns:
            The corresponding echelon form of the given matrix, calculated
            from a copy of it.
    """
    c_x = x.copy()
    factor: x.dtype
    M = x.shape[1]
    N = x.shape[0]

    for j in range(N):
        for k in range(N):
            if j == k:
                continue
            factor = (c_x.values[j + j * M] + epsilon) / (c_x.values[j + k * M] + epsilon)  # row1 / row2

            if math.isclose(factor, 0):
                continue

            c_x.values[j + k * M:j + k * M + M - j] = [val2 - val1 / factor for val1, val2 in zip(c_x.values[j + j * M:j + j * M + M - j],
                                                                                                  c_x.values[j + k * M:j + k * M + M - j])]
    return c_x

def eigh():
    pass

def eigvalsh():
    pass

def inv():
    pass

def lu():
    pass

def matmul(x1: Array, x2: Array) -> Array:
    reduced_shape_other = [[slice(0, x2.shape[0], 1)], range(x2.shape[1])]
    reduced_shape_self = [range(x1.shape[0]), [slice(0, x1.shape[1], 1)]]
    res = Array()

    res.values = [
        sum(map(_mul, x1.__fast_matrix_query_end(self_shape_), x2.__fast_matrix_query_begin(other_shape_)))
        for self_shape_ in _product(*reduced_shape_self)
        for other_shape_ in _product(*reduced_shape_other)
    ]

    res.dtype = type(res.values[0])
    res.device = x1.device
    res.ndim = 2
    res.size = x1.shape[0] * x2.shape[1]
    res.shape = (x1.shape[0], x2.shape[1])
    return res

def matrix_norm():
    pass

def matrix_power():
    pass

def matrix_rank():
    pass

def matrix_transpose():
    pass

def norm(x: Array, axis: Union[int, Tuple[int]] = None, p: Union[int, str] = 2):
    pass

def outer():
    pass

def pinv():
    pass

def qr():
    pass

def slogdet():
    pass

def solve():
    pass

def svd():
    pass

def svdvals():
    pass

def tensordot():
    pass

def trace():
    pass

def vecdot():
    pass

def vector_norm():
    pass
