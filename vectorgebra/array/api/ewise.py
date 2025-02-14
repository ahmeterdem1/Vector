"""
    Functions referenced by: https://data-apis.org/array-api/latest/API_specification/elementwise_functions.html
"""

from ..ndarray import Array, Complex
from .. import (Variable, ADD, MUL, DIV, POW, SQRT, EXP, SIN,
                COS, ARCSIN, LOG2, LN, SIG, ARCSINH, ARCTAN,
                ARCCOS, TAN, COSH, ARCCOSH, ARCTANH, TANH, SINH,
                LOG10, LN1P)
from .dtype import *
import math
from builtins import round as __builtinRound
from builtins import abs as __builtinAbs
from typing import Union

def abs(x: Array) -> Array:
    """
        Calculates element-wise absolute value of the array.

        If Array has Variable-type values, they will be casted
        to the dtype of the first element of the array. Computational
        graph will be broken after a call to this function.

        Args:
            x (Array): Array to operate on.

        Returns:
            Array: Consists of absolute values of all values
                in the original array.

    """
    res = Array()
    res.dtype = x.dtype
    res.device = x.device
    res.size = x.size
    res.ndim = x.ndim
    res.shape = x.shape
    if res.dtype == Variable:
        res.dtype = type(x.values[0])
        res.values = [__builtinAbs(k.value) for k in x.values]
    else:
        res.values = [__builtinAbs(k) for k in x.values]
    return res

def acos(x: Array) -> Array:
    """
        Calculates element-wise arccos values of the array.

        If Array has Variable-type values, this operation
        will be registered to the computational graph.

        Args:
            x (Array): Array to operate on.

        OPCODE: ARCCOS - 14

        Returns:
            Array: arccos value of each element in the
                given array
    """
    res = Array()
    res.dtype = x.dtype
    res.device = x.device
    res.size = x.size
    res.ndim = x.ndim
    res.shape = x.shape

    if x.dtype == Variable:
        res.values = [Variable.buildUnaryOperation(k, math.acos, ARCCOS) for k in x.values]
    else:
        res.values = [math.acos(k) for k in x.values]

    return res

def acosh(x: Array):
    """
        Calculates element-wise arccosh values of the array.

        If Array has Variable-type values, this operation
        will be registered to the computational graph.

        Args:
            x (Array): Array to operate on.

        OPCODE: ARCCOSH - 17

        Returns:
            Array: arccosh value of each element in the
                given array
    """
    res = Array()
    res.dtype = x.dtype
    res.device = x.device
    res.size = x.size
    res.ndim = x.ndim
    res.shape = x.shape

    if x.dtype == Variable:
        res.values = [Variable.buildUnaryOperation(k, math.acosh, ARCCOSH) for k in x.values]
    else:
        res.values = [math.acosh(k) for k in x.values]

    return res

def add(x1: Array, x2: Array) -> Array:
    """
        Apply element-wise addition to given arrays.

        Simply returns x1 + x2. Arrays must be broadcastable.

        Raises:
            DimensionError: If broadcasting cannot be done
    """
    return x1 + x2

def asin(x: Array) -> Array:
    """
        Calculates element-wise arcsin values of the array.

        If Array has Variable-type values, this operation
        will be registered to the computational graph.

        Args:
            x (Array): Array to operate on.

        OPCODE: ARCSIN - 8

        Returns:
            Array: arcsin value of each element in the
                given array
    """
    res = Array()
    res.dtype = x.dtype
    res.device = x.device
    res.size = x.size
    res.ndim = x.ndim
    res.shape = x.shape

    if x.dtype == Variable:
        res.values = [Variable.buildUnaryOperation(k, math.asin, ARCSIN) for k in x.values]
    else:
        res.values = [math.asin(k) for k in x.values]

    return res

def asinh(x: Array) -> Array:
    """
        Calculates element-wise arcsinh values of the array.

        If Array has Variable-type values, this operation
        will be registered to the computational graph.

        Args:
            x (Array): Array to operate on.

        OPCODE: ARCSINH - 12

        Returns:
            Array: arcsinh value of each element in the
                given array
    """
    res = Array()
    res.dtype = x.dtype
    res.device = x.device
    res.size = x.size
    res.ndim = x.ndim
    res.shape = x.shape

    if x.dtype == Variable:
        res.values = [Variable.buildUnaryOperation(k, math.asinh, ARCSINH) for k in x.values]
    else:
        res.values = [math.asinh(k) for k in x.values]

    return res

def atan(x: Array) -> Array:
    """
        Calculates element-wise arctan values of the array.

        If Array has Variable-type values, this operation
        will be registered to the computational graph.

        Args:
            x (Array): Array to operate on.

        OPCODE: ARCTAN - 13

        Returns:
            Array: arctan value of each element in the
                given array
    """
    res = Array()
    res.dtype = x.dtype
    res.device = x.device
    res.size = x.size
    res.ndim = x.ndim
    res.shape = x.shape

    if x.dtype == Variable:
        res.values = [Variable.buildUnaryOperation(k, math.atan, ARCTAN) for k in x.values]
    else:
        res.values = [math.atan(k) for k in x.values]

    return res

def atan2(x1: Array, x2: Array) -> Array:
    """
        Calculates element-wise arctan of (x1 / x2).

        If Array has Variable-type values, this operation
        will be registered to the computational graph.

        Args:
            x1 (Array): First array argument.

            x2 (Array): Second array argument.

        Returns:
            Array: arctan of each element in (x1 / x2)

        Raises:
            DimensionError: When given x1 and x2 arrays are not
                broadcastable.
    """
    res = x1 / x2  # We will reuse the same array to save RAM

    if res.dtype == Variable:
        res.values = [Variable.buildUnaryOperation(k, math.atan, ARCTAN) for k in res.values]
    else:
        res.values = [math.atan(k) for k in res.values]

    return res

def atanh(x: Array) -> Array:
    """
        Calculates element-wise arctanh values of the array.

        If Array has Variable-type values, this operation
        will be registered to the computational graph.

        Args:
            x (Array): Array to operate on.

        OPCODE: ARCTANH - 18

        Returns:
            Array: arctanh value of each element in the
                given array
    """
    res = Array()
    res.dtype = x.dtype
    res.device = x.device
    res.size = x.size
    res.ndim = x.ndim
    res.shape = x.shape

    if x.dtype == Variable:
        res.values = [Variable.buildUnaryOperation(k, math.atanh, ARCTANH) for k in x.values]
    else:
        res.values = [math.atanh(k) for k in x.values]

    return res

def bitwise_and(x1: Array, x2: Array) -> Array:
    """
        Calculates bitwise and operation, on each
        element of given arrays. Simply calls x1 & x2.

        Args:
            x1 (Array): First array argument.

            x2 (Array): Second array argument.

        Raises:
            DimensionError: When given x1 and x2 are not
                broadcastable.
    """
    return x1 & x2

def bitwise_left_shift(x1: Array, x2: Union[INT_TYPE, Array]) -> Array:
    """
        Calculates bitwise left shift operation, on each
        element of given arrays. Simply calls x1 << x2.
        Shift amount is defined by x2, and dtype of x2 must
        be integer.

        Args:
            x1 (Array): First array argument.

            x2 (Array): Second array argument.

        Raises:
            DimensionError: When given x1 and x2 are not
                broadcastable.
    """
    return x1 << x2

def bitwise_invert(x: Array) -> Array:
    """
        Inverts each bit in each value in given array.

        Args:
            x (Array): Array to be operated on.

        Returns:
            Array: Basically, makes a call of ~x.
    """
    return ~x

def bitwise_or(x1: Array, x2: Array) -> Array:
    """
        Calculates bitwise or operation, on each
        element of given arrays. Simply calls x1 | x2.

        Args:
            x1 (Array): First array argument.

            x2 (Array): Second array argument.

        Raises:
            DimensionError: When given x1 and x2 are not
                broadcastable.
    """
    return x1 | x2

def bitwise_right_shift(x1: Array, x2: Union[INT_TYPE, Array]) -> Array:
    """
        Calculates bitwise right shift operation, on each
        element of given arrays. Simply calls x1 >> x2.
        Shift amount is defined by x2, and dtype of x2 must
        be integer.

        Args:
            x1 (Array): First array argument.

            x2 (Array): Second array argument.

        Raises:
            DimensionError: When given x1 and x2 are not
                broadcastable.
    """
    return x1 >> x2

def bitwise_xor(x1: Array, x2: Array) -> Array:
    """
        Calculates bitwise xor operation, on each
        element of given arrays. Simply calls x1 ^ x2.

        Args:
            x1 (Array): First array argument.

            x2 (Array): Second array argument.

        Raises:
            DimensionError: When given x1 and x2 are not
                broadcastable.
    """
    return x1 ^ x2

def ceil(x: Array) -> Array:
    """
        Calculates element-wise ceil value of the array.

        If Array has Variable-type values, they will be casted
        to the dtype of the first element of the array. Computational
        graph will be broken after a call to this function.

        Args:
            x (Array): Array to operate on.

        Returns:
            Array: Consists of ceil values of all values
                in the original array.
    """
    res = Array()
    res.dtype = x.dtype
    res.device = x.device
    res.size = x.size
    res.ndim = x.ndim
    res.shape = x.shape
    if x.dtype == Variable:
        res.dtype = type(x.values[0])
        res.values = [math.ceil(k.value) for k in x.values]
    else:
        res.values = [math.ceil(k) for k in x.values]
    return res

def clip(x: Array, min: Union[INT_TYPE, FLOAT_TYPE, Array, ...], max: Union[INT_TYPE, FLOAT_TYPE, Array, ...]) -> Array:
    # TODO: Implement clip method with proper logic
    raise NotImplementedError()

def conj(x: Array) -> Array:
    """
        Returns the element wise complex conjugate of
        given array, if it is complex valued. Else,
        returns a copy of it.

        Args:
            x (Array): Array to operate on

        Returns:
            Array: Element wise complex conjugate of x.
    """
    if x.dtype != complex or x.dtype != Complex:
        return +x
    res = Array()
    res.dtype = x.dtype
    res.device = x.device
    res.size = x.size
    res.ndim = x.ndim
    res.shape = x.shape
    res.values = [k.conjugate() for k in x.values]
    return res

def copysign(x1: Array, x2: Array) -> Array:
    """
        Operates element wise, copies sign from x2 to
        value of x1. Returns the resulting array. x1 and
        x2 must be broadcastable.

        Args:
            x1 (Array): Array to take values from.

            x2 (Array): Array to take signs from.

        Returns:
            Array: An array of x1 values with x2 signs.

        Raises:
            DimensionError: When x1 and x2 are not
                broadcastable.
    """
    res = Array()
    res.dtype = float
    res.device = x1.device

    if x1.shape == x2.shape:
        res.size = x1.size
        res.ndim = x1.ndim
        res.shape = x1.shape

        # Below line will be compiled by Python interpreter into a <listcomp>
        # so it will probably not evaluate the if-else at each iteration
        res.values = [math.copysign(x1.values[i] if x1.dtype != Variable else x1.values[i].value,
                                    x2.values[i] if x2.dtype != Variable else x2.values[i].value)
                      for i in range(res.size)]
    else:
        common_shape = Array.broadcast(x1, x2)
        c_x1 = x1
        c_x2 = x2
        if c_x1.shape != common_shape:
            c_x1 = c_x1.copy().broadcast_to(common_shape)
        if c_x2.shape != common_shape:
            c_x2 = c_x2.copy().broadcast_to(common_shape)

        res.size = c_x1.size
        res.ndim = c_x1.ndim
        res.shape = c_x1.shape
        res.values = [math.copysign(c_x1.values[i] if c_x1.dtype != Variable else c_x1.values[i].value,
                                    c_x2.values[i] if c_x2.dtype != Variable else c_x2.values[i].value)
                      for i in range(res.size)]

    return res

def cos(x: Array) -> Array:
    """
        Calculates element-wise cos values of the array.

        If Array has Variable-type values, this operation
        will be registered to the computational graph.

        Args:
            x (Array): Array to operate on.

        OPCODE: COS - 7

        Returns:
            Array: cos value of each element in the
                given array
    """
    res = Array()
    res.dtype = x.dtype
    res.device = x.device
    res.size = x.size
    res.ndim = x.ndim
    res.shape = x.shape

    if x.dtype == Variable:
        res.values = [Variable.buildUnaryOperation(k, math.cos, COS) for k in x.values]
    else:
        res.values = [math.cos(k) for k in x.values]

    return res

def cosh(x: Array) -> Array:
    """
        Calculates element-wise cosh values of the array.

        If Array has Variable-type values, this operation
        will be registered to the computational graph.

        Args:
            x (Array): Array to operate on.

        OPCODE: COSH - 16

        Returns:
            Array: cosh value of each element in the
                given array
    """
    res = Array()
    res.dtype = x.dtype
    res.device = x.device
    res.size = x.size
    res.ndim = x.ndim
    res.shape = x.shape

    if x.dtype == Variable:
        res.values = [Variable.buildUnaryOperation(k, math.cosh, COSH) for k in x.values]
    else:
        res.values = [math.cosh(k) for k in x.values]

    return res

def divide(x1: Array, x2: Array) -> Array:
    """
        Apply element-wise division to given arrays.

        Simply returns x1 / x2. Arrays must be broadcastable.

        Raises:
            DimensionError: If broadcasting cannot be done
    """
    return x1 / x2

def equal(x1: Array, x2: Array) -> Array:
    """
        Calculates equality comparison, on each
        element of given arrays. Simply calls x1 == x2.

        Args:
            x1 (Array): First array argument.

            x2 (Array): Second array argument.

        Raises:
            DimensionError: When given x1 and x2 are not
                broadcastable.
    """
    return x1 == x2

def exp(x: Array) -> Array:
    """
        Calculates element-wise exp values of the array.

        If Array has Variable-type values, this operation
        will be registered to the computational graph.

        Args:
            x (Array): Array to operate on.

        OPCODE: EXP - 5

        Returns:
            Array: exp value of each element in the
                given array
    """
    res = Array()
    res.dtype = x.dtype
    res.device = x.device
    res.size = x.size
    res.ndim = x.ndim
    res.shape = x.shape

    if x.dtype == Variable:
        res.values = [Variable.buildUnaryOperation(k, math.exp, EXP) for k in x.values]
    else:
        res.values = [math.exp(k) for k in x.values]

    return res

def expm1(x: Array) -> Array:
    """
        Calculates element-wise exp(x) - 1 values of the array.

        If Array has Variable-type values, this operation
        will be registered to the computational graph.

        Args:
            x (Array): Array to operate on.

        OPCODE: EXP - 5

        Returns:
            Array: exp(x) - 1 value of each element in the
                given array
    """
    res = Array()
    res.dtype = x.dtype
    res.device = x.device
    res.size = x.size
    res.ndim = x.ndim
    res.shape = x.shape

    # Derivative of exp(x) - 1 is the same as exp(x)
    if x.dtype == Variable:
        res.values = [Variable.buildUnaryOperation(k, math.expm1, EXP) for k in x.values]
    else:
        res.values = [math.expm1(k) for k in x.values]

    return res

def floor(x: Array) -> Array:
    """
        Calculates element-wise floor value of the array.

        If Array has Variable-type values, they will be casted
        to the dtype of the first element of the array. Computational
        graph will be broken after a call to this function.

        Args:
            x (Array): Array to operate on.

        Returns:
            Array: Consists of floor values of all values
                in the original array.

    """
    res = Array()
    res.dtype = x.dtype
    res.device = x.device
    res.size = x.size
    res.ndim = x.ndim
    res.shape = x.shape
    if x.dtype == Variable:
        res.dtype = type(x.values[0])
        res.values = [math.floor(k.value) for k in x.values]
    else:
        res.values = [math.floor(k) for k in x.values]
    return res

def floor_divide(x1: Array, x2: Array) -> Array:
    """
        Apply element-wise floor division to given arrays.

        Simply returns x1 // x2. Arrays must be broadcastable.

        Raises:
            DimensionError: If broadcasting cannot be done
    """
    return x1 // x2

def greater(x1: Array, x2: Array) -> Array:
    """
        Calculates gt comparison, on each
        element of given arrays. Simply calls x1 > x2.

        Args:
            x1 (Array): First array argument.

            x2 (Array): Second array argument.

        Raises:
            DimensionError: When given x1 and x2 are not
                broadcastable.
    """
    return x1 > x2

def greater_equal(x1: Array, x2: Array) -> Array:
    """
        Calculates geq comparison, on each
        element of given arrays. Simply calls x1 >= x2.

        Args:
            x1 (Array): First array argument.

            x2 (Array): Second array argument.

        Raises:
            DimensionError: When given x1 and x2 are not
                broadcastable.
    """
    return x1 >= x2

def hypot(x1: Array, x2: Array) -> Array:
    """
        Basically, calculates the L2 norm of the difference
        of the given arrays, when flattened. An element wise
        Euclidian distance calculation.

        Args:
            x1 (Array): First array argument.

            x2 (Array): Second array argument.

        Raises:
            DimensionError: When given x1 and x2 are not
                broadcastable.
    """
    res = Array()
    res.dtype = float
    res.device = None

    if x1.shape == x2.shape:
        res.size = x1.size
        res.ndim = x1.ndim
        res.shape = x1.shape

        res.values = [math.hypot(x1.values[i] if x1.dtype != Variable else x1.values[i].value,
                                 x2.values[i] if x2.dtype != Variable else x2.values[i].value)
                      for i in range(res.size)]

    else:
        common_shape = Array.broadcast(x1, x2)
        c_x1 = x1
        c_x2 = x2
        if c_x1.shape != common_shape:
            c_x1 = c_x1.copy().broadcast_to(common_shape)
        if c_x2.shape != common_shape:
            c_x2 = c_x2.copy().broadcast_to(common_shape)

        res.size = c_x1.size
        res.ndim = c_x1.ndim
        res.shape = c_x1.shape
        res.values = [math.hypot(c_x1.values[i] if c_x1.dtype != Variable else c_x1.values[i].value,
                                 c_x2.values[i] if c_x2.dtype != Variable else c_x2.values[i].value)
                      for i in range(res.size)]

    return res

def imag(x: Array) -> Array:
    """
        Returns the element wise imaginary parts of
        given array, if it is complex valued. Else,
        returns a copy of it.

        Args:
            x (Array): Array to operate on

        Returns:
            Array: Element wise imaginary parts of x.
    """
    if x.dtype != complex or x.dtype != Complex:
        return +x
    res = Array()
    res.dtype = x.dtype
    res.device = x.device
    res.size = x.size
    res.ndim = x.ndim
    res.shape = x.shape
    res.values = [k.imag for k in x.values] if x.dtype == complex else [k.imaginary for k in x.values]
    return res

def isfinite(x: Array) -> Array:
    """
        Applies element wise "is finite" check
        to given array. Creates a boolean array
        containing the results.

        Args:
            x (Array): Array to operate on.

        Returns:
              Array: Boolean array containing
                check results.
    """
    res = Array()
    res.dtype = bool
    res.device = x.device
    res.size = x.size
    res.ndim = x.ndim
    res.shape = x.shape
    if x.dtype == Variable:
        res.values = [math.isfinite(k.value) for k in x.values]
    else:
        res.values = [math.isfinite(k) for k in x.values]

    return res

def isinf(x: Array) -> Array:
    """
        Applies element wise "is infinite" check
        to given array. Creates a boolean array
        containing the results.

        Args:
            x (Array): Array to operate on.

        Returns:
              Array: Boolean array containing
                check results.
    """
    res = Array()
    res.dtype = bool
    res.device = x.device
    res.size = x.size
    res.ndim = x.ndim
    res.shape = x.shape
    if x.dtype == Variable:
        res.values = [math.isinf(k.value) for k in x.values]
    else:
        res.values = [math.isinf(k) for k in x.values]

    return res

def isnan(x: Array) -> Array:
    """
        Applies element wise "is nan" check
        to given array. Creates a boolean array
        containing the results.

        Args:
            x (Array): Array to operate on.

        Returns:
              Array: Boolean array containing
                check results.
    """
    res = Array()
    res.dtype = bool
    res.device = x.device
    res.size = x.size
    res.ndim = x.ndim
    res.shape = x.shape
    if x.dtype == Variable:
        res.values = [math.isnan(k.value) for k in x.values]
    else:
        res.values = [math.isnan(k) for k in x.values]

    return res

def less(x1: Array, x2: Array) -> Array:
    """
        Calculates lt comparison, on each
        element of given arrays. Simply calls x1 < x2.

        Args:
            x1 (Array): First array argument.

            x2 (Array): Second array argument.

        Raises:
            DimensionError: When given x1 and x2 are not
                broadcastable.
    """
    return x1 < x2

def less_equal(x1: Array, x2: Array) -> Array:
    """
        Calculates leq comparison, on each
        element of given arrays. Simply calls x1 <= x2.

        Args:
            x1 (Array): First array argument.

            x2 (Array): Second array argument.

        Raises:
            DimensionError: When given x1 and x2 are not
                broadcastable.
    """
    return x1 <= x2

def log(x: Array) -> Array:
    """
        Calculates element-wise natural log values of the array.

        If Array has Variable-type values, this operation
        will be registered to the computational graph.

        Args:
            x (Array): Array to operate on.

        OPCODE: LN - 10

        Returns:
            Array: Natural log value of each element in the
                given array
    """
    res = Array()
    res.dtype = x.dtype
    res.device = x.device
    res.size = x.size
    res.ndim = x.ndim
    res.shape = x.shape

    if x.dtype == Variable:
        res.values = [Variable.buildUnaryOperation(k, math.log, LN) for k in x.values]
    else:
        res.values = [math.log(k) for k in x.values]

    return res

def log1p(x: Array) -> Array:
    """
        Calculates element-wise ln(x + 1) values of the array.

        If Array has Variable-type values, this operation
        will be registered to the computational graph.

        Args:
            x (Array): Array to operate on.

        OPCODE: LN1P - 22

        Returns:
            Array: ln(x + 1) value of each element in the
                given array
    """
    res = Array()
    res.dtype = x.dtype
    res.device = x.device
    res.size = x.size
    res.ndim = x.ndim
    res.shape = x.shape

    if x.dtype == Variable:

        res.values = [Variable.buildUnaryOperation(k, math.log1p, LN1P) for k in x.values]
    else:
        res.values = [math.log1p(k) for k in x.values]

    return res

def log2(x: Array) -> Array:
    """
        Calculates element-wise log base 2 values of the array.

        If Array has Variable-type values, this operation
        will be registered to the computational graph.

        Args:
            x (Array): Array to operate on.

        OPCODE: LOG2 - 9

        Returns:
            Array: log base 2 value of each element in the
                given array
    """
    res = Array()
    res.dtype = x.dtype
    res.device = x.device
    res.size = x.size
    res.ndim = x.ndim
    res.shape = x.shape

    if x.dtype == Variable:
        res.values = [Variable.buildUnaryOperation(k, math.log2, LOG2) for k in x.values]
    else:
        res.values = [math.log2(k) for k in x.values]

    return res

def log10(x: Array) -> Array:
    """
        Calculates element-wise log base 10 values of the array.

        If Array has Variable-type values, this operation
        will be registered to the computational graph.

        Args:
            x (Array): Array to operate on.

        OPCODE: LOG10 - 21

        Returns:
            Array: log base 10 value of each element in the
                given array
    """
    res = Array()
    res.dtype = x.dtype
    res.device = x.device
    res.size = x.size
    res.ndim = x.ndim
    res.shape = x.shape

    if x.dtype == Variable:
        res.values = [Variable.buildUnaryOperation(k, math.log10, LOG10) for k in x.values]
    else:
        res.values = [math.log10(k) for k in x.values]

    return res

def logaddexp(x1: Array, x2: Array) -> Array:
    """
        Calculates element-wise log(exp(x1) + exp(x2)) values of the array.

        If Array has Variable-type values, this operation
        will be registered to the computational graph.

        Args:
            x1 (Array): First array argument.

            x2 (Array): Second array argument.

        OPCODE: LOGADDEXP - 23

        Returns:
            Array: log(exp(x1) + exp(x2)) value of each element in the
                given array
    """
    # Reason being, computational graph of Variable can only
    # be preserved in a neat way, this way.
    return log(exp(x1) + exp(x2))

def logical_and(x1: Array, x2: Array) -> Array:
    """
        Calculates logical and operation, on each
        element of given arrays. Simply calls x1 and x2.

        Args:
            x1 (Array): First array argument.

            x2 (Array): Second array argument.

        Raises:
            DimensionError: When given x1 and x2 are not
                broadcastable.
    """
    return x1 and x2

def logical_or(x1: Array, x2: Array) -> Array:
    """
        Calculates logical or operation, on each
        element of given arrays. Simply calls x1 or x2.

        Args:
            x1 (Array): First array argument.

            x2 (Array): Second array argument.

        Raises:
            DimensionError: When given x1 and x2 are not
                broadcastable.
    """
    return x1 or x2

def logical_xor(x1: Array, x2: Array) -> Array:
    """
        Calculates logical xor operation, on each
        element of given arrays. Simply calls x1 xor x2.

        Args:
            x1 (Array): First array argument.

            x2 (Array): Second array argument.

        Raises:
            DimensionError: When given x1 and x2 are not
                broadcastable.
    """
    return (x1 and x2) or not (x1 or x2)

def maximum(x1: Array, x2: Array) -> Array:
    """
        Broadcasts x1 and x2, and picks the maximum
        value from same indexed value couples element
        wise. Returns the new generated array.

        Args:
            x1 (Array): First array argument.

            x2 (Array): Second array argument.

        Returns:
            Array: Array in the broadcasted shape, containing
                maximum values from broadcasted x1-x2 couples.

        Raises:
            DimensionError: When given x1 and x2 are not
                broadcastable.

        Notes:
            This function does not yet implement type promotion
            in the proper manner. Returned array will have the
            same dtype as x1.
    """
    res = Array()
    # TODO: Apply type promotion rules
    res.dtype = x1.dtype
    res.device = None

    if x1.shape == x2.shape:
        res.size = x1.size
        res.ndim = x1.ndim
        res.shape = x1.shape

        res.values = [max(x1.values[i] if x1.dtype != Variable else x1.values[i].value,
                      x2.values[i] if x2.dtype != Variable else x2.values[i].value)
                      for i in range(res.size)]

    else:
        common_shape = Array.broadcast(x1, x2)
        c_x1 = x1
        c_x2 = x2
        if c_x1.shape != common_shape:
            c_x1 = c_x1.copy().broadcast_to(common_shape)
        if c_x2.shape != common_shape:
            c_x2 = c_x2.copy().broadcast_to(common_shape)

        res.size = c_x1.size
        res.ndim = c_x1.ndim
        res.shape = c_x1.shape
        res.values = [max(c_x1.values[i] if c_x1.dtype != Variable else c_x1.values[i].value,
                      c_x2.values[i] if c_x2.dtype != Variable else c_x2.values[i].value)
                      for i in range(res.size)]

    return res

def minimum(x1: Array, x2: Array) -> Array:
    """
        Broadcasts x1 and x2, and picks the minimum
        value from same indexed value couples element
        wise. Returns the new generated array.

        Args:
            x1 (Array): First array argument.

            x2 (Array): Second array argument.

        Returns:
            Array: Array in the broadcasted shape, containing
                minimum values from broadcasted x1-x2 couples.

        Raises:
            DimensionError: When given x1 and x2 are not
                broadcastable.

        Notes:
            This function does not yet implement type promotion
            in the proper manner. Returned array will have the
            same dtype as x1.
    """
    res = Array()
    # TODO: Apply type promotion rules
    res.dtype = x1.dtype
    res.device = None

    if x1.shape == x2.shape:
        res.size = x1.size
        res.ndim = x1.ndim
        res.shape = x1.shape

        res.values = [min(x1.values[i] if x1.dtype != Variable else x1.values[i].value,
                          x2.values[i] if x2.dtype != Variable else x2.values[i].value)
                      for i in range(res.size)]

    else:
        common_shape = Array.broadcast(x1, x2)
        c_x1 = x1
        c_x2 = x2
        if c_x1.shape != common_shape:
            c_x1 = c_x1.copy().broadcast_to(common_shape)
        if c_x2.shape != common_shape:
            c_x2 = c_x2.copy().broadcast_to(common_shape)

        res.size = c_x1.size
        res.ndim = c_x1.ndim
        res.shape = c_x1.shape
        res.values = [min(c_x1.values[i] if c_x1.dtype != Variable else c_x1.values[i].value,
                          c_x2.values[i] if c_x2.dtype != Variable else c_x2.values[i].value)
                      for i in range(res.size)]

    return res

def multiply(x1: Array, x2: Array):
    """
        Apply element-wise multiplication to given arrays.

        Simply returns x1  x2. Arrays must be broadcastable.

        Raises:
            DimensionError: If broadcasting cannot be done
    """
    return x1 * x2

def negative(x: Array):
    """
        Returns the element-wise negation of given
        array. Simply returns -x.
    """
    return -x

def not_equal(x1: Array, x2: Array) -> Array:
    """
        Calculates ne comparison, on each
        element of given arrays. Simply calls x1 != x2.

        Args:
            x1 (Array): First array argument.

            x2 (Array): Second array argument.

        Raises:
            DimensionError: When given x1 and x2 are not
                broadcastable.
    """
    return x1 != x2

def positive(x: Array):
    """
        Returns the element wise positive of given
        array. Simply returns +x, which is identical
        to Array.copy() operation.
    """
    return +x

def pow(x1: Array, x2: Array):
    """
        Apply element-wise exponentiation operation to given arrays.
        Simply returns x1 ** x2. Arrays must be broadcastable.

        OPCODE: POW - 3

        Args:
            x1 (Array): First array argument.

            x2 (Array): Second array argument.

        Raises:
            DimensionError: If broadcasting cannot be done
    """
    res = Array()
    res.device = None

    if x1.shape == x2.shape:
        res.size = x1.size
        res.ndim = x1.ndim
        res.shape = x1.shape

        res.values = [x1.values[i] ** x2.values[i] for i in range(res.size)]

    else:
        common_shape = Array.broadcast(x1, x2)
        c_x1 = x1
        c_x2 = x2
        if c_x1.shape != common_shape:
            c_x1 = c_x1.copy().broadcast_to(common_shape)
        if c_x2.shape != common_shape:
            c_x2 = c_x2.copy().broadcast_to(common_shape)

        res.size = c_x1.size
        res.ndim = c_x1.ndim
        res.shape = c_x1.shape
        # There is no Variable check, Variable already implements pow
        res.values = [c_x1.values[i] ** c_x2.values[i] for i in range(res.size)]

    res.dtype = type(res.values[0])
    return res

def real(x: Array) -> Array:
    """
        Returns the element wise real part of
        given array, if it is complex valued. Else,
        returns a copy of it.

        Args:
            x (Array): Array to operate on

        Returns:
            Array: Element wise real parts of x.
    """
    if x.dtype != complex or x.dtype != Complex:
        return +x
    res = Array()
    res.dtype = x.dtype
    res.device = x.device
    res.size = x.size
    res.ndim = x.ndim
    res.shape = x.shape
    res.values = [k.real for k in x.values]
    return res

def remainder(x1: Array, x2: Array) -> Array:
    """
        Apply element-wise remainder operation to given arrays.
        Simply returns x1 % x2. Arrays must be broadcastable.

        If Array has Variable-type values, they will be casted
        to int type. Computational graph will be broken after
        a call to this function.

        Args:
            x1 (Array): First array argument.

            x2 (Array): Second array argument.

        Raises:
            DimensionError: If broadcasting cannot be done
    """
    res = Array()
    res.dtype = int
    res.device = None

    if x1.shape == x2.shape:
        res.size = x1.size
        res.ndim = x1.ndim
        res.shape = x1.shape

        res.values = [x1.values[i] if x1.dtype != Variable else x1.values[i].value %
                      x2.values[i] if x2.dtype != Variable else x2.values[i].value
                      for i in range(res.size)]

    else:
        common_shape = Array.broadcast(x1, x2)
        c_x1 = x1
        c_x2 = x2
        if c_x1.shape != common_shape:
            c_x1 = c_x1.copy().broadcast_to(common_shape)
        if c_x2.shape != common_shape:
            c_x2 = c_x2.copy().broadcast_to(common_shape)

        res.size = c_x1.size
        res.ndim = c_x1.ndim
        res.shape = c_x1.shape
        res.values = [c_x1.values[i] if c_x1.dtype != Variable else c_x1.values[i].value %
                      c_x2.values[i] if c_x2.dtype != Variable else c_x2.values[i].value
                      for i in range(res.size)]

    return res

def round(x: Array):
    """
        Returns element wise rounded array from
        given array.

        If Array has Variable-type values, they will be casted
        to the dtype of the first element of the array. Computational
        graph will be broken after a call to this function.

        Args:
            x (Array): Array to operate on.

        Returns:
            Array: Consists of rounded values of all values
                in the original array.
    """
    res = Array()
    res.dtype = x.dtype
    res.device = x.device
    res.size = x.size
    res.ndim = x.ndim
    res.shape = x.shape

    if x.dtype == Variable:
        res.dtype = type(x.values[0].value)
        res.values = [__builtinRound(k.value) for k in x.values]
    else:
        res.values = [__builtinRound(k) for k in x.values]

    return res

def sign(x: Array) -> Array:
    """
        Calculates sign function element wise on the
        given array, returns the outputs in an array
        with the same dtype as the given one.

        Args:
            x (Array): Array to operate on.

        Returns:
            Array: Containing sign(x) results, with
                same dtype as given argument.
    """
    res = Array()
    res.dtype = x.dtype
    res.device = x.device
    res.size = x.size
    res.ndim = x.ndim
    res.shape = x.shape

    if x.dtype == Variable:
        res.values = [0 if k.value == 0 else k.value / __builtinAbs(k.value) for k in x.values]
    else:
        res.values = [0 if k == 0 else k / __builtinAbs(k) for k in x.values]

def signbit(x: Array) -> Array:
    return x < 0

def sin(x: Array) -> Array:
    """
        Calculates element-wise sin values of the array.

        If Array has Variable-type values, this operation
        will be registered to the computational graph.

        Args:
            x (Array): Array to operate on.

        OPCODE: SIN - 6

        Returns:
            Array: sin value of each element in the
                given array
    """
    res = Array()
    res.dtype = x.dtype
    res.device = x.device
    res.size = x.size
    res.ndim = x.ndim
    res.shape = x.shape

    if x.dtype == Variable:
        res.values = [Variable.buildUnaryOperation(k, math.sin, SIN) for k in x.values]
    else:
        res.values = [math.sin(k) for k in x.values]

    return res

def sinh(x: Array) -> Array:
    """
        Calculates element-wise sinh values of the array.

        If Array has Variable-type values, this operation
        will be registered to the computational graph.

        Args:
            x (Array): Array to operate on.

        OPCODE: SINH - 19

        Returns:
            Array: sinh value of each element in the
                given array
    """
    res = Array()
    res.dtype = x.dtype
    res.device = x.device
    res.size = x.size
    res.ndim = x.ndim
    res.shape = x.shape

    if x.dtype == Variable:
        res.values = [Variable.buildUnaryOperation(k, math.sinh, SINH) for k in x.values]
    else:
        res.values = [math.sinh(k) for k in x.values]

    return res

def square(x: Array):
    """
        Calculates element-wise squared values of the array.

        If Array has Variable-type values, this operation
        will be registered to the computational graph.

        Args:
            x (Array): Array to operate on.

        OPCODE: POW - 3

        Returns:
            Array: Squared value of each element in the
                given array
    """
    res = Array()
    res.dtype = x.dtype
    res.device = x.device
    res.size = x.size
    res.ndim = x.ndim
    res.shape = x.shape
    res.values = [k ** 2 for k in x.values]
    return res

def sqrt(x: Array) -> Array:
    """
        Calculates element-wise sqrt values of the array.

        If Array has Variable-type values, this operation
        will be registered to the computational graph.

        Args:
            x (Array): Array to operate on.

        OPCODE: SQRT - 4

        Returns:
            Array: sqrt value of each element in the
                given array
    """
    res = Array()
    res.dtype = x.dtype
    res.device = x.device
    res.size = x.size
    res.ndim = x.ndim
    res.shape = x.shape

    if x.dtype == Variable:
        res.values = [Variable.buildUnaryOperation(k, math.sqrt, SQRT) for k in x.values]
    else:
        res.values = [math.sqrt(k) for k in x.values]

    return res

def subtract(x1: Array, x2: Array) -> Array:
    """
        Apply element-wise subtraction to given arrays.

        Simply returns x1 - x2. Arrays must be broadcastable.

        Raises:
            DimensionError: If broadcasting cannot be done
    """
    return x1 - x2

def tan(x: Array) -> Array:
    """
        Calculates element-wise tan values of the array.

        If Array has Variable-type values, this operation
        will be registered to the computational graph.

        Args:
            x (Array): Array to operate on.

        OPCODE: TAN - 15

        Returns:
            Array: tan value of each element in the
                given array
    """
    res = Array()
    res.dtype = x.dtype
    res.device = x.device
    res.size = x.size
    res.ndim = x.ndim
    res.shape = x.shape

    if x.dtype == Variable:
        res.values = [Variable.buildUnaryOperation(k, math.tan, TAN) for k in x.values]
    else:
        res.values = [math.tan(k) for k in x.values]

    return res

def tanh(x: Array) -> Array:
    """
        Calculates element-wise tanh values of the array.

        If Array has Variable-type values, this operation
        will be registered to the computational graph.

        Args:
            x (Array): Array to operate on.

        OPCODE: TANH - 20

        Returns:
            Array: tanh value of each element in the
                given array
    """
    res = Array()
    res.dtype = x.dtype
    res.device = x.device
    res.size = x.size
    res.ndim = x.ndim
    res.shape = x.shape

    if x.dtype == Variable:
        res.values = [Variable.buildUnaryOperation(k, math.tanh, TANH) for k in x.values]
    else:
        res.values = [math.tanh(k) for k in x.values]

    return res

def trunc(x: Array) -> Array:
    """
        Returns element wise truncated array from
        given array.

        If Array has Variable-type values, they will be casted
        to the dtype of the first element of the array. Computational
        graph will be broken after a call to this function.

        Args:
            x (Array): Array to operate on.

        Returns:
            Array: Consists of truncated values of all values
                in the original array.
    """
    res = Array()
    res.dtype = x.dtype
    res.device = x.device
    res.size = x.size
    res.ndim = x.ndim
    res.shape = x.shape

    if x.dtype == Variable:
        res.dtype = type(x.values[0].value)
        res.values = [math.trunc(k.value) for k in x.values]
    else:
        res.values = [math.trunc(k) for k in x.values]

    return res
