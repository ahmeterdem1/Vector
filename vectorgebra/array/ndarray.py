"""
    Base n-dimensional array implementation conforming to
    Python Array API v2023.12

    https://data-apis.org/array-api/latest/index.html
"""

# Some of below imports are not used here, but used in api module
from ..utils import *
from ..math import abs, Complex
from typing import Type, Union, Callable
from copy import deepcopy, copy
from decimal import Decimal
from ..variable import Variable

BASIC_ITERABLE = Union[list, tuple]
__NAMESPACE = None

class Array:

    """
        Base implementation of n-dimensional arrays.

        Attributes:
            dtype (Type): Data type contained in the array
            device (None): Currently not implemented, left as None. It exists to conform to Array API
            ndim (int): Number of dimensions contained in the array (length of shape)
            shape (tuple): A tuple containing the dimensionality of each axes in the array
            size (int): Total amount of data points of type *dtype* stored in the array
            values (list): Core list containing all the numerical data in the array
    """

    dtype: Type
    device = None  # No device support, yet.
    ndim: int
    shape: tuple
    size: int
    values: list

    def __init__(self, data=None, dtype: Union[Type, None] = None,
                 device=None):
        """
            Create an N dimensional array.

            - If data is none, an empty array with no dimension is created.
            - If data is a list or tuple, corresponding N dimensional array is created from the nested list structure.
            - If none above, an array with the deepcopy of the given data, with no dimensionality is created.

        """

        self.device = device if device is not None else None

        if data is None:
            self.dtype = object
            self.ndim = 0
            self.shape = (0,)
            self.size = 0
            self.values = None
        elif not isinstance(data, BASIC_ITERABLE):
            self.dtype = type(data)
            self.ndim = 0
            self.shape = (0,)
            self.size = 0
            self.values = deepcopy(data)
        else:
            if dtype is None:
                self.values = Array.flatten(data)  # A linear array
            else:
                self.values = [dtype(k) for k in Array.flatten(data)]

            shape = []
            temp = data

            while isinstance(temp, BASIC_ITERABLE) and len(temp):
                shape.append(len(temp))
                temp = temp[0]

            if isinstance(temp, Union[list, tuple]):
                # Since the subarrays must be aligned, being here
                # means that the array must be empty
                shape.append(0)
                self.dtype = object
            elif dtype is not None:
                self.dtype = dtype
            else:
                self.dtype = type(temp)

            self.shape = tuple(shape)
            self.ndim = len(self.shape)

            self.size = 1
            for k in self.shape:
                self.size *= k

    @staticmethod
    def flatten(data: BASIC_ITERABLE) -> list:
        """
            Flattens a nested Python list object, creating
            a linear list.

            Args:
                 data (Union[list, tuple]):
        """
        result = []
        for element in data:
            if isinstance(element, BASIC_ITERABLE):
                result.extend(Array.flatten(element))
            else:
                result.append(element)
        return result

    @staticmethod
    def broadcast(_array_1, _array_2) -> tuple:
        """
            Broadcasting algorithm specified by Python Array API v2023.12
            at [here](https://data-apis.org/array-api/latest/API_specification/broadcasting.html#broadcasting).

            Args:
                _array_1 (Array): First array to be broadcasted
                _array_2 (Array): Second array to be broadcasted

            Returns:
                The common shape, as a tuple, that is the result of the
                broadcasting.

            Raises:
                DimensionError: If broadcasting is not viable
        """
        N = max(_array_1.ndim, _array_2.ndim)
        n1: int
        n2: int
        d1: int
        d2: int
        shape = [None for k in range(N)]
        for i in range(N-1, -1, -1):
            n1 = _array_1.ndim - N + i
            d1 = _array_1.shape[n1] if n1 >= 0 else 1

            n2 = _array_2.ndim - N + i
            d2 = _array_2.shape[n2] if n2 >= 0 else 1

            if d1 == 1:
                shape[i] = d2
            elif d2 == 1 or d1 == d2:
                shape[i] = d1
            else:
                raise DimensionError(0)
        return tuple(shape)

    @property
    def mT(self):
        """
            Returns the matrix transpose of self.

            Example:
                (..., a, b) -> (..., b, a)

            Returns:
                New array representing the matrix transpose of self.

        """
        res = Array()
        res.dtype = self.dtype
        res.device = self.device
        res.ndim = self.ndim
        shape = [k for k in self.shape]
        __temp = shape[-1]
        shape[-1] = shape[-2]
        shape[-2] = __temp
        res.shape = tuple(shape)
        res.size = self.size
        N = self.shape[-1] * self.shape[-2]
        M = self.size // N
        data = self.values.copy()
        count: int
        for k in range(M):
            count = 0
            for i in range(self.shape[-1]):
                for j in range(self.shape[-2]):
                    data[k * N + count] = self.values[k * N + j * self.shape[-1] + i]
                    count += 1
        res.values = data
        return res

    @property
    def T(self):
        # TODO: Fix this property, this is still from nested list implementation
        res = Array()
        res.dtype = self.dtype
        res.device = self.device
        res.ndim = self.ndim
        res.shape = tuple([k for k in self.shape[::-1]])
        res.size = self.size
        res.values = Array.__transpose(deepcopy(self.values))
        #return res
        raise NotImplementedError()

    @staticmethod
    def __transpose(arraylike: Union[list, tuple]):

        if isinstance(arraylike[0], Union[list, tuple]):
            return [[arraylike[l][k] for l in range(len(arraylike))] for k in range(len(arraylike[0]))]

        transposed = []
        for colidx in range(len(arraylike[0])):
            transposed.append([Array.__transpose(row[colidx]) for row in arraylike])
        return transposed

    def __abs__(self):
        """
            Takes the absolute value of self. Does not modify
            it, returns a new one.

            Based on Python Array API v2023.12

            https://data-apis.org/array-api/latest/API_specification/generated/array_api.array.__abs__.html
        """
        res = Array()
        res.device = self.device
        res.ndim = self.ndim
        res.size = self.size
        res.shape = copy(self.shape)
        res.values = list(map(abs, self.values))
        return res

    def __add__(self, other):
        """
            Add 2 arrays, or add a numerical object to self.
            Broadcasting is applied when necessary if given an array.

            Returns:
                Element-wise addition of self and other (array or numerical)
                as an array

            Raises:
                DimensionError: If broadcasting is not viable.

            This method is based on Python Array API v2023.12

            https://data-apis.org/array-api/latest/API_specification/generated/array_api.array.__add__.html

        """
        if isinstance(other, Array):
            c_other = other.copy()
            c_self = self
            if other.shape != self.shape:
                common_shape = Array.broadcast(self, c_other)
                if c_other.shape != common_shape:
                    c_other.broadcast_to(common_shape)
                if self.shape != common_shape:
                    c_self = c_self.copy()
                    c_self.broadcast_to(common_shape)

            c_other.values = [c_other.values[i] + c_self.values[i] for i in range(c_self.size)]

            return c_other

        else:
            res = Array()
            res.device = self.device
            res.ndim = self.ndim
            res.size = self.size
            res.shape = copy(self.shape)
            res.values = [k + other for k in self.values]
            return res

    def __and__(self, other):
        """
            Calculate element-wise "and" operation with self and other.
            Broadcasting is applied when necessary if given an array.

            Returns:
                Element-wise "and" of self and other (array or numerical)
                as an array

            Raises:
                DimensionError: If broadcasting is not viable.

            This method is based on Python Array API v2023.12

            https://data-apis.org/array-api/latest/API_specification/generated/array_api.array.__and__.html
        """
        if isinstance(other, Array):
            c_other = other.copy()
            c_self = self
            if other.shape != self.shape:
                common_shape = Array.broadcast(self, c_other)
                if c_other.shape != common_shape:
                    c_other.broadcast_to(common_shape)
                if self.shape != common_shape:
                    c_self = c_self.copy()
                    c_self.broadcast_to(common_shape)

            c_other.values = [c_other.values[i] & c_self.values[i] for i in range(c_self.size)]

            c_other.dtype = bool

            return c_other
        else:
            res = Array()
            res.dtype = bool
            res.device = self.device
            res.ndim = self.ndim
            res.shape = copy(self.shape)
            res.size = self.size
            res.values = [k & other for k in self.values]
            return res

    def __array_namespace__(self, api_version: str = None):
        if api_version is None or api_version == "2023.12":
            global __NAMESPACE
            return __NAMESPACE
        raise NotImplementedError()

    def __bool__(self):
        """
            Converts self to bool, if self has no dimensionality.

            This method is based on Python Array API v2023.12

            https://data-apis.org/array-api/latest/API_specification/generated/array_api.array.__bool__.html
        """
        if self.shape == (0,):
            return bool(self.values)

    def __complex__(self):
        """
            Converts self to complex, if self has no dimensionality.

            This method is based on Python Array API v2023.12

            https://data-apis.org/array-api/latest/API_specification/generated/array_api.array.__complex__.html
        """
        if self.shape == (0,):
            return complex(self.values)

    def __dlpack__(self, *args, **kwargs):
        raise NotImplementedError()

    def __dlpack_device__(self):
        raise NotImplementedError()

    def __eq__(self, other):
        """
            Calculate element-wise equality comparisons of self and other.
            Broadcasting is applied when necessary if an array is given.

            Returns:
                Element-wise equality values of self and other (array or numerical)
                as an array

            Raises:
                DimensionError: If broadcasting is not viable.

            This method is based on Python Array API v2023.12

            https://data-apis.org/array-api/latest/API_specification/generated/array_api.array.__eq__.html
        """
        if isinstance(other, Array):
            c_other = other.copy()
            c_self = self
            if other.shape != self.shape:
                common_shape = Array.broadcast(self, c_other)
                if c_other.shape != common_shape:
                    c_other.broadcast_to(common_shape)
                if self.shape != common_shape:
                    c_self = c_self.copy()
                    c_self.broadcast_to(common_shape)

            c_other.values = [c_other.values[i] == c_self.values[i] for i in range(c_self.size)]

            c_other.dtype = bool

            return c_other
        else:
            res = Array()
            res.dtype = bool
            res.device = self.device
            res.ndim = self.ndim
            res.shape = copy(self.shape)
            res.size = self.size
            res.values = [k == other for k in self.values]
            return res

    def __float__(self):
        """
            Converts self to float, if self has no dimensionality.

            This method is based on Python Array API v2023.12

            https://data-apis.org/array-api/latest/API_specification/generated/array_api.array.__float__.html
        """
        if self.shape == (0,):
            return float(self.values)

    def __floordiv__(self, other):
        """
            Floor divide 2 arrays, or floor divide by a numerical object.
            Broadcasting is applied when necessary if given an array.

            Returns:
                Element-wise floor division of self and other (array or numerical)
                as an array

            Raises:
                DimensionError: If broadcasting is not viable.

            This method is based on Python Array API v2023.12

            https://data-apis.org/array-api/latest/API_specification/generated/array_api.array.__floordiv__.html

        """
        if isinstance(other, Array):
            c_other = other.copy()
            c_self = self
            if other.shape != self.shape:
                common_shape = Array.broadcast(self, c_other)
                if c_other.shape != common_shape:
                    c_other.broadcast_to(common_shape)
                if self.shape != common_shape:
                    c_self = c_self.copy()
                    c_self.broadcast_to(common_shape)

            c_other.values = [c_other.values[i] // c_self.values[i] for i in range(c_self.size)]

            return c_other

        else:
            res = Array()
            res.device = self.device
            res.ndim = self.ndim
            res.size = self.size
            res.shape = copy(self.shape)
            res.values = [k // other for k in self.values]
            return res

    def __ge__(self, other):
        """
            Calculate element-wise >= comparisons of self and other.
            Broadcasting is applied when necessary if an array is given.

            Returns:
                Element-wise >= values of self and other (array or numerical)
                as an array

            Raises:
                DimensionError: If broadcasting is not viable.

            This method is based on Python Array API v2023.12

            https://data-apis.org/array-api/latest/API_specification/generated/array_api.array.__ge__.html
        """
        if isinstance(other, Array):
            c_other = other.copy()
            c_self = self
            if other.shape != self.shape:
                common_shape = Array.broadcast(self, c_other)
                if c_other.shape != common_shape:
                    c_other.broadcast_to(common_shape)
                if self.shape != common_shape:
                    c_self = c_self.copy()
                    c_self.broadcast_to(common_shape)

            c_other.values = [c_self.values[i] >= c_other.values[i] for i in range(c_self.size)]

            c_other.dtype = bool

            return c_other
        else:
            res = Array()
            res.dtype = bool
            res.device = self.device
            res.ndim = self.ndim
            res.shape = copy(self.shape)
            res.size = self.size
            res.values = [k >= other for k in self.values]
            return res

    def __getitem__(self, item: Union[int, tuple]):

        """
            Indexes the data in self, returns a new array object
            with a view (not a copy) of the indexed data.

            This method, for arguments; supports integers, slices and tuples but
            does not support ellipsis, None and Array yet.

            This method is based on Python Array API v2023.12

            https://data-apis.org/array-api/latest/API_specification/generated/array_api.array.__getitem__.html
        """

        if isinstance(item, int):
            N = 1
            for k in self.shape[1:]:
                N *= k
            res = Array()
            res.dtype = self.dtype
            res.device = self.device
            res.values = self.values[item * N:(item + 1) * N]
            res.shape = self.shape[1:]
            res.ndim = self.ndim - 1
            res.size = N
            return res
        elif isinstance(item, slice):

            N = self.size // self.shape[0]

            start, stop, step = item.indices(self.shape[0])
            length = (stop - start) // step

            res = Array()
            res.dtype = self.dtype
            res.device = self.device
            res.values = self.values[start * N:stop * N:step]
            res.shape = tuple([length, *self.shape[1:]])
            res.ndim = self.ndim
            res.size = N * length
            return res
        elif isinstance(item, tuple):
            Ns = [self.size // self.shape[0]]
            for k in self.shape[1:]:
                Ns.append(Ns[-1] // k)

            data = self.values
            new_shape = []
            new_size: int = 1

            start: int
            stop: int
            step: int
            length: int

            sliced_previously = False

            for i, it in enumerate(item):
                if isinstance(it, int):
                    if sliced_previously:
                        temp_data = []
                        for k in range(new_size):  # self.shape[i - 1]
                            temp_data.extend(data[it * Ns[i] + k * Ns[i-1]:it * Ns[i] + k * Ns[i-1] + Ns[i]])
                        data = temp_data
                    else:
                        data = data[it * Ns[i]:(it + 1) * Ns[i]]
                elif isinstance(it, slice):
                    start, stop, step = it.indices(self.shape[i])
                    length = (stop - start) // step
                    dN = length * Ns[i]

                    if sliced_previously:
                        temp_data = []
                        for k in range(self.shape[i - 1]):
                            temp_data.extend(data[start * Ns[i] + k * dN:stop * Ns[i] + k * dN:step])
                        data = temp_data
                    else:
                        data = data[start * Ns[i]:stop * Ns[i]:step]

                    new_shape.append(length)
                    new_size *= length
                    sliced_previously = True

            diff = self.ndim - len(item)
            if diff > 0:
                temp_new_shape = []
                for k in range(diff):
                    new_size *= self.shape[-(k+1)]
                    temp_new_shape.append(self.shape[-(k+1)])

                temp_new_shape.reverse()
                new_shape.extend(temp_new_shape)

            res = Array()
            res.dtype = self.dtype
            res.device = self.device
            res.values = data
            res.shape = tuple(new_shape)
            res.ndim = len(res.shape)
            res.size = new_size
            return res

    def __gt__(self, other):
        """
            Calculate element-wise > comparisons of self and other.
            Broadcasting is applied when necessary if an array is given.

            Returns:
                Element-wise > values of self and other (array or numerical)
                as an array

            Raises:
                DimensionError: If broadcasting is not viable.

            This method is based on Python Array API v2023.12

            https://data-apis.org/array-api/latest/API_specification/generated/array_api.array.__gt__.html
        """

        if isinstance(other, Array):
            c_other = other.copy()
            c_self = self
            if other.shape != self.shape:
                common_shape = Array.broadcast(self, c_other)
                if c_other.shape != common_shape:
                    c_other.broadcast_to(common_shape)
                if self.shape != common_shape:
                    c_self = c_self.copy()
                    c_self.broadcast_to(common_shape)

            c_other.values = [c_self.values[i] > c_other.values[i] for i in range(c_self.size)]

            c_other.dtype = bool

            return c_other
        else:
            res = Array()
            res.dtype = bool
            res.device = self.device
            res.ndim = self.ndim
            res.shape = copy(self.shape)
            res.size = self.size
            res.values = [k > other for k in self.values]
            return res

    def __index__(self):
        if self.dtype != int:
            raise TypeError("Must be int type")
        if self.shape == (0,):
            return int(self.values)

    def __int__(self):
        """
            Converts self to int, if self has no dimensionality.

            This method is based on Python Array API v2023.12

            https://data-apis.org/array-api/latest/API_specification/generated/array_api.array.__int__.html
        """
        if self.shape == (0,):
            return int(self.values)

    def __invert__(self):
        res = Array()
        res.dtype = self.dtype
        res.device = self.device
        res.ndim = self.ndim
        res.shape = copy(self.shape)
        res.size = self.size
        res.values = [~k for k in self.values]
        return res

    def __le__(self, other):
        """
            Calculate element-wise <= comparisons of self and other.
            Broadcasting is applied when necessary if an array is given.

            Returns:
                Element-wise <= values of self and other (array or numerical)
                as an array

            Raises:
                DimensionError: If broadcasting is not viable.

            This method is based on Python Array API v2023.12

            https://data-apis.org/array-api/latest/API_specification/generated/array_api.array.__le__.html
        """
        if isinstance(other, Array):
            c_other = other.copy()
            c_self = self
            if other.shape != self.shape:
                common_shape = Array.broadcast(self, c_other)
                if c_other.shape != common_shape:
                    c_other.broadcast_to(common_shape)
                if self.shape != common_shape:
                    c_self = c_self.copy()
                    c_self.broadcast_to(common_shape)

            c_other.values = [c_self.values[i] <= c_other.values[i] for i in range(c_self.size)]

            c_other.dtype = bool

            return c_other
        else:
            res = Array()
            res.dtype = bool
            res.device = self.device
            res.ndim = self.ndim
            res.shape = copy(self.shape)
            res.size = self.size
            res.values = [k <= other for k in self.values]
            return res

    def __lshift__(self, other):
        res = Array()
        res.dtype = self.dtype
        res.device = self.device
        if isinstance(other, Array):
            c_other = other.copy()
            c_self = self
            if other.shape != self.shape:
                common_shape = Array.broadcast(self, c_other)
                if c_other.shape != common_shape:
                    c_other.broadcast_to(common_shape)
                if self.shape != common_shape:
                    c_self = c_self.copy()
                    c_self.broadcast_to(common_shape)

            res.ndim = c_self.ndim
            res.shape = copy(c_self.shape)
            res.size = c_self.size
            res.values = [c_self.values[i] << c_other.values[i] for i in range(c_self.size)]
        else:
            res.ndim = self.ndim
            res.shape = copy(self.shape)
            res.size = self.size
            res.values = [k << other for k in self.values]
        return res

    def __lt__(self, other):
        """
            Calculate element-wise < comparisons of self and other.
            Broadcasting is applied when necessary if an array is given.

            Returns:
                Element-wise < values of self and other (array or numerical)
                as an array

            Raises:
                DimensionError: If broadcasting is not viable.

            This method is based on Python Array API v2023.12

            https://data-apis.org/array-api/latest/API_specification/generated/array_api.array.__lt__.html
        """
        if isinstance(other, Array):
            c_other = other.copy()
            c_self = self
            if other.shape != self.shape:
                common_shape = Array.broadcast(self, c_other)
                if c_other.shape != common_shape:
                    c_other.broadcast_to(common_shape)
                if self.shape != common_shape:
                    c_self = c_self.copy()
                    c_self.broadcast_to(common_shape)

            c_other.values = [c_self.values[i] < c_other.values[i] for i in range(c_self.size)]

            c_other.dtype = bool

            return c_other
        else:
            res = Array()
            res.dtype = bool
            res.device = self.device
            res.ndim = self.ndim
            res.shape = copy(self.shape)
            res.size = self.size
            res.values = [k < other for k in self.values]
            return res

    def __matmul__(self, other):
        pass

    def __mod__(self, other):
        if isinstance(other, Array):
            c_other = other.copy()
            c_self = self
            if other.shape != self.shape:
                common_shape = Array.broadcast(self, c_other)
                if c_other.shape != common_shape:
                    c_other.broadcast_to(common_shape)
                if self.shape != common_shape:
                    c_self = c_self.copy()
                    c_self.broadcast_to(common_shape)

            c_other.values = [c_self.values[i] % c_other.values[i] for i in range(c_self.size)]

            c_other.dtype = int

            return c_other
        else:
            res = Array()
            res.dtype = int
            res.device = self.device
            res.ndim = self.ndim
            res.shape = copy(self.shape)
            res.size = self.size
            res.values = [k % other for k in self.values]
            return res

    def __mul__(self, other):
        """
            Calculate element-wise multiplication of self and other.
            Broadcasting is applied when necessary if an array is given.

            Returns:
                Element-wise multiplication of self and other (array or numerical)
                as an array

            Raises:
                DimensionError: If broadcasting is not viable.

            This method is based on Python Array API v2023.12

            https://data-apis.org/array-api/latest/API_specification/generated/array_api.array.__mul__.html
        """
        if isinstance(other, Array):
            c_other = other.copy()
            c_self = self
            if other.shape != self.shape:
                common_shape = Array.broadcast(self, c_other)
                if c_other.shape != common_shape:
                    c_other.broadcast_to(common_shape)
                if self.shape != common_shape:
                    c_self = c_self.copy()
                    c_self.broadcast_to(common_shape)

            c_other.values = [c_self.values[i] * c_other.values[i] for i in range(c_self.size)]

            return c_other
        else:
            res = Array()
            res.dtype = self.dtype
            res.device = self.device
            res.ndim = self.ndim
            res.shape = copy(self.shape)
            res.size = self.size
            res.values = [k * other for k in self.values]
            return res

    def __ne__(self, other):
        """
            Calculate element-wise != comparisons of self and other.
            Broadcasting is applied when necessary if an array is given.

            Returns:
                Element-wise != values of self and other (array or numerical)
                as an array

            Raises:
                DimensionError: If broadcasting is not viable.

            This method is based on Python Array API v2023.12

            https://data-apis.org/array-api/latest/API_specification/generated/array_api.array.__ne__.html
        """
        if isinstance(other, Array):
            c_other = other.copy()
            c_self = self
            if other.shape != self.shape:
                common_shape = Array.broadcast(self, c_other)
                if c_other.shape != common_shape:
                    c_other.broadcast_to(common_shape)
                if self.shape != common_shape:
                    c_self = c_self.copy()
                    c_self.broadcast_to(common_shape)

            c_other.values = [c_self.values[i] != c_other.values[i] for i in range(c_self.size)]

            c_other.dtype = bool

            return c_other
        else:
            res = Array()
            res.dtype = bool
            res.device = self.device
            res.ndim = self.ndim
            res.shape = copy(self.shape)
            res.size = self.size
            res.values = [k != other for k in self.values]
            return res

    def __neg__(self):
        """
            Negates each element of self, returns
            the resulting array.

            This method is based on Python Array API v2023.12

            https://data-apis.org/array-api/latest/API_specification/generated/array_api.array.__neg__.html
        """
        res = Array()
        res.dtype = self.dtype
        res.device = self.device
        res.ndim = self.ndim
        res.shape = copy(self.shape)
        res.size = self.size
        res.values = [-k for k in self.values]
        return res

    def __or__(self, other):
        """
            Calculate element-wise or operations of self and other.
            Broadcasting is applied when necessary if an array is given.

            Returns:
                Element-wise or values of self and other (array or numerical)
                as an array

            Raises:
                DimensionError: If broadcasting is not viable.

            This method is based on Python Array API v2023.12

            https://data-apis.org/array-api/latest/API_specification/generated/array_api.array.__or__.html
        """
        if isinstance(other, Array):
            c_other = other.copy()
            c_self = self
            if other.shape != self.shape:
                common_shape = Array.broadcast(self, c_other)
                if c_other.shape != common_shape:
                    c_other.broadcast_to(common_shape)
                if self.shape != common_shape:
                    c_self = c_self.copy()
                    c_self.broadcast_to(common_shape)

            c_other.values = [c_self.values[i] | c_other.values[i] for i in range(c_self.size)]

            c_other.dtype = bool

            return c_other
        else:
            res = Array()
            res.dtype = bool
            res.device = self.device
            res.ndim = self.ndim
            res.shape = copy(self.shape)
            res.size = self.size
            res.values = [k | other for k in self.values]
            return res

    def __pos__(self):
        """
            Basically, returns a copy of self.

            This method is based on Python Array API v2023.12

            https://data-apis.org/array-api/latest/API_specification/generated/array_api.array.__pos__.html
        """
        res = Array()
        res.dtype = self.dtype
        res.device = self.device
        res.ndim = self.ndim
        res.shape = copy(self.shape)
        res.size = self.size
        res.values = copy(self.values)
        return res

    def __pow__(self, power):
        res = Array()
        res.device = None

        if isinstance(power, Union[int, float, Decimal, Variable]):
            res.size = self.size
            res.ndim = self.ndim
            res.shape = self.shape
            res.values = [self.values[i] ** power for i in range(res.size)]

        else:  # power is array

            if self.shape == power.shape:
                res.size = self.size
                res.ndim = self.ndim
                res.shape = self.shape
                res.values = [self.values[i] ** power.values[i] for i in range(res.size)]

            else:
                common_shape = Array.broadcast(self, power)
                c_x1 = self
                c_x2 = power
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

    def __radd__(self, other):
        if isinstance(other, Array):
            c_other = other.copy()
            c_self = self
            if other.shape != self.shape:
                common_shape = Array.broadcast(self, c_other)
                if c_other.shape != common_shape:
                    c_other.broadcast_to(common_shape)
                if self.shape != common_shape:
                    c_self = c_self.copy()
                    c_self.broadcast_to(common_shape)

            c_other.values = [c_self.values[i] + c_other.values[i] for i in range(c_self.size)]

            return c_other

        else:
            res = Array()
            res.device = self.device
            res.ndim = self.ndim
            res.size = self.size
            res.shape = copy(self.shape)
            res.values = [k + other for k in self.values]
            return res

    def __rshift__(self, other):
        res = Array()
        res.dtype = self.dtype
        res.device = self.device
        if isinstance(other, Array):
            c_other = other.copy()
            c_self = self
            if other.shape != self.shape:
                common_shape = Array.broadcast(self, c_other)
                if c_other.shape != common_shape:
                    c_other.broadcast_to(common_shape)
                if self.shape != common_shape:
                    c_self = c_self.copy()
                    c_self.broadcast_to(common_shape)

            res.ndim = c_self.ndim
            res.shape = copy(c_self.shape)
            res.size = c_self.size
            res.values = [c_self.values[i] >> c_other.values[i] for i in range(c_self.size)]
        else:
            res.ndim = self.ndim
            res.shape = copy(self.shape)
            res.size = self.size
            res.values = [k >> other for k in self.values]
        return res

    def __rsub__(self, other):
        if isinstance(other, Array):
            c_other = other.copy()
            c_self = self
            if other.shape != self.shape:
                common_shape = Array.broadcast(self, c_other)
                if c_other.shape != common_shape:
                    c_other.broadcast_to(common_shape)
                if self.shape != common_shape:
                    c_self = c_self.copy()
                    c_self.broadcast_to(common_shape)

            c_other.values = [c_other.values[i] - c_self.values[i] for i in range(c_self.size)]

            return c_other

        else:
            res = Array()
            res.device = self.device
            res.ndim = self.ndim
            res.size = self.size
            res.shape = copy(self.shape)
            res.values = [other - k for k in self.values]
            return res

    def __setitem__(self, key, value):
        if isinstance(key, int):
            N = 1
            for k in self.shape[1:]:
                N *= k
            self.values[key * N:(key + 1) * N] = value

        elif isinstance(key, slice):

            N = self.size // self.shape[0]
            start, stop, step = key.indices(self.shape[0])

            if isinstance(value, list):
                self.values[start * N:stop * N:step] = value
            else:
                for i in range(start, stop, step):
                    self.values[i * N:(i + 1) * N] = [value] * N

        elif isinstance(key, tuple):
            Ns = [self.size // self.shape[0]]
            for k in self.shape[1:]:
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
                            temp_indices.extend(indices[it * Ns[i] + k * Ns[i-1]:it * Ns[i] + k * Ns[i-1] + Ns[i]])
                        indices = temp_indices
                    elif indices:
                        indices = indices[it * Ns[i]:(it + 1) * Ns[i]]
                    else:
                        indices = list(range(it * Ns[i], (it + 1) * Ns[i]))

                elif isinstance(it, slice):
                    start, stop, step = it.indices(self.shape[i])

                    if sliced_previously:
                        dN = Ns[i] * (stop - start) // step

                        temp_indices = []
                        for k in range(self.shape[i - 1]):
                            temp_indices.extend(indices[start * Ns[i] + k * dN:stop * Ns[i] + k * dN:step])
                        indices = temp_indices
                    elif indices:
                        indices = indices[start * Ns[i]:stop * Ns[i]:step]
                    else:
                        indices = list(range(start * Ns[i], stop * Ns[i], step))

                    sliced_previously = True

            if isinstance(value, list):
                for idx, val in zip(indices, value):
                    self.values[idx] = val
            else:
                for idx in indices:
                    self.values[idx] = value
        elif isinstance(key, list):
            # TODO: If array indexing is done with a list in __setitem__, all elements must be direct indexes to internal list.
            #       Therefore, this option is only valid for outputs from api.utility.index_query_()
            for i, idx in enumerate(key):
                self.values[i] = value[idx]  # This will automatically raise an error if illegally indexed

    def __sub__(self, other):
        """
            Subtracts 2 arrays, or subtracts a numerical object from self.
            Broadcasting is applied when necessary if given an array.

            Returns:
                Element-wise subtraction of self and other (array or numerical)
                as an array

            Raises:
                DimensionError: If broadcasting is not viable.

            This method is based on Python Array API v2023.12

            https://data-apis.org/array-api/latest/API_specification/generated/array_api.array.__sub__.html

        """
        if isinstance(other, Array):
            c_other = other.copy()
            c_self = self
            if other.shape != self.shape:
                common_shape = Array.broadcast(self, c_other)
                if c_other.shape != common_shape:
                    c_other.broadcast_to(common_shape)
                if self.shape != common_shape:
                    c_self = c_self.copy()
                    c_self.broadcast_to(common_shape)

            c_other.values = [c_self.values[i] - c_other.values[i] for i in range(c_self.size)]

            return c_other

        else:
            res = Array()
            res.device = self.device
            res.ndim = self.ndim
            res.size = self.size
            res.shape = copy(self.shape)
            res.values = [k - other for k in self.values]
            return res

    def __truediv__(self, other):
        """
            Divide 2 arrays, or divide by a numerical object.
            Broadcasting is applied when necessary if given an array.

            Returns:
                Element-wise division of self and other (array or numerical)
                as an array

            Raises:
                DimensionError: If broadcasting is not viable.

            This method is based on Python Array API v2023.12

            https://data-apis.org/array-api/latest/API_specification/generated/array_api.array.__truediv__.html

        """
        if isinstance(other, Array):
            c_other = other.copy()
            c_self = self
            if other.shape != self.shape:
                common_shape = Array.broadcast(self, c_other)
                if c_other.shape != common_shape:
                    c_other.broadcast_to(common_shape)
                if self.shape != common_shape:
                    c_self = c_self.copy()
                    c_self.broadcast_to(common_shape)
            c_other.values = [c_self.values[i] / c_other.values[i] for i in range(c_self.size)]

            return c_other

        else:
            res = Array()
            res.device = self.device
            res.ndim = self.ndim
            res.size = self.size
            res.shape = copy(self.shape)
            res.values = [k / other for k in self.values]
            return res

    def __xor__(self, other):
        """
            Calculate element-wise exclusive or operations of self and other.
            Broadcasting is applied when necessary if an array is given.

            Returns:
                Element-wise exclusive or or values of self and other (array or numerical)
                as an array

            Raises:
                DimensionError: If broadcasting is not viable.

            This method is based on Python Array API v2023.12

            https://data-apis.org/array-api/latest/API_specification/generated/array_api.array.__xor__.html
        """
        if isinstance(other, Array):
            c_other = other.copy()
            c_self = self
            if other.shape != self.shape:
                common_shape = Array.broadcast(self, c_other)
                if c_other.shape != common_shape:
                    c_other.broadcast_to(common_shape)
                if self.shape != common_shape:
                    c_self = c_self.copy()
                    c_self.broadcast_to(common_shape)

            c_other.values = [c_self.values[i] ^ c_other.values[i] for i in range(c_self.size)]

            c_other.dtype = bool

            return c_other
        else:
            res = Array()
            res.dtype = bool
            res.device = self.device
            res.ndim = self.ndim
            res.shape = copy(self.shape)
            res.size = self.size
            res.values = [k ^ other for k in self.values]
            return res

    def to_device(self, device):
        """
            This method is not implemented and only exists to
            conform to Array API.

        """
        raise NotImplementedError()

    def reshape(self, shape: BASIC_ITERABLE):
        """
            Reshape self to given new shape. Size of new
            shape must equal to the size of current shape.
            -1 can be used in new shape, but at most one
            instance is allowed.

            This method does not return anything, but modifies
            self.

            Raises:
                DimensionError: If sizes of shapes do not match

        """
        if -1 not in shape:
            n = 1
            for k in shape:
                n *= k
            if n == self.size:
                self.shape = shape
        else:
            n = 1
            for k in shape:
                if k != -1:
                    n *= k
            N = self.size // n
            if N * n == self.size:
                temp = list(shape)
                temp[temp.index(-1)] = N
                self.shape = tuple(temp)
            else:
                raise DimensionError(0)


    def __broadcast(self, shape: BASIC_ITERABLE) -> list:
        """
            Internal function used to broadcast self into
            given shape.

            Possibly modifies ndim and shape attributes of self.

            Returns:
                The data as a list, broadcasted to desired
                shape.
        """

        if self.ndim == 1 and len(shape) == 1:
            if self.shape[0] == shape[0]:
                return self.values
            else:
                new_data = []
                for k in range(shape[0] // self.shape[0]):
                    new_data += self.values
                return new_data

        if len(shape) == self.ndim:
            data = []
            for k in range(shape[0] // self.shape[0]):
                for l in range(self.shape[0]):
                    data = data + self[l].__broadcast(shape[1:])
            return data
        else:
            # this means that shape is bigger than ndim
            temp = list(self.shape)
            while len(temp) < len(shape):
                temp.insert(0, 1)
            self.ndim = len(shape)
            self.shape = tuple(temp)
            return self.__broadcast(shape)

    def broadcast_to(self, shape: BASIC_ITERABLE):
        """
            Applies broadcasting to self to conform to given shape.
            Modifies attributes of self, including the data.

            Returns:
                None, modifies the data in self.
        """
        self.values = self.__broadcast(shape)
        self.shape = copy(tuple(shape))
        self.size = len(self.values)

    def copy(self):
        """
            Copy self into a new array, return the new array.
        """
        res = Array()
        res.dtype = self.dtype
        res.device = self.device
        res.ndim = self.ndim
        res.shape = copy(self.shape)
        res.size = self.size
        res.values = self.values.copy()
        return res

    def astype(self, dtype: Union[Type, None] = None):
        """
            Cast self to given type. Return the casted array.
            
            Args:
                dtype (Type): Dtype to cast to. Must be a Python
                    or Vectorgebra type (includes aliased ctypes).
                    If left as none, simply copies self and returns
                    the new array.
            
            Returns:
                Array: The casted array object
        """
        if dtype is None:
            return self.copy()
        else:
            c_self = self.copy()
            c_self.dtype = dtype
            c_self.values = [dtype(k) for k in c_self.values]
            return c_self

    def all(self) -> bool:
        """
            Evaluate the truth value of all elements in self, combined.

            Returns:
                bool: True if all values in self are true, otherwise False
        """
        control = True
        for k in self.values:
            control &= k
        return control

    def any(self) -> bool:
        """
            Evaluate the truth value of any element in self, combined.

            Returns:
                bool: True if any value in self is true, otherwise False
        """
        control = False
        for k in self.values:
            control |= k
        return control

    def item(self):
        """
            Return the singular item stored in the array.

            Raises:
                DimensionError: If the size of self is not 1.
        """
        if self.size == 1:
            return self.values[0]
        raise DimensionError(0)


