from ..math import *
from ..utils import *
from . import minimum, maximum, Matrix
from ..variable import Variable
import random
from typing import Union, Callable
from decimal import Decimal

class Vector:
    """
        Vector class representing mathematical vectors and supporting various operations.

        Attributes:
            values (list): List containing the elements of the vector.
            shape[0] (int): shape[0]ality of the vector.
    """
    def __init__(self, *args):
        """
                Initializes a Vector object with the given arguments.

                Args:
                    *args: Variable number of arguments representing the elements of the vector.

                Raises:
                    ArgTypeError: If any argument is not numeric, boolean, Vector, Matrix or callable.
        """
        for k in args:
            if not isinstance(k, Union[int, float, Decimal, Infinity, Undefined,
                              Complex, Callable, Vector, Matrix, Variable, list, tuple]):
                raise ArgTypeError("Arguments must be numeric or boolean.")

        if isinstance(args[0], Union[list, tuple]):
            self.values = args[0].copy()
            self.shape = (len(args[0]),)
        else:
            self.values = [_ for _ in args]
            self.shape = (len(args),)

    def __str__(self):
        """Returns a string representation of the vector."""
        return str(self.values)

    def __repr__(self):
        """Returns a string representation of the vector."""
        return str(self.values)

    def __getitem__(self, index):
        """
        Returns the element at the specified index in the vector.

        Args:
            index (int): The index of the element to retrieve.

        Returns:
            Any: The element at the specified index in the vector.

        Raises:
            IndexError: If the index is out of range.
        """
        return self.values[index]

    def __setitem__(self, key, value):
        """
        Sets the element at the specified index in the vector to the given value.

        Args:
            key (int): The index of the element to set.
            value (Any): The value to set at the specified index.

        Raises:
            IndexError: If the index is out of range.
        """
        self.values[key] = value

    def __call__(self, *args):
        """
        Returns the value at tuple "args" of the vector-function.

        Returns:
            number: Return value of each function in the vector.
        """
        return Vector(*[k(*args) for k in self.values])

    def __iter__(self):
        return iter(self.values)

    @property
    def T(self):
        """
            Returns the transpose of self, as a matrix object.

            Returns:
                Matrix: Transpose of self, which would be analogous
                    to a column vector.

        """
        return Matrix([[self.values[k]] for k in range(self.shape[0])])

    def __len__(self):
        """
        Returns the number of elements in the vector.

        Returns:
            int: The number of elements in the vector.
        """
        return len(self.values)

    def __add__(self, arg):
        if isinstance(arg, Union[int, float, Decimal, Infinity, Undefined, Complex, Variable]) and not isinstance(self.values[0], Callable):
            return Vector(*[self.values[k] + arg for k in range(self.shape[0])])

        if isinstance(arg, Union[int, float, Decimal, Infinity, Undefined, Complex]) and isinstance(self.values[0], Callable):
            return Vector(*[lambda *args, k=k: self.values[k](*args) + arg for k in range(self.shape[0])])

        if isinstance(arg, Callable) and not isinstance(arg, Vector) and not isinstance(self.values[0], Callable):
            return Vector(
                *[lambda *args, k=k: self.values[k] + arg(*args) for k in range(len(self.values))])

        if isinstance(arg, Callable) and not isinstance(arg, Vector) and isinstance(self.values[0], Callable):
            return Vector(
                *[lambda *args, k=k: self.values[k](*args) + arg(*args) for k in range(len(self.values))])

        if not isinstance(arg, Vector):
            raise ArgTypeError("Must be a numerical value or a callable.")

        if self.shape[0] != arg.shape[0]:
            raise DimensionError(0)

        if isinstance(self.values[0], Callable) and isinstance(arg.values[0], Callable):
            return Vector(*[lambda *args, k=k: self.values[k](*args) + arg.values[k](*args) for k in range(len(self.values))])

        return Vector(*[self.values[k] + arg.values[k] for k in range(self.shape[0])])

    def __radd__(self, arg):
        if isinstance(arg, Union[int, float, Decimal, Infinity, Undefined, Complex, Variable]) and not isinstance(self.values[0], Callable):
            return Vector(*[self.values[k] + arg for k in range(self.shape[0])])

        if isinstance(arg, Union[int, float, Decimal, Infinity, Undefined, Complex]) and isinstance(self.values[0], Callable):
            return Vector(*[lambda *args, k=k: self.values[k](*args) + arg for k in range(self.shape[0])])

        if isinstance(arg, Callable) and isinstance(self.values[0], Callable):
            return Vector(*[lambda *args, k=k: self.values[k](*args) + arg(*args) for k in range(self.shape[0])])

        if isinstance(arg, Callable):
            return Vector(*[lambda *args, k=k: self.values[k] + arg(*args) for k in range(self.shape[0])])

        raise ArgTypeError("Must be a numerical value or a callable.")

    def __sub__(self, arg):
        if isinstance(arg, Union[int, float, Decimal, Infinity, Undefined, Complex, Variable]) and not isinstance(self.values[0], Callable):
            return Vector(*[self.values[k] - arg for k in range(0, self.shape[0])])

        if isinstance(arg, Union[int, float, Decimal, Infinity, Undefined, Complex]) and isinstance(self.values[0], Callable):
            return Vector(*[lambda *args, k=k: self.values[k](*args) - arg  for k in range(self.shape[0])])

        if isinstance(arg, Callable) and not isinstance(arg, Vector) and not isinstance(self.values[0], Callable):
            return Vector(
                *[lambda *args, k=k: self.values[k] - arg(*args) for k in range(len(self.values))])

        if isinstance(arg, Callable) and not isinstance(arg, Vector) and isinstance(self.values[0], Callable):
            return Vector(
                *[lambda *args, k=k: self.values[k](*args) - arg(*args) for k in range(len(self.values))])

        if not isinstance(arg, Vector):
            raise ArgTypeError("Must be a numerical value or a callable.")

        if self.shape[0] != arg.shape[0]:
            raise DimensionError(0)

        if isinstance(self.values[0], Callable) and isinstance(arg.values[0], Callable):
            return Vector(*[lambda *args, k=k: self.values[k](*args) - arg.values[k](*args) for k in range(len(self.values))])

        return Vector(*[self.values[k] - arg.values[k] for k in range(self.shape[0])])

    def __rsub__(self, arg):
        if isinstance(arg, Union[int, float, Decimal, Infinity, Undefined, Complex, Variable]) and not isinstance(self.values[0], Callable):
            return Vector(*[arg - self.values[k] for k in range(0, self.shape[0])])

        if isinstance(arg, Union[int, float, Decimal, Infinity, Undefined, Complex]) and isinstance(self.values[0], Callable):
            return Vector(*[lambda *args, k=k: arg - self.values[k](*args) for k in range(self.shape[0])])

        if isinstance(arg, Callable) and isinstance(self.values[0], Callable):
            return Vector(*[lambda *args, k=k: arg(*args) - self.values[k](*args) for k in range(self.shape[0])])

        if isinstance(arg, Callable):
            return Vector(*[lambda *args, k=k: arg(*args) - self.values[k] for k in range(self.shape[0])])

        raise ArgTypeError("Must be a numerical value or a callable.")

    def dot(self, arg):
        """
            Computes the dot product of the vector with another vector.

            Args:
                arg (Vector): The vector with which the dot product is computed.

            Returns:
                The dot product of the two vectors.

            Raises:
                ArgTypeError: If the argument `arg` is not a vector.
                DimensionError: If the shape[0]s of the two vectors are not the same.
        """
        if not isinstance(arg, Vector):
            raise ArgTypeError("Must be a vector.")
        if self.shape[0] != arg.shape[0]:
            raise DimensionError(0)
        mul = [self.values[k] * arg.values[k] for k in range(self.shape[0])]
        sum = 0
        for k in mul:
            sum += k
        return sum

    def __mul__(self, arg):
        if not isinstance(arg, Union[int, float, Decimal, Infinity, Undefined, Complex, Callable, Variable]):
            raise ArgTypeError("Must be a numerical value.")

        if isinstance(self.values[0], Callable) and not isinstance(arg, Callable):
            return Vector(*[lambda *args, k=k: self.values[k](*args) * arg for k in range(self.shape[0])])

        if isinstance(self.values[0], Callable) and isinstance(arg, Callable):
            return Vector(*[lambda *args, k=k: self.values[k](*args) * arg(*args) for k in range(self.shape[0])])

        return Vector(*[self.values[k] * arg for k in range(self.shape[0])])

    def __rmul__(self, arg):
        if not isinstance(arg, Union[int, float, Decimal, Infinity, Undefined, Complex, Callable, Variable]):
            raise ArgTypeError("Must be a numerical value.")

        if isinstance(self.values[0], Callable) and not isinstance(arg, Callable):
            return Vector(*[lambda *args, k=k: self.values[k](*args) * arg for k in range(self.shape[0])])

        if isinstance(self.values[0], Callable) and isinstance(arg, Callable):
            return Vector(*[lambda *args, k=k: self.values[k](*args) * arg(*args) for k in range(self.shape[0])])

        return Vector(*[self.values[k] * arg for k in range(self.shape[0])])

    def __truediv__(self, arg):
        if not isinstance(arg, Union[int, float, Decimal, Infinity, Undefined, Complex, Variable]):
            raise ArgTypeError("Must be a numerical value.")

        if isinstance(self.values[0], Callable) and not isinstance(arg, Callable):
            return Vector(*[lambda *args, k=k: self.values[k](*args) / arg for k in range(self.shape[0])])

        if isinstance(self.values[0], Callable) and isinstance(arg, Callable):
            return Vector(*[lambda *args, k=k: self.values[k](*args) / arg(*args) for k in range(self.shape[0])])

        return Vector(*[self.values[k] / arg for k in range(self.shape[0])])

    def __floordiv__(self, arg):
        if not isinstance(arg, Union[int, float, Decimal, Infinity, Undefined, Complex, Variable]):
            raise ArgTypeError("Must be a numerical value.")

        if isinstance(self.values[0], Callable) and not isinstance(arg, Callable):
            return Vector(*[lambda *args, k=k: self.values[k](*args) // arg for k in range(self.shape[0])])

        if isinstance(self.values[0], Callable) and isinstance(arg, Callable):
            return Vector(*[lambda *args, k=k: self.values[k](*args) // arg(*args) for k in range(self.shape[0])])

        return Vector(*[self.values[k] // arg for k in range(self.shape[0])])

    def __iadd__(self, arg):
        if isinstance(arg, Union[int, float, Decimal, Infinity, Undefined, Complex, Variable]) and not isinstance(self.values[0], Callable):
            return Vector(*[self.values[k] + arg for k in range(self.shape[0])])

        if isinstance(arg, Union[int, float, Decimal, Infinity, Undefined, Complex]) and isinstance(self.values[0], Callable):
            return Vector(*[lambda *args, k=k: self.values[k](*args) + arg for k in range(self.shape[0])])

        if isinstance(arg, Callable) and not isinstance(arg, Vector) and not isinstance(self.values[0], Callable):
            return Vector(
                *[lambda *args, k=k: self.values[k] + arg(*args) for k in range(len(self.values))])

        if isinstance(arg, Callable) and not isinstance(arg, Vector) and isinstance(self.values[0], Callable):
            return Vector(
                *[lambda *args, k=k: self.values[k](*args) + arg(*args) for k in range(len(self.values))])

        if not isinstance(arg, Vector):
            raise ArgTypeError("Must be a numerical value.")

        if self.shape[0] != arg.shape[0]:
            raise DimensionError(0)

        if isinstance(self.values[0], Callable) and isinstance(arg.values[0], Callable):
            return Vector(*[lambda *args, k=k: self.values[k](*args) + arg.values[k](*args) for k in range(len(self.values))])

        return Vector(*[self.values[k] + arg.values[k] for k in range(self.shape[0])])

    def __isub__(self, arg):
        if isinstance(arg, Union[int, float, Decimal, Infinity, Undefined, Complex, Variable]) and not isinstance(self.values[0], Callable):
            return Vector(*[self.values[k] - arg for k in range(self.shape[0])])

        if isinstance(arg, Union[int, float, Decimal, Infinity, Undefined, Complex]) and isinstance(self.values[0], Callable):
            return Vector(*[lambda *args, k=k: self.values[k](*args) - arg  for k in range(self.shape[0])])

        if isinstance(arg, Callable) and not isinstance(arg, Vector) and not isinstance(self.values[0], Callable):
            return Vector(
                *[lambda *args, k=k: self.values[k] - arg(*args) for k in range(len(self.values))])

        if isinstance(arg, Callable) and not isinstance(arg, Vector) and isinstance(self.values[0], Callable):
            return Vector(
                *[lambda *args, k=k: self.values[k](*args) - arg(*args) for k in range(len(self.values))])

        if not isinstance(arg, Vector):
            raise ArgTypeError("Must be a numerical value.")

        if self.shape[0] != arg.shape[0]:
            raise DimensionError(0)

        if isinstance(self.values[0], Callable) and isinstance(arg.values[0], Callable):
            return Vector(*[lambda *args, k=k: self.values[k](*args) - arg.values[k](*args) for k in range(len(self.values))])

        return Vector(*[self.values[k] - arg.values[k] for k in range(self.shape[0])])

    def __gt__(self, arg):
        if isinstance(arg, Union[int, float, Decimal, Infinity, Undefined, Complex, Variable]):
            sum = 0
            for k in self.values:
                sum += k**2
            return sum > arg**2
        if not isinstance(arg, Vector):
            raise ArgTypeError("Must be a numerical value.")
        if self.shape[0] != arg.shape[0]:
            raise DimensionError(0)
        sum = 0
        for k in range(self.shape[0]):
            sum += (self.values[k] - arg.values[k]) * (self.values[k] + arg.values[k])
        return sum > 0

    def __ge__(self, arg):
        if isinstance(arg, Union[int, float, Decimal, Infinity, Undefined, Complex, Variable]):
            sum = 0
            for k in self.values:
                sum += k**2
            return sum >= arg**2
        if not isinstance(arg, Vector):
            raise ArgTypeError("Must be a numerical value.")
        if self.shape[0] != arg.shape[0]:
            raise DimensionError(0)
        sum = 0
        for k in range(self.shape[0]):
            sum += (self.values[k] - arg.values[k]) * (self.values[k] + arg.values[k])
        return sum >= 0

    def __lt__(self, arg):
        if isinstance(arg, Union[int, float, Decimal, Infinity, Undefined, Complex, Variable]):
            sum = 0
            for k in self.values:
                sum += k**2
            return sum < arg**2
        if not isinstance(arg, Vector):
            raise ArgTypeError("Must be a numerical value.")
        if self.shape[0] != arg.shape[0]:
            raise DimensionError(0)
        sum = 0
        for k in range(self.shape[0]):
            sum += (self.values[k] - arg.values[k]) * (self.values[k] + arg.values[k])
        return sum < 0

    def __le__(self, arg):
        if isinstance(arg, Union[int, float, Decimal, Infinity, Undefined, Complex, Variable]):
            sum = 0
            for k in self.values:
                sum += k**2
            return sum <= arg**2
        if not isinstance(arg, Vector):
            raise ArgTypeError("Must be a numerical value.")
        if self.shape[0] != arg.shape[0]:
            raise DimensionError(0)
        sum = 0
        for k in range(self.shape[0]):
            sum += (self.values[k] - arg.values[k]) * (self.values[k] + arg.values[k])
        return sum <= 0

    def __eq__(self, arg):
        if isinstance(arg, Union[int, float, Decimal, Infinity, Undefined, Complex, Variable]):
            for k in self.values:
                if k != arg:
                    return False
            return True
        if not isinstance(arg, Vector):
            raise ArgTypeError("Must be a numerical value.")
        if self.shape[0] != arg.shape[0]:
            raise DimensionError(0)

        for k in range(self.shape[0]):
            if self.values[k] != arg.values[k]:
                return False
        return True

    def __neg__(self):
        return Vector(*[-k for k in self.values])

    def __pos__(self):
        return Vector(*[k for k in self.values])

    def __and__(self, arg):
        if not isinstance(arg, Vector):
            raise ArgTypeError("Must be a vector.")
        if self.shape[0] != arg.shape[0]:
            raise DimensionError(0)
        return Vector(*[(self.values[k] and arg.values[k]) for k in range(self.shape[0])])

    def __iand__(self, arg):
        if not isinstance(arg, Vector):
            raise ArgTypeError("Must be a vector.")
        if self.shape[0] != arg.shape[0]:
            raise DimensionError(0)
        return Vector(*[(self.values[k] and arg.values[k]) for k in range(self.shape[0])])

    def __or__(self, arg):
        if not isinstance(arg, Vector):
            raise ArgTypeError("Must be a vector.")
        if self.shape[0] != arg.shape[0]:
            raise DimensionError(0)
        return Vector(*[(self.values[k] or arg.values[k]) for k in range(self.shape[0])])

    def __ior__(self, arg):
        if not isinstance(arg, Vector):
            raise ArgTypeError("Must be a vector.")
        if self.shape[0] != arg.shape[0]:
            raise DimensionError(0)
        return Vector(*[(self.values[k] or arg.values[k]) for k in range(self.shape[0])])

    def __xor__(self, arg):
        if not isinstance(arg, Vector):
            raise ArgTypeError("Must be a vector.")
        if self.shape[0] != arg.shape[0]:
            raise DimensionError(0)
        return Vector(*[(self.values[k] ^ arg.values[k]) for k in range(self.shape[0])])

    def __ixor__(self, arg):
        if not isinstance(arg, Vector):
            raise ArgTypeError("Must be a vector.")
        if self.shape[0] != arg.shape[0]:
            raise DimensionError(0)
        return Vector(*[(self.values[k] ^ arg.values[k]) for k in range(self.shape[0])])

    def __invert__(self):
        return Vector(*[int(not self.values[k]) for k in range(self.shape[0])])

    def append(self, arg):
        """
            Appends the elements of the given argument to the end of the vector.

            Args:
                arg: The element or iterable containing elements to append to the vector.

            Raises:
                ArgTypeError: If the argument is not a numeric value, boolean, vector, list, or tuple.
        """
        if isinstance(arg, Union[int, float, Decimal, Infinity, Undefined, Complex, Callable, Variable]):
            self.values.append(arg)
            self.shape[0] += 1
            return
        if isinstance(arg, Vector):
            for k in arg.values:
                self.values.append(k)
            self.shape[0] += arg.shape[0]
        elif isinstance(arg, Union[list, tuple]):
            for k in arg:
                self.values.append(k)
            self.shape[0] += len(arg)
        raise ArgTypeError("Must be a numerical value.")

    def copy(self):
        """
            Returns a copy of the vector.

            Returns:
                Vector: A copy of the vector with the same elements and shape[0]ality.
        """
        return Vector(*self.values.copy())

    def pop(self, ord=-1):
        """
            Removes and returns the element at the specified index in the vector.

            Args:
                ord (int, optional): The index of the element to remove. Defaults to -1,
                    meaning the last element is removed.

            Returns:
                Any: The removed element from the vector.

            Raises:
                RangeError: If the specified index is out of range.
        """
        try:
            popped = self.values.pop(ord)
            self.shape[0] -= 1
            return popped
        except IndexError:
            raise RangeError()

    def length(self):
        """
            Computes the length (magnitude) of the vector.

            Returns:
                Union[int, float]: The length of the vector.

            Example:
                If `v` is a vector, `v.length()` returns the length of the vector,
                which is computed as the square root of the sum of the squares of its components.
        """
        sum = 0
        for k in self.values:
            sum += k**2
        return sqrt(sum)

    def proj(self, arg):
        """
            Computes the projection of the vector onto another vector.

            The projection of vector `self` onto vector `arg` is computed as follows:
            proj_v(arg) = (self ⋅ arg) / ||arg||² * arg

            Args:
                arg (Vector): The vector onto which the projection is computed.

            Returns:
                Vector: The projection of the vector onto the specified vector.

            Raises:
                ArgTypeError: If `arg` is not a vector.
                DimensionError: If the shape[0]s of `self` and `arg` are not equal.
        """
        if not isinstance(arg, Vector):
            raise ArgTypeError("Must be a vector.")
        if self.shape[0] != arg.shape[0]:
            raise DimensionError(0)
        if not self.shape[0]:
            return 0
        dot = self.dot(arg)
        sum = 0
        for k in arg.values:
            sum += k**2
        try:
            dot /= sum
        except ZeroDivisionError:
            # Can only be positive or 0
            dot = Infinity()
        res = Vector(*arg.values)
        return res * dot

    def unit(self):
        """
            Computes the unit vector in the same direction as the current vector.

            Returns:
                Vector: The unit vector in the same direction as the current vector.

            Example:
                If `v` is a vector, `v.unit()` returns a new vector that has the same direction
                as `v` but has a length of 1.
        """
        l = self.length()
        if l:
            return Vector(*[k / l for k in self.values])
        return Vector(*[Infinity() for k in range(self.shape[0])])

    @staticmethod
    def spanify(*args):
        """
            Computes a set of orthogonal unit vectors that span the subspace defined by the input vectors.

            The `spanify` method calculates a set of orthogonal unit vectors that span the subspace defined
            by the input vectors. It uses the Gram-Schmidt process to orthogonalize the vectors and then
            normalizes them to unit length.

            Args:
                *args (Vector): Variable number of vectors forming the basis of the subspace.

            Returns:
                list: A list containing orthogonal unit vectors that span the subspace.

            Raises:
                ArgTypeError: If any argument is not a vector.
                AmountError: If the number of vectors provided does not match their shape[0]s.
        """
        v_list = []
        for k in args:
            if not isinstance(k, Vector):
                raise ArgTypeError("Must be a vector.")
            if k.shape[0] != len(args):
                raise AmountError()
            v_list.append(k)
        for k in range(1, len(v_list)):
            temp = v_list[k]
            for l in range(k):
                temp -= v_list[k].proj(v_list[l])
            v_list[k] = temp.unit()
        v_list[0] = v_list[0].unit()
        return v_list

    @staticmethod
    def does_span(*args):
        """
            Checks whether the input vectors form a spanning set for their respective subspace.

            The `does_span` method determines whether the input vectors form a spanning set for their
            respective subspace by verifying if each vector is orthogonal to all the others. If all
            vectors are orthogonal to each other, they form a spanning set.

            Args:
                *args (Vector): Variable number of vectors to be checked for spanning.

            Returns:
                bool: True if the vectors form a spanning set, False otherwise.

            Raises:
                ArgTypeError: If any argument is not a vector.
                AmountError: If the number of vectors provided does not match their shape[0]s.
        """
        v_list = Vector.spanify(*args)
        N = len(v_list)
        for k in range(N):
            for l in range(N):
                if v_list[k].dot(v_list[l]) >= 0.0000000001 and k != l:
                    return False
        return True

    @staticmethod
    def randVint(dim: int, a: int, b: int, decimal: bool = False):
        """
            Generates a random integer vector with specified shape[0]s and range.

            The `randVint` method creates a random integer vector with the specified shape[0]s and
            values within the given range [a, b].

            Args:
                dim (int): The shape[0]ality of the random vector.
                a (int): The lower bound of the range for random integer generation.
                b (int): The upper bound of the range for random integer generation.
                decimal (bool, optional): If True, generated values will be of Decimal type. Defaults to False.

            Returns:
                Vector: A random integer vector with the specified shape[0]s and values within the range [a, b].

            Raises:
                ArgTypeError: If any input argument is not an integer.
                RangeError: If the shape[0] is not a positive integer.
        """
        if not (isinstance(dim, int) and isinstance(a, int) and isinstance(b, int)):
            raise ArgTypeError("Must be an integer.")
        if dim <= 0:
            raise RangeError()
        if decimal:
            return Vector(*[Decimal(random.randint(a, b)) for k in range(dim)])
        return Vector(*[random.randint(a, b) for k in range(dim)])

    @staticmethod
    def randVfloat(dim, a: float, b: float, decimal: bool = False):
        """
            Generates a random float vector with specified shape[0]s and range.

            The `randVfloat` method creates a random float vector with the specified shape[0]s and
            values within the given range [a, b].

            Args:
                dim (int): The shape[0]ality of the random vector.
                a (float): The lower bound of the range for random float generation.
                b (float): The upper bound of the range for random float generation.
                decimal (bool, optional): If True, generated values will be of Decimal type. Defaults to False.

            Returns:
                Vector: A random float vector with the specified shape[0]s and values within the range [a, b].

            Raises:
                ArgTypeError: If any input argument is not a numerical value.
                RangeError: If the shape[0] is not a positive integer.
        """
        if not (isinstance(dim, int) and
                (isinstance(a, Union[int, float, Decimal])) and
                (isinstance(b, Union[int, float, Decimal]))):
            raise ArgTypeError("Must be a numerical value.")
        if dim <= 0:
            raise RangeError()
        if decimal:
            return Vector(*[Decimal(random.uniform(a, b)) for k in range(dim)])
        return Vector(*[random.uniform(a, b) for k in range(dim)])

    @staticmethod
    def randVbool(dim, decimal: bool = False):
        """
            Generates a random boolean vector with specified shape[0]s.

            The `randVbool` method creates a random boolean vector with the specified shape[0]s.

            Args:
                dim (int): The shape[0]ality of the random vector.
                decimal (bool, optional): If True, generated values will be of Decimal type. Defaults to False.

            Returns:
                Vector: A random boolean vector with the specified shape[0]s.

            Raises:
                ArgTypeError: If the shape[0] is not an integer.
                RangeError: If the shape[0] is not a positive integer.
        """
        if not isinstance(dim, int):
            raise ArgTypeError("Must be an integer.")
        if dim <= 0:
            raise RangeError()

        if decimal:
            return Vector(*[Decimal(random.randrange(0, 2)) for k in range(dim)])
        return Vector(*[random.randrange(0, 2) for k in range(dim)])

    @staticmethod
    def randVgauss(dim, mu=0, sigma=0, decimal: bool = False):
        """
            Generates a random vector of Gaussian (normal) distribution with specified shape[0]s.

            The `randVgauss` method creates a random vector of Gaussian (normal) distribution with the specified shape[0]s,
            mean, and standard deviation.

            Args:
                dim (int): The shape[0]ality of the random vector.
                mu: The mean (average) value of the distribution.
                sigma: The standard deviation of the distribution.
                decimal (bool, optional): If True, generated values will be of Decimal type. Defaults to False.

            Returns:
                Vector: A random vector of Gaussian (normal) distribution with the specified shape[0]s.

            Raises:
                ArgTypeError: If the shape[0], mean, or standard deviation is not a numerical value.
                RangeError: If the shape[0] is not a positive integer.
        """
        if not isinstance(dim, int): raise ArgTypeError("Must be an integer.")
        if not ((isinstance(mu, Union[int, float, Decimal])) and
                (isinstance(sigma, Union[int, float, Decimal]))):
            raise ArgTypeError("Must be a numerical value.")
        if dim <= 0:
            raise RangeError()

        if decimal:
            return Vector(*[Decimal(random.gauss(mu, sigma)) for k in range(dim)])
        return Vector(*[random.gauss(mu, sigma) for k in range(dim)])

    @staticmethod
    def determinant(*args):
        """
            Calculates the determinant of a square matrix represented by the given vectors.

            The `determinant` method calculates the determinant of a square matrix represented by the provided vectors.
            The method supports matrices of shape[0]s 2x2 and higher.

            Args:
                *args (Vector): One or more vectors representing the rows or columns of the square matrix.

            Returns:
                The determinant of the square matrix.

            Raises:
                ArgTypeError: If any argument is not a vector or if the shape[0]s of the vectors are inconsistent.
                DimensionError: If the shape[0]s of the vectors do not form a square matrix.
                AmountError: If the number of vectors does not match the shape[0] of the square matrix.
        """
        N = args[0].shape[0]
        for k in args:
            if not isinstance(k, Vector):
                raise ArgTypeError("Must be a vector.")
            if N != k.shape[0]:
                raise DimensionError(0)
        if len(args) != N:
            raise AmountError()

        if len(args) == 2 and N == 2:
            return (args[0].values[0] * args[1].values[1]) - (args[0].values[1] * args[1].values[0])

        result = 0
        for k in range(N):
            vector_list = []
            for a in range(1, N):
                temp = []
                for b in range(N):
                    if b != k:
                        temp.append(args[a].values[b])
                vector_list.append(Vector(*temp))
            result += Vector.determinant(*vector_list) * ((-1)**(k)) * args[0].values[k]
        return result

    @staticmethod
    def cross(*args):
        """
            Calculates the cross product of vectors.

            The cross product is a binary operation on two vectors in three-shape[0]al space. It results in a vector that
            is perpendicular to both input vectors. For vectors in higher shape[0]s, the cross product is generalized as the
            determinant of a matrix formed by the input vectors and the standard basis vectors.

            Args:
                *args (Vector): Vectors to compute the cross product.

            Returns:
                Vector: The vector representing the cross product of the input vectors.

            Raises:
                ArgTypeError: If any argument is not a vector or if the shape[0]s of the vectors are inconsistent.
                DimensionError: If the shape[0]s of the vectors do not match the requirements for calculating the cross product.
        """
        N = args[0].shape[0]
        for k in args:
            if not isinstance(k, Vector):
                raise ArgTypeError("Must be a vector.")
            if N != k.shape[0]:
                raise DimensionError(0)

        if len(args) == 2 and N == 2:
            return args[0].values[0] * args[1].values[1] - args[0].values[1] * args[1].values[0]
        if len(args) != N - 1:
            raise AmountError()

        end_list = []
        for k in range(N):
            vector_list = []
            for a in range(N - 1):
                temp = []
                for b in range(N):
                    if b != k:
                        temp.append(args[a].values[b])
                vector_list.append(Vector(*temp))
            end_list.append((Vector.determinant(*vector_list)) * (-1)**(k))
        return Vector(*end_list)

    def outer(v, w):
        """
            Computes the outer product of two vectors.

            The outer product of two vectors results in a matrix where each element (i, j) is the product of the i-th element
            of the first vector and the j-th element of the second vector.

            Args:
                v (Vector): The first vector.
                w (Vector): The second vector.

            Returns:
                Matrix: The matrix representing the outer product of the input vectors.

            Raises:
                ArgTypeError: If either v or w is not a vector.
                DimensionError: If the shape[0]s of v and w are not compatible for computing the outer product.
        """
        if not (isinstance(v, Vector) and isinstance(w, Vector)):
            raise ArgTypeError("Must be a vector.")
        if v.shape[0] != w.shape[0]:
            raise DimensionError(0)

        #v_list = []
        #for i in range(v.shape[0]):
            #temp = [v.values[i] * w.values[j] for j in range(v.shape[0])]
            #for j in range(v.shape[0]):
            #    temp.append(v.values[i] * w.values[j])
            #v_list.append(Vector(*temp))

        return Matrix(*[Vector(*[v.values[i] * w.values[j] for j in range(v.shape[0])]) for i in range(v.shape[0])])

    def cumsum(self):
        """
            Computes the cumulative sum of the elements in the vector.

            Returns the sum of all elements in the vector.

            Returns:
                Union[int, float, Decimal, Infinity, Undefined]: The cumulative sum of the elements in the vector.

            Raises:
                None
        """
        sum = 0
        for k in self.values:
            sum += k
        return sum

    @staticmethod
    def zero(dim: int, decimal: bool = False):
        """
            Generates a zero vector of a specified shape[0].

            Args:
                dim (int): The shape[0]ality of the zero vector to be created.
                decimal (bool, optional): A boolean flag indicating whether the elements of the vector should be Decimal numbers. Defaults to False.

            Returns:
                Vector: A zero vector of the specified shape[0].

            Raises:
                RangeError: If the shape[0] specified is negative.
        """
        # We use the RangeError because shape[0] can be 0.
        if dim < 0:
            raise RangeError()
        if decimal:
            return Vector(*[Decimal(0) for k in range(dim)])
        else:
            return Vector(*[0 for k in range(dim)])

    @staticmethod
    def one(dim: int, decimal: bool = False):
        """
            Generates a vector with all elements set to one.

            Args:
                dim (int): The shape[0]ality of the vector to be created.
                decimal (bool, optional): A boolean flag indicating whether the elements of the vector should be Decimal numbers. Defaults to False.

            Returns:
                Vector: A vector with all elements set to one.

            Raises:
                RangeError: If the shape[0] specified is negative.
        """
        if dim < 0:
            raise RangeError()
        if decimal:
            return Vector(*[Decimal(1) for k in range(dim)])
        else:
            return Vector(*[1 for k in range(dim)])

    def reshape(self, m: int, n: int):
        """
            Reshapes the vector into a matrix with the specified shape[0]s.

            Args:
                m (int): The number of rows in the resulting matrix.
                n (int): The number of columns in the resulting matrix.

            Returns:
                Matrix: A matrix with shape[0]s m x n reshaped from the vector.

            Raises:
                RangeError: If the product of m and n does not equal the shape[0] of the vector.
        """
        if not m * n == self.shape[0]:
            raise RangeError()
        v_list = []
        count = 0
        temp = []
        for k in self.values:
            if count == n:
                count = 0
                v_list.append(Vector(*temp))
                temp.clear()
            temp.append(k)
            count += 1
        v_list.append(Vector(*temp))
        return Matrix(*v_list)

    def rotate(self, i, j, angle, resolution: int = 15):
        """
            Rotates the vector by the specified angle in the plane defined by the indices `i` and `j`.

            Args:
                i (int): Index of the first component defining the plane of rotation.
                j (int): Index of the second component defining the plane of rotation.
                angle: The angle of rotation in degrees.
                resolution (int, optional): The number of points in the rotation. Defaults to 15.

            Returns:
                Vector: The rotated vector.

            Example:
                If `v` is a vector, `v.rotate(0, 1, math.pi / 2)` rotates the vector `v` by
                90 degrees in the plane defined by the first two components.
        """
        return Matrix.givens(self.shape[0], i, j, angle, resolution) * self

    def softmax(self, resolution: int = 15):
        """
            Computes the softmax function of the vector.

            Args:
                resolution (int, optional): The resolution of the softmax function. Defaults to 15.

            Returns:
                Vector: The softmax function of the vector.

            Example:
                If `v` is a vector, `v.softmax()` computes the softmax function of `v`,
                normalizing each element of the vector such that the resulting vector sums to 1.
        """

        temp = Vector(*[e(k, resolution) for k in self.values])
        temp /= temp.cumsum()
        return temp

    def minmax(self):
        """
            Normalizes the vector between 0 and 1.

            Returns:
                Vector: The normalized vector.

            Example:
                If `v` is a vector, `v.minmax()` returns a new vector where each element
                is scaled to the range [0, 1] based on the minimum and maximum values in `v`.
        """
        minima = minimum(self)
        maxima = maximum(self)
        val = maxima - minima
        if val == 0:
            return self
        return Vector(*[(k - minima) / val for k in self.values])

    def relu(self, leak=0, cutoff=0):
        """
            Applies the Rectified Linear Unit (ReLU) activation function element-wise to the vector.

            Args:
                leak: The slope of the negative part of the activation function for values less than
                    zero. Defaults to 0.
                cutoff: Values less than this cutoff will be set to zero. Defaults to 0.

            Returns:
                Vector: A new vector obtained by applying the ReLU activation function to each element of the
                original vector.

            Example:
                If `v` is a vector, `v.relu()` applies the ReLU activation function to each element of `v`.
                For each element `x` in `v`, if `x` is greater than or equal to zero, it remains unchanged.
                Otherwise, it is set to zero. The resulting vector is returned.

            Note:
                - The ReLU activation function is defined as: f(x) = max(0, x), where x is the input.
                - The `leak` parameter controls the slope of the negative part of the function.
                  For traditional ReLU, it is set to 0.
                - The `cutoff` parameter allows setting a threshold below which values will be forced to zero.
                  If `cutoff` is set to a value greater than zero, all elements less than `cutoff` will be set to zero.
        """
        if not ((isinstance(leak, Union[int, float, Decimal, Infinity, Undefined]))
                and (isinstance(cutoff, Union[int, float, Decimal, Infinity, Undefined]))):
            raise ArgTypeError("Must be a numerical value.")

        return Vector(*[ReLU(k, leak, cutoff) for k in self.values])

    def sig(self, a=1, cutoff=None):
        """
            Applies the sigmoid activation function element-wise to the vector.

            Args:
                a: Scaling factor for the input. Defaults to 1.
                cutoff: Values greater than this cutoff will be saturated to 1, and values less than the
                    negative of this cutoff will be saturated to 0. If `None`, no saturation is applied.
                    Defaults to None.

            Returns:
                Vector: A new vector obtained by applying the sigmoid activation function to each element of the
                original vector.

            Example:
                If `v` is a vector, `v.sig()` applies the sigmoid activation function to each element of `v`.
                The sigmoid function transforms each element `x` to the range (0, 1) according to the formula:
                f(x) = 1 / (1 + exp(-a * x)), where x is the input and a is the scaling factor.
                If `cutoff` is specified, values greater than the cutoff saturate to 1, and values less than the
                negative of the cutoff saturate to 0.

            Note:
                - The sigmoid activation function is defined as: f(x) = 1 / (1 + exp(-a * x)), where x is the input.
                - The `a` parameter controls the scaling factor of the input. Higher `a` values lead to sharper
                  transitions around zero.
                - The `cutoff` parameter allows setting a threshold beyond which values saturate to 1 or 0.
                  If `cutoff` is `None`, no saturation is applied.
        """
        if not (isinstance(cutoff, Union[int, float, Decimal, Infinity, Undefined]) or (cutoff is None)):
            raise ArgTypeError("Must be a numerical value.")
        # The reason i do that, i want this to be as fast as possible. I restrain myself to use almost always comprehensions.
        # This is not a code that needs to be readable or sth.
        if cutoff is not None:
            return Vector(*[1 if (x > cutoff) else 0 if (x < -cutoff) else sigmoid(x, a) for x in self.values])
        return Vector(*[sigmoid(x, a) for x in self.values])

    def toInt(self):
        """
            Converts the elements of the vector to integers.

            Returns:
                Vector: A new vector with elements converted to integers.
        """
        return Vector(*[int(k) for k in self.values])

    def toFloat(self):
        """
            Converts the elements of the vector to floating-point numbers.

            Returns:
                Vector: A new vector with elements converted to floating-point numbers.
        """
        return Vector(*[float(k) for k in self.values])

    def toBool(self):
        """
            Converts the elements of the vector to boolean values.

            Returns:
                Vector: A new vector with elements converted to boolean values.
        """
        return Vector(*[bool(k) for k in self.values])

    def toDecimal(self):
        """
            Converts the elements of the vector to Decimal objects.

            Returns:
                Vector: A new vector with elements converted to Decimal objects.
        """
        return Vector(*[Decimal(k) for k in self.values])

    def toVariable(self):
        """
            Converts the elements of the vector to Variable objects.

            Returns:
                Vector: A new vector with elements converted to Variable objects.
        """
        return Vector(*[Variable(k) for k in self.values])

    def map(self, f):
        """
            Applies a function `f` to each element of the vector and returns a new vector with the results.

            Args:
                f (callable): A function that takes an element of the vector as input and returns a transformed value.

            Returns:
                Vector: A new vector containing the results of applying the function `f` to each element of the original vector.

            Example:
                If `v` is a vector, `v.map(lambda x: x ** 2)` returns a new vector where each element is squared.

            Note:
                - The function `f` should accept one argument (an element of the vector) and return a transformed value.
                - The order of elements in the resulting vector matches their order in the original vector.

            Raises:
                TypeError: If the function `f` is not callable.
        """
        if not isinstance(f, Callable): raise ArgTypeError("f must be a callable.")
        return Vector(*[f(k) for k in self.values])

    def filter(self, f):
        """
            Filters the elements of the vector based on a given predicate function `f`.

            Args:
                f (callable): A predicate function that takes an element of the vector as input
                              and returns True if the element should be included, False otherwise.

            Returns:
                Vector: A new vector containing elements from the original vector for which `f` returns True.

            Example:
                If `v` is a vector, `v.filter(lambda x: x > 0)` returns a new vector containing only
                the elements of `v` that are greater than zero.

            Note:
                - The predicate function `f` should accept one argument (an element of the vector) and return a boolean.
                - The order of elements in the resulting vector matches their order in the original vector.
                - If no elements match the predicate, an empty vector is returned.

            Raises:
                TypeError: If the predicate function `f` is not callable.
        """
        if not isinstance(f, Callable):
            raise ArgTypeError("f must be a callable.")
        vals = []
        for k in self.values:
            if f(k):
                vals.append(k)
        return Vector(*vals)

    def sort(self, reverse: bool=False):
        """
            Sorts the elements of the vector in ascending order by default or descending order if `reverse` is True.

            Args:
                reverse (bool, optional): If True, sorts the vector in descending order. Defaults to False.

            Returns:
                Vector: A reference to the sorted vector.

            Example:
                If `v` is a vector, `v.sort()` sorts the elements of `v` in ascending order.
                To sort in descending order, use `v.sort(reverse=True)`.

            Note:
                - This method modifies the vector in place and returns a reference to the sorted vector.
                - If the vector contains elements of mixed types, sorting behavior is based on their comparison.

            Raises:
                TypeError: If the elements of the vector cannot be compared.
        """
        self.values.sort(reverse=reverse)
        return self

    def avg(self):
        """
            Computes the average (mean) of the elements in the vector.

            Returns:
                Union[int, float, Decimal, Infinity, Undefined]: The average of the elements in the vector.

            Example:
                If `v` is a vector, `v.avg()` returns the average value of all elements in the vector.

            Note:
                - The average is computed as the sum of all elements divided by the number of elements in the vector.
                - If the vector is empty, the average is considered as undefined (Undefined).
        """
        if self.shape[0] == 0:
            return Undefined()
        sum = 0
        for k in self.values:
            sum += k
        return sum / self.shape[0]

    def dump(self):
        """
            Sets all data to 0 without changing the shape[0].

            Notes:
                Omits the possibility that elements could be Decimal objects.

        """
        for k in range(self.shape[0]):
            self.values[k] = 0
