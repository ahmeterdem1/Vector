from ..math import *
from ..utils import *
from ..variable import Variable
import random
from typing import Union, Tuple, List, Callable
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

class Matrix:
    """
        Matrix class representing mathematical matrices and supporting various operations.

        Attributes:
            values (list): List containing the elements of the matrix.
            shape (tuple): Dimensionality of the matrix.
    """
    def __init__(self, *args):
        """
            Initializes a Matrix object.

            Args:
                *args (Vector): Variable number of Vector objects to form the matrix.
        """



        for k in args:
            if not isinstance(k, Union[Matrix, Vector, list, tuple]):
                raise ArgTypeError("Must be a vector, matrix, list or tuple.")

        if len(args) == 0:
            self.values = [[]]
            self.shape = (1, 0)

        elif isinstance(args[0], Matrix):
            self.values = [k.copy() for k in args[0].values]
            self.shape = tuple([k for k in args[0].shape])

        elif isinstance(args[0], Vector):
            for k in args:
                if not (args[0].shape[0] == k.shape[0]):
                    raise DimensionError(0)
            self.values = [k.values for k in args]
            self.shape = (len(args), args[0].shape[0],)

        elif isinstance(args[0], Union[list, tuple]):
            for k in args[0]:
                if len(args[0][0]) != len(k):
                    raise DimensionError(0)

            self.values = [k.copy() for k in args[0]]
            self.shape = (len(args[0]), len(args[0][0]),)

    def __str__(self):
        """
            Returns a string representation of the matrix.

            Returns:
                str: String representation of the matrix.
        """
        strs = []
        for k in self.values:
            for l in k:
                strs.append(len(str(l)))
        maximalength = maximum(strs) + 1
        res = ""
        index1 = 0
        for row in self.values:
            index2 = 0
            res += "["
            for item in row:
                i = str(item)
                itemlength = len(i)
                diff = maximalength - itemlength
                res += " " + i
                res += " " * (diff - 1 if diff > 0 else 0)
                if not index2 == len(row) - 1:
                    res += ","
                index2 += 1
            if not index1 == self.shape[0] - 1:
                res += "]\n"
            index1 += 1
        res += "]"
        return res

    def __repr__(self):
        """
            Returns a string representation of the matrix.

            Returns:
                str: String representation of the matrix.
        """
        strs = []
        for k in self.values:
            for l in k:
                strs.append(len(str(l)))
        maximalength = maximum(strs) + 1
        res = ""
        index1 = 0
        for row in self.values:
            index2 = 0
            res += "["
            for item in row:
                i = str(item)
                itemlength = len(i)
                diff = maximalength - itemlength
                res += " " + i
                res += " " * (diff - 1 if diff > 0 else 0)
                if not index2 == len(row) - 1:
                    res += ","
                index2 += 1
            if not index1 == len(self.values) - 1:
                res += "]\n"
            index1 += 1
        res += "]"
        return res

    def __subdimension__(self):
        """
            Returns the subdimension of the matrix.

            Returns:
                list: A list containing the row and column dimensions of the matrix.

            Notes:
                This method is for compatibility with Tensor class.
        """

        return self.shape

    def __getitem__(self, index):
        """
            Returns the element at the specified index.

            Args:
                index (int): Index of the element to retrieve.

            Returns:
                list: The element at the specified index.
        """
        return self.values[index]

    def __setitem__(self, key, value):
        """
            Sets the element at the specified index.

            Args:
                key (int): Index of the element to set.
                value (list): The value to set at the specified index.
        """
        self.values[key] = value

    def __call__(self, *args):
        """
            Returns the value at tuple "args" of each function member of the matrix, as a matrix.

            Returns: the value at tuple "args" of each function member of the matrix, as a matrix.

        """
        return Matrix([[l(*args) for l in k] for k in self.values])

    def __add__(self, arg):
        if isinstance(arg, Union[int, float, Decimal, Infinity, Undefined, Complex, Variable]):
            return Matrix([[l + arg for l in k] for k in self.values])
        if not isinstance(arg, Matrix):
            raise ArgTypeError("Must be a numerical value.")
        if self.shape != arg.shape:
            raise DimensionError(0)
        return Matrix([[self.values[k][l] + arg.values[k][l] for l in range(self.shape[1])] for k in range(self.shape[0])])

    def __radd__(self, arg):
        if isinstance(arg, Union[int, float, Decimal, Infinity, Undefined, Complex, Variable]):
            return Matrix([[l + arg for l in k] for k in self.values])
        raise ArgTypeError("Must be a numerical value.")

    def __iadd__(self, arg):
        if isinstance(arg, Union[int, float, Decimal, Infinity, Undefined, Complex, Variable]):
            return Matrix([[l + arg for l in k] for k in self.values])
        if not isinstance(arg, Matrix):
            raise ArgTypeError("Must be a numerical value.")
        if self.shape != arg.shape:
            raise DimensionError(0)
        return Matrix([[self.values[k][l] + arg.values[k][l] for l in range(self.shape[1])] for k in range(self.shape[0])])

    def __sub__(self, arg):
        if isinstance(arg, Union[int, float, Decimal, Infinity, Undefined, Complex, Variable]):
            return Matrix([[l - arg for l in k] for k in self.values])
        if not isinstance(arg, Matrix):
            raise ArgTypeError("Must be a numerical value.")
        if self.shape != arg.shape:
            raise DimensionError(0)
        return Matrix([[self.values[k][l] - arg.values[k][l] for l in range(self.shape[1])] for k in range(self.shape[0])])

    def __rsub__(self, arg):
        if isinstance(arg, Union[int, float, Decimal, Infinity, Undefined, Complex, Variable]):
            return Matrix([[arg - l for l in k] for k in self.values])
        raise ArgTypeError("Must be a numerical value.")

    def __isub__(self, arg):
        if isinstance(arg, Union[int, float, Decimal, Infinity, Undefined, Complex, Variable]):
            return Matrix([[l - arg for l in k] for k in self.values])
        if not isinstance(arg, Matrix):
            raise ArgTypeError("Must be a numerical value.")
        if self.shape != arg.shape:
            raise DimensionError(0)
        return Matrix([[self.values[k][l] - arg.values[k][l] for l in range(self.shape[1])] for k in range(self.shape[0])])

    def __mul__(self, arg):
        if isinstance(arg, Union[int, float, Decimal, Infinity, Undefined, Complex, Variable]):
            return Matrix([[l * arg for l in k] for k in self.values])
        if isinstance(arg, Vector):
            v = []
            if self.shape[1] != arg.shape[0]:
                raise DimensionError(0)
            for k in range(self.shape[0]):
                sum = 0
                for l in range(arg.shape[0]):
                    sum += self.values[k][l] * arg.values[l]
                v.append(sum)
            return Vector(v)

        if not isinstance(arg, Matrix):
            raise ArgTypeError("Must be a numerical value.")
        if self.shape[1] != arg.shape[0]:
            raise DimensionError(0)

        v = []
        for k in range(self.shape[0]):
            n = []
            for l in range(arg.shape[1]):
                sum = 0
                for m in range(arg.shape[0]):
                    sum += self.values[k][m] * arg.values[m][l]
                n.append(sum)
            v.append(n)
        return Matrix(v)

    def __rmul__(self, arg):
        if isinstance(arg, Union[int, float, Decimal, Infinity, Undefined, Complex, Variable]):
            return Matrix([[l * arg for l in k] for k in self.values])
        raise ArgTypeError("Must be a numerical value.")

    def __neg__(self):
        return Matrix(*[Vector(*[-l for l in k]) for k in self.values])

    def __truediv__(self, arg):
        if not isinstance(arg, Union[int, float, Decimal, Infinity, Undefined, Complex, Variable]):
            raise ArgTypeError("Must be a numerical value.")
        return Matrix([[l / arg for l in k] for k in self.values])

    def __floordiv__(self, arg):
        if not isinstance(arg, Union[int, float, Decimal, Infinity, Undefined, Complex, Variable]):
            raise ArgTypeError("Must be a numerical value.")
        return Matrix([[l // arg for l in k] for k in self.values])

    def __pow__(self, p, decimal: bool = False):
        temp = Matrix.identity(self.shape[0], decimal)
        for k in range(p):
            temp *= self
        return temp

    def determinant(self, choice: str = "echelon"):
        """
            Calculates the determinant of the matrix.

            Args:
                choice (str, optional): The method to use for determinant calculation.
                    It can be either "analytic" for direct computation or "echelon" for
                    echelon form method. Defaults to "echelon".

            Returns:
                Union[int, float, Decimal]: The determinant of the matrix.
        """
        if self.shape == (1, 1,):
            return self.values[0][0]

        if choice == "analytic":
            return Vector.determinant(*[Vector(*k) for k in self.values])

        if choice == "echelon":
            a = self.echelon()
            sum = 1
            for k in range(a.shape[0]):
                sum *= a.values[k][k]
            return sum

        raise RangeError("Not a correct method choice.")

    def __or__(self, arg):
        if not isinstance(arg, Matrix):
            raise ArgTypeError("Must be a numerical value.")
        if self.shape != arg.shape:
            raise DimensionError(0)
        return Matrix([[self.values[k][l] or arg.values[k][l] for l in range(self.shape[1])] for k in range(self.shape[0])])

    def __ior__(self, arg):
        if not isinstance(arg, Matrix):
            raise ArgTypeError("Must be a matrix.")
        if self.shape != arg.shape:
            raise DimensionError(0)
        return Matrix([[self.values[k][l] or arg.values[k][l] for l in range(self.shape[1])] for k in range(self.shape[0])])

    def __and__(self, arg):
        if not isinstance(arg, Matrix):
            raise ArgTypeError("Must be a matrix.")
        if self.shape != arg.shape:
            raise DimensionError(0)
        return Matrix([[self.values[k][l] and arg.values[k][l] for l in range(self.shape[1])] for k in range(self.shape[0])])

    def __iand__(self, arg):
        if not isinstance(arg, Matrix):
            raise ArgTypeError("Must be a matrix.")
        if self.shape != arg.shape:
            raise DimensionError(0)
        return Matrix([[self.values[k][l] and arg.values[k][l] for l in range(self.shape[1])] for k in range(self.shape[0])])

    def __xor__(self, arg):
        if not isinstance(arg, Matrix):
            raise ArgTypeError("Must be a matrix.")
        if self.shape != arg.shape:
            raise DimensionError(0)
        return Matrix([[self.values[k][l] ^ arg.values[k][l] for l in range(self.shape[1])] for k in range(self.shape[0])])

    def __ixor__(self, arg):
        if not isinstance(arg, Matrix):
            raise ArgTypeError("Must be a matrix.")
        if self.shape != arg.shape:
            raise DimensionError(0)
        return Matrix([[self.values[k][l] ^ arg.values[k][l] for l in range(self.shape[1])] for k in range(self.shape[0])])

    def __invert__(self):
        return Matrix([[int(not l) for l in k] for k in self.values])

    def __eq__(self, arg):
        if not isinstance(arg, Matrix):
            raise ArgTypeError("Must be a matrix.")
        return self.values == arg.values

    def append(self, arg):
        """
            Appends a vector to the matrix as a new row.

            Args:
                arg (Vector): The vector to be appended as a new row to the matrix.

            Raises:
                ArgTypeError: If the argument is not a vector.
                DimensionError: If the dimension of the vector does not match the matrix's column dimension.
        """
        if not isinstance(arg, Vector):
            raise ArgTypeError("Must be a vector.")
        if self.shape[1] != arg.shape[0]:
            raise DimensionError(0)
        self.values.append(arg.values)
        self.shape = (self.shape[0] + 1, self.shape[1],)

    def copy(self):
        """
            Creates a copy of the matrix.

            Returns:
                Matrix: A new matrix object containing a copy of the original matrix.

            Notes:
                The copy method performs a deep copy of the matrix, including its values and dimension.
        """
        return Matrix(self.values)  # initializer already makes a deep copy

    def pop(self, ord=-1):
        """
            Removes and returns a vector from the matrix.

            Args:
                ord (int, optional): The index of the vector to remove. Default is -1,
                    which removes the last vector from the matrix.

            Returns:
                Vector: The vector removed from the matrix.

            Raises:
                RangeError: If the specified index is out of range.
        """
        try:
            self.values[ord]
        except IndexError:
            raise RangeError()

        popped = self.values.pop(ord)
        self.shape = (self.shape[0] - 1, self.shape[1],)
        return Vector(popped)

    def dot(self, arg):
        """
            Computes the dot product of two matrices.

            Args:
                arg (Matrix): The matrix to compute the dot product with.

            Returns:
                float: The dot product of the two matrices.

            Raises:
                ArgTypeError: If the argument is not a matrix.
                DimensionError: If the dimensions of the matrices are incompatible for dot product.

            Notes:
                The dot product of two matrices is computed by taking the sum of the products
                of corresponding elements in the matrices. Main purpose of this method is the
                compatibility with the Tensor class.
        """
        if not isinstance(arg, Matrix):
            raise ArgTypeError("Must be a matrix.")
        if self.shape != arg.shape:
            raise DimensionError(0)

        sum = 0
        M = self.shape[1]
        for k in range(self.shape[0]):
            for l in range(M):
                sum += self.values[k][l] * arg.values[k][l]
        return sum

    def transpose(self):
        """
            Transposes the matrix by swapping its rows and columns.

            Returns:
                Matrix: The transposed matrix.
        """

        return Matrix([[self.values[l][k] for l in range(self.shape[0])] for k in range(self.shape[1])])

    def conjugate(self):
        """
            Computes the conjugate of the matrix.

            Returns:
                Matrix: The conjugate of the matrix.

            Notes:
                The conjugate of a matrix involves taking the conjugate of each complex
                element in the matrix. For real numbers, the conjugate is the number itself.

        """

        return Matrix([[self.values[k][l].conjugate() if isinstance(self[k][l], Complex) else self.values[k][l] for l in range(self.shape[1])]
                       for k in range(self.shape[0])])

    def normalize(self, d_method: str = "echelon"):
        """
            Normalizes the matrix using the specified method for determinant calculation.

            Args:
                d_method (str, optional): The method to use for determinant calculation. Defaults to "echelon".

            Returns:
                Matrix: The normalized matrix.

            Raises:
                ArgTypeError: If the determinant calculation method is invalid.
                Undefined: If the determinant is zero, making normalization impossible.

            Notes:
                The normalization process involves dividing each element of the matrix by its determinant.
                The determinant calculation method can be either "echelon" or "analytic".
                If the determinant is zero, the matrix cannot be normalized, and an Undefined object is returned.
        """
        d = self.determinant(d_method)
        if d == 0:
            return Undefined()
        return Matrix([[val / d for val in k] for k in self.values])

    def hconj(self):  # Hermitian conjugate
        """
            Computes the Hermitian conjugate of the matrix.

            Returns:
                Matrix: The Hermitian conjugate of the matrix.

            Notes:
                The Hermitian conjugate of a matrix is obtained by taking the conjugate transpose
                of the matrix, where each element is replaced by its complex conjugate, and then
                the rows and columns are swapped.
        """
        return self.transpose().conjugate()

    def norm(self, p: Union[int, float, Decimal, Variable, str] = 2):
        """
            Computes the p-norm of the matrix.

            Args:
                p (Union[int, float, Decimal, Variable, str]): The p value representing
                    the L_{p} norm. If p="max", infinity norm is returned. If p="min",
                    -infinity norm is returned.

            Returns:
                Union[int, float, Decimal]: The norm of the matrix.

            Raises:
                ArgTypeError: If p is not of type Union[int, float, Decimal, Variable, str].
        """

        if not isinstance(p, Union[int, float, Decimal, Variable, str]):
            raise ArgTypeError("Must be a numerical value, or 'max' or 'min'.")

        if p == 'max':
            return maximum(self.values)

        if p == 'min':
            return minimum(self.values)

        if isinstance(p, str):
            raise ArgTypeError("String value must be one of ['max', 'min'].")

        _sum = 0
        for k in range(self.shape[0]):
            for l in range(self.shape[1]):
                _sum += self.values[k][l] ** p
        return _sum ** (1/p)

    def inverse(self, method: str = "iterative", resolution: int = 10, lowlimit=0.0000000001, highlimit=100000, decimal: bool = False):
        """
            Computes the inverse of the matrix using specified method.

            Args:
                method (str, optional): The method to use for inverse computation. Defaults to "iterative".
                resolution (int, optional): The resolution for iterative or Neumann series method. Defaults to 10.
                lowlimit (Union[int, float, Decimal], optional): The lower limit for gauss method. Defaults to 0.0000000001.
                highlimit (Union[int, float, Decimal], optional): The upper limit for gauss method. Defaults to 100000.
                decimal (bool, optional): Whether to use decimal precision for computation. Defaults to False.

            Returns:
                Matrix: The inverse of the matrix.

            Raises:
                ArgTypeError: If the method is not supported.
                RangeError: If the resolution is less than 1.
                DimensionError: If the matrix is not square.
                ArgTypeError: If the lowlimit or highlimit is not a numerical value.

            Notes:
                The method parameter determines the algorithm used for inverse computation. Supported methods include
                "gauss", "analytic", "iterative", "neumann" and "lu". The resolution parameter affects the accuracy
                of iterative or Neumann series methods. The lowlimit and highlimit parameters are used to control
                the convergence behavior of the gauss method. The inverse matrix is obviously only defined for
                square matrices. LU method only uses "decimal" choice.
        """
        if method not in ["gauss", "analytic", "iterative", "neumann", "lu"]:
            raise ArgTypeError()
        if resolution < 1:
            raise RangeError()
        if not ((isinstance(lowlimit, Union[int, float, Decimal]))
                and (isinstance(highlimit, Union[int, float, Decimal]))):
            raise ArgTypeError("Must be a numerical value.")
        if not (self.shape[0] == self.shape[1]):
            raise DimensionError(2)
        if self.shape == (1, 1,):
            return 1 / self.values[0][0]

        if method == "analytic":
            det = self.determinant()
            if not det:
                return
            end = []
            N = self.shape[0]
            for k in range(N):
                temp = []
                for l in range(N):
                    sub = []
                    for a in range(N):
                        n = []
                        for b in range(N):
                            if k != a and l != b:
                                n.append(self.values[a][b])
                        if len(n) > 0:
                            sub.append(n)
                    temp.append(((-1)**(k+l)) * Matrix.determinant(Matrix(sub)))
                end.append(temp)
            return Matrix(end).transpose() / det

        elif method == "gauss":

            N = self.shape[0]
            i = Matrix.identity(N, decimal)
            i_values = i.values.copy()
            v = self.values.copy()
            taken_list = []
            taken_list_i = []
            counter = 0

            for k in range(N):
                for l in range(N):
                    if self.values[k][l] != 0 and l not in taken_list:
                        v[l] = self.values[k]
                        i_values[l] = i.values[k]
                        counter += 1
                        if l != k and counter % 2 == 0:
                            v[l] = [-z for z in self.values[k]]
                            i_values[l] = [-z for z in i.values[k]]
                        else:
                            v[l] = self.values[k]
                            i_values[l] = i.values[k]
                        taken_list.append(l)
                        taken_list_i.append(l)
                        break
                    elif self.values[k][l] != 0 and l in taken_list:
                        for m in range(l, N):
                            if m not in taken_list:
                                v[m] = self.values[k]
                                i_values[m] = i.values[k]
                                counter += 1
                                if m != k and counter % 2 == 0:
                                    v[m] = [-z for z in self.values[k]]
                                    i_values[m] = [-z for z in i.values[k]]

            for k in range(N):
                if v[k][k] == 0:
                    continue
                for l in range(N):
                    if l == k:
                        continue
                    try:
                        factor = (v[l][k]) / (v[k][k])
                        if abs(factor) < lowlimit or abs(factor) > highlimit:
                            factor = 0
                        factored_list = [v[l][m] - (factor * v[k][m]) for m in range(N)]
                        factored_list_i = [i_values[l][m] - (factor * i_values[k][m]) for m in
                                           range(N)]
                        v[l] = factored_list
                        i_values[l] = factored_list_i
                    except ZeroDivisionError:
                        continue

            v = v[::-1]
            iden_values = i_values.copy()
            iden_values = iden_values[::-1]

            for k in range(N):
                if v[k][k] == 0:
                    continue
                for l in range(N):
                    if l == k:
                        continue
                    try:
                        factor = (v[l][k]) / (v[k][k])
                        if abs(factor) < lowlimit or abs(factor) > highlimit:
                            factor = 0
                        factored_list = [v[l][m] - (factor * v[k][m]) for m in range(N)]
                        factored_list_i = [iden_values[l][m] - (factor * iden_values[k][m]) for m in
                                           range(N)]
                        v[l] = factored_list
                        iden_values[l] = factored_list_i
                    except ZeroDivisionError:
                        continue

            iden_values = iden_values[::-1]
            v = v[::-1]

            for k in range(N):
                if v[k][k] == 0:  # Might introduce cutoffs here too.
                    continue
                for l in range(N):
                    if l == k:
                        continue
                    try:
                        factor = (v[l][k]) / (v[k][k])
                        if abs(factor) < lowlimit or abs(factor) > highlimit:
                            factor = 0
                        factored_list = [v[l][m] - (factor * v[k][m]) for m in range(N)]
                        factored_list_i = [iden_values[l][m] - (factor * iden_values[k][m]) for m in
                                           range(N)]
                        v[l] = factored_list
                        iden_values[l] = factored_list_i
                    except ZeroDivisionError:
                        continue

            for k in range(N):
                try:
                    # Not using "map" may be faster. I haven't tried it yet.
                    iden_values[k] = list(map(lambda x: x if (abs(x) > lowlimit) else 0,
                                              [iden_values[k][l] / v[k][k] for l in range(N)]))
                except:
                    pass

            return Matrix(*[Vector(*k) for k in iden_values])

        elif method == "iterative":
            tpose = self.transpose()
            control_matrix = self * tpose
            sum_list = []
            for k in control_matrix.values:
                sum = 0
                for l in k:
                    sum += abs(l)
                sum_list.append(sum)
            max = 0  # Sums consist of absolute values, this is perfectly fine to do.
            for k in sum_list:
                if k > max:
                    max = k

            alpha = 1 / max

            guess = tpose * alpha

            identity = Matrix.identity(self.shape[0], decimal) * 2

            for k in range(resolution):
                guess = guess * (identity - self * guess)  # Since "guess" changes each iteration, there is no
                                                           # way to further optimize this.
            return guess

        elif method == "neumann":
            # don't forget to calibrate the resolution here
            i = Matrix.identity(self.shape[0], decimal)
            M = self - i

            for k in range(resolution):
                i += (-1)**(k+1) * pow(M, k + 1, decimal)  # This is mandatory, because decimal

            return i

        elif method == "lu":
            N = self.shape[0]

            U = self.values.copy()
            taken_list = []
            counter = 0

            for k in range(N):
                for l in range(N):
                    if self.values[k][l] != 0 and l not in taken_list:
                        counter += 1
                        if l != k and counter % 2 == 0:
                            U[l] = [-z for z in self.values[k]]
                        else:
                            U[l] = self.values[k]
                        taken_list.append(l)
                        break
                    elif self.values[k][l] != 0 and l in taken_list:
                        for m in range(l, N):
                            if m not in taken_list:
                                counter += 1
                                if m != k and counter % 2 == 0:
                                    U[m] = [-z for z in self.values[k]]
                                else:
                                    U[m] = self.values[k]

            del taken_list, counter

            history = []  # This will hold elementary operations
            for k in range(N):
                if U[k][k] == 0:  # Might introduce a cutoff here
                    continue
                for l in range(k + 1, N):
                    try:
                        factor = U[l][k] / U[k][k]
                        if abs(factor) < 0.0000000001:
                            factor = 0
                        factored_list = [U[l][m] - (factor * U[k][m]) for m in range(N)]
                        U[l] = factored_list
                        history.append((l, k, factor))
                    except ZeroDivisionError:
                        continue

            L = Matrix.identity(N, decimal).values
            while history:
                l, k, factor = history.pop()  # Reverse the elementary operations
                try:
                    factored_list = [L[l][m] + factor * L[k][m] for m in range(N)]
                    L[l] = factored_list
                except ZeroDivisionError:
                    continue

            del history

            sum: float
            Bcolumn: list
            Xcolumn: list

            B = [[None] * N for k in range(N)]
            for k in range(N):

                for i in range(N):
                    sum = 0
                    for j in range(i):
                        sum += L[i][j] * B[k][j]
                    B[k][i] = 1 - sum if i == k else -sum

            X = [[None] * N for k in range(N)]

            for k in range(N):
                for i in range(N - 1, -1, -1):
                    sum = 0
                    for j in range(N - 1, i, -1):
                        sum += U[i][j] * X[k][j]
                    X[k][i] = (B[k][i] - sum) / U[i][i]

            return Matrix([[X[l][k] for l in range(N)] for k in range(N)])

    @staticmethod
    def identity(dim, decimal: bool = False):
        """
            Creates an identity matrix of the specified dimension.

            Args:
                dim (int): The dimension of the identity matrix.
                decimal (bool, optional): Whether to use decimal precision for computation. Defaults to False.

            Returns:
                Matrix: The identity matrix of the specified dimension.

            Raises:
                ArgTypeError: If the dimension is not an integer.
                RangeError: If the dimension is less than or equal to 0.
        """
        if not isinstance(dim, int):
            raise ArgTypeError("Must be an integer.")
        if dim <= 0:
            raise RangeError()

        if decimal:
            return Matrix([[Decimal(1) if k == l else Decimal(0) for l in range(dim)] for k in range(dim)])
        return Matrix([[1 if k == l else 0 for l in range(dim)] for k in range(dim)])

    @staticmethod
    def zero(a, b, decimal: bool = False):
        """
            Creates a zero matrix with the specified number of rows and columns.

            Args:
                a (int): The number of rows in the zero matrix.
                b (int): The number of columns in the zero matrix.
                decimal (bool, optional): Whether to use decimal precision for computation. Defaults to False.

            Returns:
                Matrix: The zero matrix with the specified number of rows and columns.

            Raises:
                ArgTypeError: If rows or columns are not integers.
                RangeError: If rows or columns are less than or equal to 0.
        """
        if not (isinstance(a, int) and isinstance(b, int)):
            raise ArgTypeError("Must be an integer.")
        if a <= 0 or b <= 0:
            raise RangeError()

        if decimal:
            return Matrix([[Decimal(0) for l in range(b)] for k in range(a)])
        return Matrix([[0 for l in range(b)] for k in range(a)])

    @staticmethod
    def one(a, b, decimal: bool = False):
        """
            Creates a one matrix with the specified number of rows and columns.

            Args:
                a (int): The number of rows in the one matrix.
                b (int): The number of columns in the one matrix.
                decimal (bool, optional): Whether to use decimal precision for computation. Defaults to False.

            Returns:
                Matrix: The one matrix with the specified number of rows and columns.

            Raises:
                ArgTypeError: If rows or columns are not integers.
                RangeError: If rows or columns are less than or equal to 0.
        """
        if not (isinstance(a, int) and isinstance(b, int)):
            raise ArgTypeError("Must be an integer.")
        if a <= 0 or b <= 0:
            raise RangeError()

        if decimal:
            return Matrix([[Decimal(1) for l in range(b)] for k in range(a)])
        return Matrix([[1 for l in range(b)] for k in range(a)])

    @staticmethod
    def randMint(m, n, a, b, decimal: bool = False):
        """
            Generates a matrix of random integers with the specified dimensions and range.

            Args:
                m (int): The number of rows in the matrix.
                n (int): The number of columns in the matrix.
                a (int): The lower bound of the random integer range.
                b (int): The upper bound of the random integer range.
                decimal (bool, optional): Whether to use decimal precision for computation. Defaults to False.

            Returns:
                Matrix: A matrix of random integers with dimensions m x n.

            Raises:
                ArgTypeError: If m, n, a, or b are not integers.
                RangeError: If m or n are less than or equal to 0.
        """
        if not (isinstance(m, int) and isinstance(n, int) and isinstance(a, int) and isinstance(b, int)):
            raise ArgTypeError("Must be an integer.")
        if m <= 0 or n <= 0:
            raise RangeError()

        if decimal:
            return Matrix([[Decimal(random.randint(0, 1)) for l in range(n)] for k in range(m)])
        return Matrix([[random.randint(a, b) for l in range(n)] for k in range(m)])

    @staticmethod
    def randMfloat(m, n, a, b, decimal: bool = False):
        """
            Generates a matrix of random floating-point numbers with the specified dimensions and range.

            Args:
                m (int): The number of rows in the matrix.
                n (int): The number of columns in the matrix.
                a (float): The lower bound of the random floating-point range.
                b (float): The upper bound of the random floating-point range.
                decimal (bool, optional): Whether to use decimal precision for computation. Defaults to False.

            Returns:
                Matrix: A matrix of random floating-point numbers with dimensions m x n.

            Raises:
                ArgTypeError: If m, n, a, or b are not numerical values.
                RangeError: If m or n are less than or equal to 0.
        """
        if not (isinstance(m, int) and isinstance(n, int)):
            raise ArgTypeError("Must be an integer.")
        if not ((isinstance(a, Union[int, float, Decimal]))
                and (isinstance(b, Union[int, float, Decimal]))):
            raise ArgTypeError("Must be a numerical value")
        if m <= 0 or n <= 0:
            raise RangeError()

        if decimal:
            return Matrix([[Decimal(random.uniform(a, b)) for l in range(n)] for k in range(m)])
        return Matrix([[random.uniform(a, b) for l in range(n)] for k in range(m)])

    @staticmethod
    def randMbool(m, n, decimal: bool = False):
        """
            Generates a matrix of random boolean values with the specified dimensions.

            Args:
                m (int): The number of rows in the matrix.
                n (int): The number of columns in the matrix.
                decimal (bool, optional): Whether to use decimal precision for computation. Defaults to False.

            Returns:
                Matrix: A matrix of random boolean values with dimensions m x n.

            Raises:
                ArgTypeError: If m or n are not integers.
                RangeError: If m or n are less than or equal to 0.
        """
        if not (isinstance(m, int) and isinstance(n, int)):
            raise ArgTypeError("Must be an integer.")
        if m <= 0 or n <= 0:
            raise RangeError()

        if decimal:
            return Matrix([[Decimal(random.randint(0, 1)) for l in range(n)] for k in range(m)])
        return Matrix([[random.randint(0, 1) for l in range(n)] for k in range(m)])

    @staticmethod
    def randMgauss(m, n, mu, sigma, decimal: bool = False):
        """
            Generates a matrix of random numbers following a Gaussian (normal) distribution.

            Args:
                m (int): The number of rows in the matrix.
                n (int): The number of columns in the matrix.
                mu (float): The mean (average) value of the distribution.
                sigma (float): The standard deviation (spread) of the distribution.
                decimal (bool, optional): Whether to use decimal precision for computation. Defaults to False.

            Returns:
                Matrix: A matrix of random numbers following a Gaussian distribution with dimensions m x n.

            Raises:
                ArgTypeError: If m, n, mu, or sigma are not numerical values.
                RangeError: If m or n are less than or equal to 0.
        """
        if not (isinstance(m, int) and isinstance(n, int)):
            raise ArgTypeError("Must be an integer.")
        if not ((isinstance(mu, Union[int, float, Decimal]))
                and (isinstance(sigma, Union[int, float, Decimal]))):
            raise ArgTypeError("Must be a numerical value.")
        if m <= 0 or n <= 0:
            raise RangeError()

        if decimal:
            return Matrix([[Decimal(random.gauss(mu, sigma)) for l in range(n)] for k in range(m)])
        return Matrix([[random.gauss(mu, sigma) for l in range(n)] for k in range(m)])

    def echelon(self):
        """
            Computes the echelon form of the matrix.

            Returns:
                Matrix: The echelon form of the matrix.
        """
        v: list = self.values.copy()
        taken_list: list = []
        counter = 0

        N, M = self.shape[0], self.shape[1]

        for k in range(N):
            for l in range(M):
                if self.values[k][l] != 0 and l not in taken_list:
                    counter += 1
                    if l != k and counter % 2 == 0:
                        v[l] = [-z for z in self.values[k]]
                    else:
                        v[l] = self.values[k]
                    taken_list.append(l)
                    break
                elif self.values[k][l] != 0 and l in taken_list:
                    for m in range(l, N):
                        if m not in taken_list:
                            v[m] = self.values[k]
                            counter += 1
                            if m != k and counter % 2 == 0:
                                v[m] = [-z for z in self.values[k]]
        for k in range(N):
            if v[k][k] == 0:  # Might introduce a cutoff here
                continue
            for l in range(N):
                if l == k:
                    continue
                try:
                    factor = (v[l][k]) / (v[k][k])
                    if abs(factor) < 0.0000000001:
                        factor = 0
                    factored_list = [v[l][m] - (factor * v[k][m]) for m in range(M)]
                    v[l] = factored_list
                except ZeroDivisionError:
                    continue
        taken_list = []
        end_list = v.copy()
        for k in range(N):
            for l in range(M):
                if v[k][l] != 0 and l not in taken_list:
                    end_list[l] = v[k]
                    counter += 1
                    if k != l and counter % 2 == 0:
                        end_list[l] = [-z for z in v[k]]
                    taken_list.append(l)
                    break
                elif v[k][l] != 0 and l in taken_list:
                    for m in range(l, N):
                        if m not in taken_list:
                            end_list[m] = v[k]
                            counter += 1
                            if m != l and counter % 2 == 0:
                                end_list[m] = [-z for z in v[k]]
        return Matrix(end_list)

    def cramer(self, number: int):
        """
            Solves a linear system using Cramer's Rule.

            Args:
                a (Matrix): The coefficient matrix of the linear system.
                number (int): The column number corresponding to the solution.

            Returns:
                float: The solution to the linear system.

            Raises:
                ArgTypeError: If a is not a Matrix object or number is not an integer.
                RangeError: If number is out of range.
        """

        N = self.shape[1] - 1
        if not number < N or number < 0:
            raise RangeError()

        M = self.shape[0]

        first = Matrix([[self.values[k][l] if l != number else self.values[k][N] for l in range(N)] for k in range(M)]).determinant()

        second = Matrix([[self.values[k][l] for l in range(N)] for k in range(M)]).determinant()
        try:
            sol = first / second
        except ZeroDivisionError:
            sol = None
        return sol

    def cumsum(self):
        """
            Computes the cumulative sum of all elements in the matrix.
            Returns the summed values as a vector.

            Returns:
                float: The cumulative sum of all elements.

        """
        sum = 0
        new_matrix = [None] * self.shape[0] * self.shape[1]
        count = 0
        for k in self.values:
            for l in k:
                sum += l
                new_matrix[count] = sum
                count += 1

        return Vector(new_matrix)

    def reshape(self, shape: Union[List[int], Tuple[int]]):
        """
            Reshapes the matrix to the specified dimensions.

            Args:
                shape (Union[List[int], Tuple[int]]): The dimensions to reshape the matrix into.

            Returns:
                Vector: The reshaped matrix.

            Raises:
                AmountError: If the number of arguments is not in the range [1, 2].
                RangeError: If any dimension value is less than or equal to 0, or if the total number of elements does not match the original matrix size.

            Notes:
                This method reshapes the matrix to the specified dimensions while preserving the order of elements.
        """
        if not (0 < len(shape) < 3):
            raise AmountError()
        for k in shape:
            if not isinstance(k, int):
                raise RangeError()
            if k <= 0: raise RangeError()

        temp = []
        for k in self.values:
            for l in k:
                temp.append(l)
        v = Vector(*temp)
        if len(shape) == 1:
            if shape[0] != self.shape[0] * self.shape[1]:
                raise RangeError()
            temp = []
            for k in self.values:
                for l in k:
                    temp.append(l)
            return v
        if shape[0] * shape[1] != self.shape[0] * self.shape[1]:
            raise RangeError()
        return v.reshape(shape[0], shape[1])

    def eigenvalue(self, resolution: int = 10, decimal: bool = False):
        """
            Computes the eigenvalues of a square matrix using the QR algorithm.

            Args:
                resolution (int, optional): The number of iterations for the QR algorithm. Defaults to 10.
                decimal (bool, optional): Whether to use decimal precision for computation. Defaults to False.

            Returns:
                list: The eigenvalues of the matrix.

            Raises:
                DimensionError: If the matrix is not square.
                RangeError: If resolution is less than 1.
        """
        if self.shape[0] != self.shape[1]:
            raise DimensionError(2)
        if resolution < 1:
            raise RangeError()

        to_work = self.copy()
        for k in range(resolution):
            Q, R = to_work.qr(decimal)
            to_work = R * Q

        return [to_work.values[k][k] for k in range(len(to_work.values))]

    def lu(self, decimal: bool = False):
        """
            Applies LU decomposition to self.

            Args:
                decimal (bool): The choice to use the decimal.Decimal
                    objects.

            Returns:
                L and U matrices, as a tuple, in order.

            Raises:
                DimensionError: self must be a square matrix. Otherwise,
                    this error is raised.
        """

        N = self.shape[0]

        if N != self.shape[1]:
            raise DimensionError(2)

        U = self.values.copy()
        taken_list = []
        counter = 0

        for k in range(N):
            for l in range(N):
                if self.values[k][l] != 0 and l not in taken_list:
                    counter += 1
                    if l != k and counter % 2 == 0:
                        U[l] = [-z for z in self.values[k]]
                    else:
                        U[l] = self.values[k]
                    taken_list.append(l)
                    break
                elif self.values[k][l] != 0 and l in taken_list:
                    for m in range(l, N):
                        if m not in taken_list:
                            counter += 1
                            if m != k and counter % 2 == 0:
                                U[m] = [-z for z in self.values[k]]
                            else:
                                U[m] = self.values[k]

        history = []  # This will hold elementary operations
        for k in range(N):
            if U[k][k] == 0:  # Might introduce a cutoff here
                continue
            for l in range(k + 1, N):
                try:
                    factor = U[l][k] / U[k][k]
                    if abs(factor) < 0.0000000001:
                        factor = 0
                    factored_list = [U[l][m] - (factor * U[k][m]) for m in range(N)]
                    U[l] = factored_list
                    history.append((l, k, factor))
                except ZeroDivisionError:
                    continue

        L = Matrix.identity(N, decimal)
        while history:
            l, k, factor = history.pop()  # Reverse the elementary operations
            try:
                factored_list = [L.values[l][m] + factor * L.values[k][m] for m in range(N)]
                L.values[l] = factored_list
            except ZeroDivisionError:
                continue

        return L, Matrix(U)


    def qr(self, decimal: bool = False):
        """
            Computes the QR decomposition of the matrix.

            Args:
                decimal (bool, optional): Whether to use decimal precision for computation. Defaults to False.

            Returns:
                Tuple[Matrix, Matrix]: The Q and R matrices of the QR decomposition.

            Raises:
                DimensionError: If the matrix is not square.
        """
        if self.shape[0] != self.shape[1]:
            raise DimensionError(2)

        v_list = [Vector(*k) for k in self.transpose()]

        if not Vector.does_span(*v_list):
            m = Matrix.zero(self.shape[0], self.shape[0], decimal)
            return m, m
        result_list = [k.unit() for k in Vector.spanify(*v_list)]

        Q_t = Matrix(*result_list)
        return Q_t.transpose(), Q_t * self

    def cholesky(self):
        """
            Computes the Cholesky decomposition of the matrix.

            Returns:
                Matrix: The lower triangular Cholesky factor of the matrix.

            Raises:
                DimensionError: If the matrix is not square.
        """

        N = self.shape[0]
        if N != self.shape[1]:
            raise DimensionError(2)

        L = Matrix.zero(N, N)
        L.values[0][0] = sqrt(self.values[0][0])

        for i in range(N):
            for j in range(i + 1):
                sum = 0
                for k in range(j):
                    sum += L.values[i][k] * L.values[j][k]

                if i == j:
                    L.values[i][j] = sqrt(self.values[i][i] - sum)
                else:
                    L.values[i][j] = (1.0 / L.values[j][j]) * (self.values[i][j] - sum)
        return L

    def get_diagonal(self):
        """
            Retrieves the diagonal elements of the matrix.

            Returns:
                Matrix: A matrix containing only the diagonal elements of the original matrix.

            Raises:
                DimensionError: If the matrix is not square.

            Notes:
                This method is specifically useful for L + D + U decomposition.
        """
        N = self.shape[0]
        if N != self.shape[1]:
            raise DimensionError(2)

        v_list = []
        for k in range(N):
            temp = [0 for i in range(N)]
            for l in range(N):
                if l == k:
                    temp[l] = self[k][l]
            v_list.append(temp)

        return Matrix(v_list)

    def get_lower(self):
        """
            Retrieves the lower triangular portion of the matrix.

            Returns:
                Matrix: A matrix containing only the elements below the main diagonal of the original matrix.

            Raises:
                DimensionError: If the matrix is not square.


            Notes:
                This method is specifically useful for L + D + U decomposition.
        """
        N = self.shape[0]
        if N != self.shape[1]:
            raise DimensionError(2)

        return Matrix([[self.values[k][i] if i < k else 0 for i in range(N)] for k in range(N)])

    def get_upper(self):
        """
            Retrieves the upper triangular portion of the matrix.

            Returns:
                Matrix: A matrix containing only the elements above the main diagonal of the original matrix.

            Raises:
                DimensionError: If the matrix is not square.

            Notes:
                This method is specifically useful for L + D + U decomposition.
        """
        N = self.shape[0]
        if N != self.shape[1]:
            raise DimensionError(2)

        return Matrix([[self.values[k][i] if i > k else 0 for i in range(N)] for k in range(N)])

    @staticmethod
    def givens(dim, i, j, angle, resolution: int = 15):
        """
            Generates a Givens rotation matrix of the specified dimensions and parameters.

            Args:
                dim (int): The dimension of the square matrix.
                i (int): The row index of the first element to rotate.
                j (int): The row index of the second element to rotate.
                angle (float): The rotation angle in degrees.
                resolution (int, optional): The number of iterations for angle resolution. Defaults to 15.

            Returns:
                Matrix: A Givens rotation matrix.

            Raises:
                RangeError: If i or j are out of range.
                RangeError: If resolution is less than 1.

            Notes:
                A Givens rotation matrix is used to introduce zeros into a matrix by rotating selected rows.
        """
        if i >= dim or j >= dim:
            raise RangeError()

        if resolution < 1:
            raise RangeError()

        v_list = [[0 if l != k else 1 for l in range(dim)] for k in range(dim)]

        c = cos(angle, resolution=resolution)
        s = sin(angle, resolution=resolution)
        v_list[i][i] = c
        v_list[j][j] = c
        v_list[i][j] = s
        v_list[j][i] = -s
        return Matrix(v_list)

    def frobenius_product(a, b):
        """
            Computes the Frobenius inner product of two matrices.

            Args:
                a (Matrix): The first matrix.
                b (Matrix): The second matrix.

            Returns:
                float: The Frobenius inner product of the matrices.

            Raises:
                ArgTypeError: If a or b are not Matrix objects.
                DimensionError: If the dimensions of a and b are not equal.

            Notes:
                The only difference of this method from dot product, is conjugation.
                For Real valued matrices, there is no difference at all.
        """
        if not (isinstance(a, Matrix) and isinstance(b, Matrix)):
            raise ArgTypeError("Must be a matrix.")
        if a.shape != b.shape:
            raise DimensionError(0)

        temp = a.copy().conjugate()

        result = 0
        M = a.shape[1]
        for i in range(a.shape[0]):
            for j in range(M):
                result += temp[i][j] * b[i][j]

        return result

    def trace(self):
        """
            Computes the trace of the square matrix.

            Returns:
                float: The trace of the matrix.

            Raises:
                DimensionError: If the matrix is not square.
        """
        if self.shape[0] != self.shape[1]:
            raise DimensionError(2)
        sum = 0
        for k in range(self.shape[0]):
            sum += self.values[k][k]
        return sum

    def diagonals(self):
        """
            Retrieves the diagonal elements of the matrix.

            Returns:
                list: A list containing the diagonal elements of the matrix.

            Raises:
                DimensionError: If the matrix is not square.
        """
        N = self.shape[0]
        if N != self.shape[1]:
            raise DimensionError(2)

        return [self.values[k][k] for k in range(N)]

    def diagonal_mul(self):
        """
            Computes the product of the diagonal elements of the square matrix.

            Returns:
                float: The product of the diagonal elements.

            Raises:
                DimensionError: If the matrix is not square.
        """
        N = self.shape[0]
        if N != self.shape[1]:
            raise DimensionError(2)

        sum = 1
        for k in range(N):
            sum *= self.values[k][k]
        return sum

    def gauss_seidel(self, b: Vector, initial=None, resolution: int = 20, decimal: bool = False):
        """
            Solves a system of linear equations using the Gauss-Seidel method.

            Args:
                b (Vector): The vector of constant terms in the equations.
                initial (Vector, optional): The initial guess for the solution. Defaults to None.
                resolution (int, optional): The number of iterations for convergence. Defaults to 20.
                decimal (bool, optional): Whether to use decimal precision for computation. Defaults to False.

            Returns:
                Vector: The solution vector.

            Raises:
                ArgTypeError: If initial is provided and not a Vector object.
                DimensionError: If the dimensions of b do not match the number of rows in the matrix.
                RangeError: If resolution is less than 1.
        """
        if initial is not None:
            if not isinstance(initial, Vector):
                raise ArgTypeError("Must be a vector.")

        N = self.shape[0]
        if b.shape[0] != N:
            raise DimensionError(0)

        if resolution < 1:
            raise RangeError()

        if initial is None:
            initial = Vector(*[b[i] / self.values[i][i] for i in range(N)])

        for_l = []
        for_u = []
        M = len(self.values[0])
        for k in range(N):
            for_l.append(Vector.zero(N, decimal))
            for_u.append(Vector.zero(N, decimal))
            for l in range(M):
                if l <= k:
                    for_l[-1].values[l] = self.values[k][l]
                else:
                    for_u[-1].values[l] = self.values[k][l]

        L_inverse = Matrix(*for_l).inverse(resolution=resolution, decimal=decimal)
        U = Matrix(*for_u)
        L_i_b = L_inverse * b
        L_i_U = L_inverse * U

        for k in range(resolution):
            initial = L_i_b - L_i_U * initial

        return initial

    def least_squares(self, b, method: str = "iterative", resolution: int = 10, lowlimit=0.0000000001, highlimit=100000, decimal: bool = False):
        """
            Computes the least squares solution of the system of linear equations.

            Args:
                b (Vector): The vector of constant terms in the equations.
                method (str, optional): The method for computing the inverse. Defaults to "iterative".
                resolution (int, optional): The resolution for iterative methods. Defaults to 10.
                lowlimit (float, optional): The lower limit for gauss methods. Defaults to 0.0000000001.
                highlimit (float, optional): The upper limit for gauss methods. Defaults to 100000.
                decimal (bool, optional): Whether to use decimal precision for computation. Defaults to False.

            Returns:
                Vector: The least squares solution vector.

            Raises:
                ArgTypeError: If b is not a Vector object.
                DimensionError: If the dimensions of the matrix and b do not match.
        """
        if not isinstance(b, Vector):
            raise ArgTypeError("Must be a vector.")

        if self.shape[0] != b.shape[0]:
            raise DimensionError(0)

        t = self.transpose()
        temp = (t * self).inverse(method=method, resolution=resolution, lowlimit=lowlimit, highlimit=highlimit, decimal=decimal)
        return temp * (t * b)

    def jacobi_solve(self, b, resolution: int = 15):
        """
            Solves a system of linear equations using the Jacobi iterative method.

            Args:
                b (Vector): The vector of constant terms in the equations.
                resolution (int, optional): The number of iterations for convergence. Defaults to 15.

            Returns:
                Vector: The solution vector.

            Raises:
                ArgTypeError: If b is not a Vector object.
                DimensionError: If the dimensions of the matrix and b do not match.
                RangeError: If resolution is less than 1.
        """
        if not isinstance(b, Vector):
            raise ArgTypeError("Must be a vector.")
        if self.shape[0] != b.shape[0]:
            raise DimensionError(0)
        if resolution < 1:
            raise RangeError()

        D = self.get_diagonal().values
        N = self.shape[0]

        D_inverse = Matrix([[0 if i != k else 1 / D[k][i] for i in range(N)] for k in range(N)])

        T = -D_inverse * (self - D)
        C = D_inverse * b
        del D, D_inverse

        x = Vector([1 for k in range(N)])
        for i in range(resolution):
            x = T * x + C
        return x

    def toInt(self):
        """
            Converts the matrix to an integer matrix.

            Returns:
                Matrix: The integer matrix.
        """
        return Matrix([[int(item) for item in row] for row in self.values])

    def toFloat(self):
        """
            Converts the matrix to a float matrix.

            Returns:
                Matrix: The float matrix.
        """
        return Matrix([[float(item) for item in row] for row in self.values])

    def toBool(self):
        """
            Converts the matrix to a boolean matrix.

            Returns:
                Matrix: The boolean matrix.
        """
        return Matrix([[bool(item) for item in row] for row in self.values])

    def toDecimal(self):
        """
            Converts the matrix to a decimal matrix.

            Returns:
                Matrix: The decimal matrix.
        """
        return Matrix([[Decimal(item) for item in row] for row in self.values])

    def toVariable(self):
        """
            Converts the matrix to a "Variable" matrix.

            Returns:
                Matrix: The "Variable" matrix.
        """
        return Matrix([[Variable(item) for item in row] for row in self.values])

    def map(self, f):
        """
            Applies a function element-wise to the matrix.

            Args:
                f (Callable): The function to apply.

            Returns:
                Matrix: The matrix with the function applied.
        """
        if not isinstance(f, Callable):
            raise ArgTypeError("f must be a callable.")
        return Matrix([[f(item) for item in row] for row in self.values])

    def filter(self, f):
        """
            Filters the elements of the matrix based on a condition.

            Args:
                f (Callable): The filter condition.

            Returns:
                Matrix: The filtered matrix.

            Notes:
                This method can easily raise a DimensionError. Ensure that
                each row generated after the filtration has the same amount
                of elements.
        """
        # This can easily raise a DimensionError
        if not isinstance(f, Callable):
            raise ArgTypeError("f must be a callable.")
        vlist = []
        for row in self.values:
            temp = []
            for item in row:
                if f(item):
                    temp.append(item)
            vlist.append(temp)
        return Matrix(vlist)

    def submatrix(self, a, b, c, d):
        """
            Extracts a submatrix from the original matrix.

            Args:
                a (int): The starting row index.
                b (int): The ending row index.
                c (int): The starting column index.
                d (int): The ending column index.

            Returns:
                Matrix: The submatrix.

            Raises:
                ArgTypeError: If a, b, c, or d are not integers.
        """
        if not (isinstance(a, int) or isinstance(b, int) or isinstance(c, int) or isinstance(d, int)):
            raise ArgTypeError("Must be an integer.")

        return Matrix([row[c:d] for row in self.values[a:b]])

    def avg(self):
        """
            Computes the average of all elements in the matrix.

            Returns:
                float: The average value.
        """
        sum = 0
        for k in self.values:
            for l in k:
                sum += l
        return sum / (self.shape[0] * self.shape[1])

    def convolve(self, kernel, strafe: Union[list, tuple] = (1, 1)):
        if not isinstance(kernel, Matrix):
            raise ArgTypeError("Must be a matrix.")
        if len(strafe) != 2:
            raise DimensionError(0)

        y_change = kernel.shape[0]
        x_change = kernel.shape[1]
        y_count = self.shape[0] - y_change + strafe[0]
        x_count = self.shape[1] - x_change + strafe[1]

        return Matrix([[self.submatrix(k, k + y_change, l, l + x_change).frobenius_product(kernel) for l in range(x_count)] for k in range(y_count)])

    def dump(self):
        """
            Sets all data in self to 0.

            Notes:
                  Omits the possibility that the data could be of type Decimal.

        """
        M = self.shape[1]
        for k in range(self.shape[0]):
            for l in range(M):
                self.values[k][l] = 0

def maximum(dataset: Union[tuple, list, Vector, Matrix]):
    """
        Finds the maximum value in the given dataset.

        Args:
            dataset: The dataset to search for the maximum value.

        Returns:
            Any: The maximum value found in the dataset.

        Raises:
            ArgTypeError: If the dataset type is not supported.
    """
    maxima = Infinity(False)
    if isinstance(dataset, Union[tuple, list]):
        for data in dataset:
            if data > maxima:
                maxima = data
        return maxima  # Then you can get the index of it manually
    if isinstance(dataset, Vector):
        for data in dataset.values:
            if data > maxima:
                maxima = data
        return maxima
    if isinstance(dataset, Matrix):
        for k in dataset.values:
            for l in k:
                if l > maxima:
                    maxima = l
        return maxima
    raise ArgTypeError()

def minimum(dataset: Union[tuple, list, Vector, Matrix]):
    """
        Finds the minimum value in the given dataset.

        Args:
            dataset: The dataset to search for the minimum value.

        Returns:
            Any: The minimum value found in the dataset.

        Raises:
            ArgTypeError: If the dataset type is not supported.
        """
    minima = Infinity(True)
    if isinstance(dataset, Union[tuple, list]):
        for data in dataset:
            if data < minima:
                minima = data
        return minima  # Then you can get the index of it manually
    if isinstance(dataset, Vector):
        for data in dataset.values:
            if data < minima:
                minima = data
        return minima
    if isinstance(dataset, Matrix):
        for k in dataset.values:
            for l in k:
                if l < minima:
                    minima = l
        return minima
    raise ArgTypeError()
