from .functions import *
import random

class Vector:
    """
        Vector class representing mathematical vectors and supporting various operations.

        Attributes:
            values (list): List containing the elements of the vector.
            dimension (int): Dimensionality of the vector.
    """
    def __init__(self, *args):
        """
                Initializes a Vector object with the given arguments.

                Args:
                    *args: Variable number of arguments representing the elements of the vector.

                Raises:
                    ArgTypeError: If any argument is not numeric or boolean.
        """
        for k in args:
            if not isinstance(k, Union[int, float, Decimal, Infinity, Undefined, Complex]):
                raise ArgTypeError("Arguments must be numeric or boolean.")
        self.dimension = len(args)
        self.values = [_ for _ in args]

    def __str__(self):
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

    def __len__(self):
        """
        Returns the number of elements in the vector.

        Returns:
            int: The number of elements in the vector.
        """
        return len(self.values)

    def __add__(self, arg):
        if isinstance(arg, Union[int, float, Decimal, Infinity, Undefined, Complex]):
            return Vector(*[self.values[k] + arg for k in range(0, self.dimension)])
        if not isinstance(arg, Vector):
            raise ArgTypeError("Must be a numerical value.")
        if not (self.dimension == arg.dimension):
            raise DimensionError(0)
        return Vector(*[self.values[k] + arg.values[k] for k in range(0, self.dimension)])

    def __radd__(self, arg):
        if isinstance(arg, Union[int, float, Decimal, Infinity, Undefined, Complex]):
            return Vector(*[self.values[k] + arg for k in range(0, self.dimension)])
        raise ArgTypeError("Must be a numerical value.")

    def __sub__(self, arg):
        if isinstance(arg, Union[int, float, Decimal, Infinity, Undefined, Complex]):
            return Vector(*[self.values[k] - arg for k in range(0, self.dimension)])
        if not isinstance(arg, Vector):
            raise ArgTypeError("Must be a numerical value.")
        if not (self.dimension == arg.dimension):
            raise DimensionError(0)
        return Vector(*[self.values[k] - arg.values[k] for k in range(0, self.dimension)])

    def __rsub__(self, arg):
        if isinstance(arg, Union[int, float, Decimal, Infinity, Undefined, Complex]):
            return -Vector(*[self.values[k] - arg for k in range(0, self.dimension)])
        raise ArgTypeError("Must be a numerical value.")

    def dot(self, arg):
        """
            Computes the dot product of the vector with another vector.

            Args:
                arg (Vector): The vector with which the dot product is computed.

            Returns:
                Union[int, float, Decimal, Infinity, Undefined, Complex]: The dot product of the two vectors.

            Raises:
                ArgTypeError: If the argument `arg` is not a vector.
                DimensionError: If the dimensions of the two vectors are not the same.
        """
        if not isinstance(arg, Vector):
            raise ArgTypeError("Must be a vector.")
        if not (self.dimension == arg.dimension):
            raise DimensionError(0)
        mul = [self.values[k] * arg.values[k] for k in range(0, self.dimension)]
        sum = 0
        for k in mul:
            sum += k
        return sum

    def __mul__(self, arg):
        if not isinstance(arg, Union[int, float, Decimal, Infinity, Undefined, Complex]):
            raise ArgTypeError("Must be a numerical value.")
        return Vector(*[self.values[k] * arg for k in range(0, self.dimension)])

    def __rmul__(self, arg):
        if not isinstance(arg, Union[int, float, Decimal, Infinity, Undefined, Complex]):
            raise ArgTypeError("Must be a numerical value.")
        return Vector(*[self.values[k] * arg for k in range(0, self.dimension)])

    def __truediv__(self, arg):
        if not isinstance(arg, Union[int, float, Decimal, Infinity, Undefined, Complex]):
            raise ArgTypeError("Must be a numerical value.")
        return Vector(*[self.values[k] / arg for k in range(0, self.dimension)])


    def __floordiv__(self, arg):
        if not isinstance(arg, Union[int, float, Decimal, Infinity, Undefined, Complex]):
            raise ArgTypeError("Must be a numerical value.")
        return Vector(*[self.values[k] // arg for k in range(0, self.dimension)])

    def __iadd__(self, arg):
        if isinstance(arg, Union[int, float, Decimal, Infinity, Undefined, Complex]):
            return Vector(*[self.values[k] + arg for k in range(0, self.dimension)])
        if not isinstance(arg, Vector):
            raise ArgTypeError("Must be a numerical value.")
        if not (self.dimension == arg.dimension):
            raise DimensionError(0)
        return Vector(*[self.values[k] + arg.values[k] for k in range(0, self.dimension)])

    def __isub__(self, arg):
        if isinstance(arg, Union[int, float, Decimal, Infinity, Undefined, Complex]):
            return Vector(*[self.values[k] - arg for k in range(0, self.dimension)])
        if not isinstance(arg, Vector):
            raise ArgTypeError("Must be a numerical value.")
        if not (self.dimension == arg.dimension):
            raise DimensionError(0)
        return Vector(*[self.values[k] - arg.values[k] for k in range(0, self.dimension)])

    def __gt__(self, arg):
        if isinstance(arg, Union[int, float, Decimal, Infinity, Undefined, Complex]):
            sum = 0
            for k in self.values:
                sum += k * k
            if sum > arg * arg:
                return True
            return False
        if not isinstance(arg, Vector):
            raise ArgTypeError("Must be a numerical value.")
        if not (self.dimension == arg.dimension):
            raise DimensionError(0)
        sum = 0
        for k in self.values:
            sum += k*k
        for k in arg.values:
            sum -= k*k
        if sum > 0:
            return True
        return False

    def __ge__(self, arg):
        if isinstance(arg, Union[int, float, Decimal, Infinity, Undefined, Complex]):
            sum = 0
            for k in self.values:
                sum += k * k
            if sum >= arg * arg:
                return True
            return False
        if not isinstance(arg, Vector):
            raise ArgTypeError("Must be a numerical value.")
        if not (self.dimension == arg.dimension):
            raise DimensionError(0)
        sum = 0
        for k in self.values:
            sum += k*k
        for k in arg.values:
            sum -= k*k
        if sum >= 0:
            return True
        return False

    def __lt__(self, arg):
        if isinstance(arg, Union[int, float, Decimal, Infinity, Undefined, Complex]):
            sum = 0
            for k in self.values:
                sum += k * k
            if sum > arg * arg:
                return True
            return False
        if not isinstance(arg, Vector):
            raise ArgTypeError("Must be a numerical value.")
        if not (self.dimension == arg.dimension):
            raise DimensionError(0)
        sum = 0
        for k in self.values:
            sum += k*k
        for k in arg.values:
            sum -= k*k
        if sum < 0:
            return True
        return False

    def __le__(self, arg):
        if isinstance(arg, Union[int, float, Decimal, Infinity, Undefined, Complex]):
            sum = 0
            for k in self.values:
                sum += k * k
            if sum <= arg * arg:
                return True
            return False
        if not isinstance(arg, Vector):
            raise ArgTypeError("Must be a numerical value.")
        if not (self.dimension == arg.dimension):
            raise DimensionError(0)
        sum = 0
        for k in self.values:
            sum += k*k
        for k in arg.values:
            sum -= k*k
        if sum <= 0:
            return True
        return False

    def __eq__(self, arg):
        if isinstance(arg, Union[int, float, Decimal, Infinity, Undefined, Complex]):
            for k in self.values:
                if not (k == arg):
                    return False
            return True
        if not isinstance(arg, Vector):
            raise ArgTypeError("Must be a numerical value.")
        if not (self.dimension == arg.dimension):
            raise DimensionError(0)
        factor = True
        for k in self.values:
            for l in arg.values:
                factor = factor and (k == l)
        return factor

    def __neg__(self):
        return Vector(*[-k for k in self.values])

    def __pos__(self):
        return Vector(*[k for k in self.values])

    def __and__(self, arg):
        if not isinstance(arg, Vector):
            raise ArgTypeError("Must be a vector.")
        if not (self.dimension == arg.dimension):
            raise DimensionError(0)
        return Vector(*[(self.values[k] and arg.values[k]) for k in range(0, self.dimension)])

    def __iand__(self, arg):
        if not isinstance(arg, Vector):
            raise ArgTypeError("Must be a vector.")
        if not (self.dimension == arg.dimension):
            raise DimensionError(0)
        return Vector(*[(self.values[k] and arg.values[k]) for k in range(0, self.dimension)])

    def __or__(self, arg):
        if not isinstance(arg, Vector):
            raise ArgTypeError("Must be a vector.")
        if not (self.dimension == arg.dimension):
            raise DimensionError(0)
        return Vector(*[(self.values[k] or arg.values[k]) for k in range(0, self.dimension)])

    def __ior__(self, arg):
        if not isinstance(arg, Vector):
            raise ArgTypeError("Must be a vector.")
        if not (self.dimension == arg.dimension):
            raise DimensionError(0)
        return Vector(*[(self.values[k] or arg.values[k]) for k in range(0, self.dimension)])

    def __xor__(self, arg):
        if not isinstance(arg, Vector):
            raise ArgTypeError("Must be a vector.")
        if not (self.dimension == arg.dimension):
            raise DimensionError(0)
        return Vector(*[(self.values[k] ^ arg.values[k]) for k in range(0, self.dimension)])

    def __ixor__(self, arg):
        if not isinstance(arg, Vector):
            raise ArgTypeError("Must be a vector.")
        if not (self.dimension == arg.dimension):
            raise DimensionError(0)
        return Vector(*[(self.values[k] ^ arg.values[k]) for k in range(0, self.dimension)])

    def __invert__(self):
        return Vector(*[int(not self.values[k]) for k in range(0, self.dimension)])

    def append(self, arg):
        """
            Appends the elements of the given argument to the end of the vector.

            Args:
                arg (Union[int, float, Decimal, Infinity, Undefined, Complex, Vector, list, tuple]):
                    The element or iterable containing elements to append to the vector.

            Raises:
                ArgTypeError: If the argument is not a numeric value, boolean, vector, list, or tuple.
        """
        if isinstance(arg, Union[int, float, Decimal, Infinity, Undefined, Complex]):
            self.values.append(arg)
            self.dimension += 1
            return
        if isinstance(arg, Vector):
            for k in arg.values:
                self.values.append(k)
            self.dimension += arg.dimension
        elif isinstance(arg, Union[list, tuple]):
            for k in arg:
                self.values.append(k)
            self.dimension += len(arg)
        raise ArgTypeError("Must be a numerical value.")

    def copy(self):
        """
            Returns a copy of the vector.

            Returns:
                Vector: A copy of the vector with the same elements and dimensionality.
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
            self.values[ord]
        except IndexError:
            raise RangeError()
        popped = self.values.pop(ord)
        self.dimension -= 1
        return popped

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
            sum += k*k
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
                DimensionError: If the dimensions of `self` and `arg` are not equal.
        """
        if not isinstance(arg, Vector):
            raise ArgTypeError("Must be a vector.")
        if not (self.dimension == arg.dimension):
            raise DimensionError(0)
        if not self.dimension:
            return 0
        dot = self.dot(arg)
        sum = 0
        for k in arg.values:
            sum += k*k
        try:
            dot = dot/sum
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
            temp = [k/l for k in self.values]
        else:
            temp = [Infinity()] * self.dimension
        return Vector(*temp)

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
                AmountError: If the number of vectors provided does not match their dimensions.
        """
        v_list = []
        for k in args:
            if not isinstance(k, Vector):
                raise ArgTypeError("Must be a vector.")
            if not (k.dimension == (len(args))):
                raise AmountError
            v_list.append(k)
        for k in range(1, len(v_list)):
            temp = v_list[k]
            for l in range(0, k):
                temp -= v_list[k].proj(v_list[l])
            v_list[k] = temp.unit()
        v_list[0] = v_list[0].unit()
        return v_list

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
                AmountError: If the number of vectors provided does not match their dimensions.
        """
        v_list = Vector.spanify(*args)
        for k in range(0, len(v_list)):
            for l in range(0, len(v_list)):
                if not v_list[k].dot(v_list[l]) < 0.0000000001 and not k == l:
                    return False
        return True

    def randVint(dim: int, a: int, b: int, decimal: bool = False):
        """
            Generates a random integer vector with specified dimensions and range.

            The `randVint` method creates a random integer vector with the specified dimensions and
            values within the given range [a, b].

            Args:
                dim (int): The dimensionality of the random vector.
                a (int): The lower bound of the range for random integer generation.
                b (int): The upper bound of the range for random integer generation.
                decimal (bool, optional): If True, generated values will be of Decimal type. Defaults to False.

            Returns:
                Vector: A random integer vector with the specified dimensions and values within the range [a, b].

            Raises:
                ArgTypeError: If any input argument is not an integer.
                RangeError: If the dimension is not a positive integer.
        """
        if not (isinstance(dim, int) and isinstance(a, int) and isinstance(b, int)):
            raise ArgTypeError("Must be an integer.")
        if not (dim > 0):
            raise RangeError
        if decimal:
            return Vector(*[Decimal(random.randint(a, b)) for k in range(0, dim)])
        return Vector(*[random.randint(a, b) for k in range(0, dim)])

    def randVfloat(dim, a: float, b: float, decimal: bool = False):
        """
            Generates a random float vector with specified dimensions and range.

            The `randVfloat` method creates a random float vector with the specified dimensions and
            values within the given range [a, b].

            Args:
                dim (int): The dimensionality of the random vector.
                a (float): The lower bound of the range for random float generation.
                b (float): The upper bound of the range for random float generation.
                decimal (bool, optional): If True, generated values will be of Decimal type. Defaults to False.

            Returns:
                Vector: A random float vector with the specified dimensions and values within the range [a, b].

            Raises:
                ArgTypeError: If any input argument is not a numerical value.
                RangeError: If the dimension is not a positive integer.
        """
        if not (isinstance(dim, int) and
                (isinstance(a, Union[int, float, Decimal])) and
                (isinstance(b, Union[int, float, Decimal]))):
            raise ArgTypeError("Must be a numerical value.")
        if not (dim > 0):
            raise RangeError
        if decimal:
            return Vector(*[Decimal(random.uniform(a, b)) for k in range(0, dim)])
        return Vector(*[random.uniform(a, b) for k in range(0, dim)])

    def randVbool(dim, decimal: bool = False):
        """
            Generates a random boolean vector with specified dimensions.

            The `randVbool` method creates a random boolean vector with the specified dimensions.

            Args:
                dim (int): The dimensionality of the random vector.
                decimal (bool, optional): If True, generated values will be of Decimal type. Defaults to False.

            Returns:
                Vector: A random boolean vector with the specified dimensions.

            Raises:
                ArgTypeError: If the dimension is not an integer.
                RangeError: If the dimension is not a positive integer.
        """
        if not isinstance(dim, int): raise ArgTypeError("Must be an integer.")
        if not (dim > 0): raise RangeError
        if decimal:
            return Vector(*[Decimal(random.randrange(0, 2)) for k in range(0, dim)])
        return Vector(*[random.randrange(0, 2) for k in range(0, dim)])

    def randVgauss(dim, mu=0, sigma=0, decimal: bool = False):
        """
            Generates a random vector of Gaussian (normal) distribution with specified dimensions.

            The `randVgauss` method creates a random vector of Gaussian (normal) distribution with the specified dimensions,
            mean, and standard deviation.

            Args:
                dim (int): The dimensionality of the random vector.
                mu (Union[int, float, Decimal]): The mean (average) value of the distribution.
                sigma (Union[int, float, Decimal]): The standard deviation of the distribution.
                decimal (bool, optional): If True, generated values will be of Decimal type. Defaults to False.

            Returns:
                Vector: A random vector of Gaussian (normal) distribution with the specified dimensions.

            Raises:
                ArgTypeError: If the dimension, mean, or standard deviation is not a numerical value.
                RangeError: If the dimension is not a positive integer.
        """
        if not isinstance(dim, int): raise ArgTypeError("Must be an integer.")
        if not ((isinstance(mu, Union[int, float, Decimal])) and
                (isinstance(sigma, Union[int, float, Decimal]))):
            raise ArgTypeError("Must be a numerical value.")
        if not (dim > 0): raise RangeError
        if decimal:
            return Vector(*[Decimal(random.gauss(mu, sigma)) for k in range(dim)])
        return Vector(*[random.gauss(mu, sigma) for k in range(dim)])

    def determinant(*args):
        """
            Calculates the determinant of a square matrix represented by the given vectors.

            The `determinant` method calculates the determinant of a square matrix represented by the provided vectors.
            The method supports matrices of dimensions 2x2 and higher.

            Args:
                *args (Vector): One or more vectors representing the rows or columns of the square matrix.

            Returns:
                Union[int, float, Decimal]: The determinant of the square matrix.

            Raises:
                ArgTypeError: If any argument is not a vector or if the dimensions of the vectors are inconsistent.
                DimensionError: If the dimensions of the vectors do not form a square matrix.
                AmountError: If the number of vectors does not match the dimension of the square matrix.
        """
        for k in args:
            if not isinstance(k, Vector): raise ArgTypeError("Must be a vector.")
            if not (args[0].dimension == k.dimension): raise DimensionError(0)
        if not (len(args) == args[0].dimension): raise AmountError

        if len(args) == 2 and args[0].dimension == 2:
            return (args[0].values[0] * args[1].values[1]) - (args[0].values[1] * args[1].values[0])

        result = 0
        for k in range(0, args[0].dimension):
            vector_list = list()
            for a in range(1, args[0].dimension):
                temp = list()
                for b in range(0, args[0].dimension):
                    if not b == k:
                        temp.append(args[a].values[b])
                vector_list.append(Vector(*temp))
            result += Vector.determinant(*vector_list) * pow(-1, k) * args[0].values[k]
        return result

    def cross(*args):
        """
            Calculates the cross product of vectors.

            The cross product is a binary operation on two vectors in three-dimensional space. It results in a vector that
            is perpendicular to both input vectors. For vectors in higher dimensions, the cross product is generalized as the
            determinant of a matrix formed by the input vectors and the standard basis vectors.

            Args:
                *args (Vector): Vectors to compute the cross product.

            Returns:
                Vector: The vector representing the cross product of the input vectors.

            Raises:
                ArgTypeError: If any argument is not a vector or if the dimensions of the vectors are inconsistent.
                DimensionError: If the dimensions of the vectors do not match the requirements for calculating the cross product.
        """
        for k in args:
            if not isinstance(k, Vector): raise ArgTypeError("Must be a vector.")
            if not (args[0].dimension == k.dimension): raise DimensionError(0)

        if len(args) == 2 and args[0].dimension == 2:
            return args[0].values[0] * args[1].values[1] - args[0].values[1] * args[1].values[0]
        if not (len(args) == args[0].dimension - 1): raise AmountError

        end_list = list()
        for k in range(0, args[0].dimension):
            vector_list = list()
            for a in range(0, args[0].dimension-1):
                temp = list()
                for b in range(0, args[0].dimension):
                    if not b == k:
                        temp.append(args[a].values[b])
                vector_list.append(Vector(*temp))
            end_list.append((Vector.determinant(*vector_list)) * pow(-1, k))
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
                DimensionError: If the dimensions of v and w are not compatible for computing the outer product.
        """
        if not (isinstance(v, Vector) and isinstance(w, Vector)): raise ArgTypeError("Must be a vector.")
        if v.dimension != w.dimension: raise DimensionError(0)

        v_list = []
        for i in range(v.dimension):
            temp = []
            for j in range(v.dimension):
                temp.append(v.values[i] * w.values[j])
            v_list.append(Vector(*temp))

        return Matrix(*v_list)

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

    def zero(dim: int, decimal: bool = False):
        """
            Generates a zero vector of a specified dimension.

            Args:
                dim (int): The dimensionality of the zero vector to be created.
                decimal (bool, optional): A boolean flag indicating whether the elements of the vector should be Decimal numbers. Defaults to False.

            Returns:
                Vector: A zero vector of the specified dimension.

            Raises:
                RangeError: If the dimension specified is negative.
        """
        # We use the RangeError because dimension can be 0.
        if dim < 0: raise RangeError()
        if decimal:
            return Vector(*[Decimal(0) for k in range(dim)])
        else:
            return Vector(*[0 for k in range(dim)])

    def one(dim: int, decimal: bool = False):
        """
            Generates a vector with all elements set to one.

            Args:
                dim (int): The dimensionality of the vector to be created.
                decimal (bool, optional): A boolean flag indicating whether the elements of the vector should be Decimal numbers. Defaults to False.

            Returns:
                Vector: A vector with all elements set to one.

            Raises:
                RangeError: If the dimension specified is negative.
        """
        if dim < 0: raise RangeError()
        if decimal:
            return Vector(*[Decimal(1) for k in range(dim)])
        else:
            return Vector(*[1 for k in range(dim)])

    def reshape(self, m: int, n: int):
        """
            Reshapes the vector into a matrix with the specified dimensions.

            Args:
                m (int): The number of rows in the resulting matrix.
                n (int): The number of columns in the resulting matrix.

            Returns:
                Matrix: A matrix with dimensions m x n reshaped from the vector.

            Raises:
                RangeError: If the product of m and n does not equal the dimension of the vector.
        """
        if not m * n == self.dimension: raise RangeError()
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
                angle (Union[int, float]): The angle of rotation in degrees.
                resolution (int, optional): The number of points in the rotation. Defaults to 15.

            Returns:
                Vector: The rotated vector.

            Example:
                If `v` is a vector, `v.rotate(0, 1, math.pi / 2)` rotates the vector `v` by
                90 degrees in the plane defined by the first two components.
        """
        return Matrix.givens(self.dimension, i, j, angle, resolution) * self

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
                leak (Union[int, float, Decimal, Infinity, Undefined], optional): The slope of the negative part
                    of the activation function for values less than zero. Defaults to 0.
                cutoff (Union[int, float, Decimal, Infinity, Undefined], optional): Values less than this cutoff
                    will be set to zero. Defaults to 0.

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
                a (Union[int, float, Decimal, Infinity, Undefined], optional): Scaling factor for the input.
                    Defaults to 1.
                cutoff (Union[int, float, Decimal, Infinity, Undefined], optional): Values greater than this cutoff
                    will be saturated to 1, and values less than the negative of this cutoff will be saturated to 0.
                    If `None`, no saturation is applied. Defaults to None.

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
        if not isinstance(f, Callable): raise ArgTypeError("f must be a callable.")
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
        if self.dimension == 0:
            return Undefined()
        sum = 0
        for k in self.values:
            sum += k
        return sum / self.dimension

class Matrix:
    """
        Matrix class representing mathematical matrices and supporting various operations.

        Attributes:
            values (list): List containing the elements of the matrix.
            dimension (str): Dimensionality of the matrix.
    """
    def __init__(self, *args):
        """
            Initializes a Matrix object.

            Args:
                *args (Vector): Variable number of Vector objects to form the matrix.
        """
        for k in args:
            if not isinstance(k, Vector):
                raise ArgTypeError("Must be a vector.")
            if not (args[0].dimension == k.dimension):
                raise DimensionError(0)
        self.values = [k.values for k in args]
        if args:
            self.dimension = f"{len(args[0])}x{len(args)}"
        else:
            self.dimension = "0x0"

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
        vals = self.dimension.split("x")
        return [int(vals[0]), int(vals[1])]

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

    def __add__(self, arg):
        v = []
        if isinstance(arg, Union[int, float, Decimal, Infinity, Undefined, Complex]):
            for k in self.values:
                v.append(Vector(*[l + arg for l in k]))
            return Matrix(*v)
        if not isinstance(arg, Matrix):
            raise ArgTypeError("Must be a numerical value.")
        if not (self.dimension == arg.dimension):
            raise DimensionError(0)
        for k in range(0, len(self.values)):
            m = []
            for l in range(0, len(self.values[0])):
                m.append(self.values[k][l] + arg.values[k][l])
            v.append(m)
        return Matrix(*[Vector(*k) for k in v])

    def __radd__(self, arg):
        v = []
        if isinstance(arg, Union[int, float, Decimal, Infinity, Undefined, Complex]):
            for k in self.values:
                v.append(Vector(*[l + arg for l in k]))
            return Matrix(*v)
        raise ArgTypeError("Must be a numerical value.")

    def __iadd__(self, arg):
        v = []
        if isinstance(arg, Union[int, float, Decimal, Infinity, Undefined, Complex]):
            for k in self.values:
                v.append(Vector(*[l + arg for l in k]))
            return Matrix(*v)
        if not isinstance(arg, Matrix):
            raise ArgTypeError("Must be a numerical value.")
        if not (self.dimension == arg.dimension):
            raise DimensionError(0)
        for k in range(0, len(self.values)):
            m = []
            for l in range(0, len(self.values[0])):
                m.append(self.values[k][l] + arg.values[k][l])
            v.append(m)
        return Matrix(*[Vector(*k) for k in v])

    def __sub__(self, arg):
        v = []
        if isinstance(arg, Union[int, float, Decimal, Infinity, Undefined, Complex]):
            for k in self.values:
                v.append(Vector(*[l - arg for l in k]))
            return Matrix(*v)
        if not isinstance(arg, Matrix):
            raise ArgTypeError("Must be a numerical value.")
        if not (self.dimension == arg.dimension):
            raise DimensionError(0)
        for k in range(0, len(self.values)):
            m = []
            for l in range(0, len(self.values[0])):
                m.append(self.values[k][l] - arg.values[k][l])
            v.append(m)
        return Matrix(*[Vector(*k) for k in v])

    def __rsub__(self, arg):
        v = []
        if isinstance(arg, Union[int, float, Decimal, Infinity, Undefined, Complex]):
            for k in self.values:
                v.append(Vector(*[l - arg for l in k]))
            return -Matrix(*v)
        raise ArgTypeError("Must be a numerical value.")

    def __isub__(self, arg):
        v = []
        if isinstance(arg, Union[int, float, Decimal, Infinity, Undefined, Complex]):
            for k in self.values:
                v.append(Vector(*[l - arg for l in k]))
            return Matrix(*v)
        if not isinstance(arg, Matrix):
            raise ArgTypeError("Must be a numerical value.")
        if not (self.dimension == arg.dimension):
            raise DimensionError(0)
        for k in range(0, len(self.values)):
            m = []
            for l in range(0, len(self.values[0])):
                m.append(self.values[k][l] - arg.values[k][l])
            v.append(m)
        return Matrix(*[Vector(*k) for k in v])

    def __mul__(self, arg):
        v = []
        if isinstance(arg, Union[int, float, Decimal, Infinity, Undefined, Complex]):
            for k in self.values:
                v.append(Vector(*[l * arg for l in k]))
            return Matrix(*v)
        if isinstance(arg, Vector):
            if not (self.dimension.split("x")[0] == str(arg.dimension)):
                raise DimensionError(0)
            for k in range(0, len(self.values)):
                sum = 0
                for l in range(0, len(arg.values)):
                    sum += self.values[k][l] * arg.values[l]
                v.append(sum)
            return Vector(*v)

        if not isinstance(arg, Matrix):
            raise ArgTypeError("Must be a numerical value.")
        if not (self.dimension.split("x")[1] == arg.dimension.split("x")[0]):
            raise DimensionError(0)
        for k in range(0, len(self.values)):
            n = []
            for l in range(0, len(arg.values[0])):
                sum = 0
                for m in range(0, len(arg.values)):
                    sum += self.values[k][m] * arg.values[m][l]
                n.append(sum)
            v.append(n)
        return Matrix(*[Vector(*k) for k in v])

    def __rmul__(self, arg):
        v = []
        if isinstance(arg, Union[int, float, Decimal, Infinity, Undefined, Complex]):
            for k in self.values:
                v.append(Vector(*[l * arg for l in k]))
            return Matrix(*v)
        raise ArgTypeError("Must be a numerical value.")

    def __neg__(self):
        return Matrix(*[Vector(*[-l for l in k]) for k in self.values])

    def __truediv__(self, arg):
        v = []
        if not isinstance(arg, Union[int, float, Decimal, Infinity, Undefined, Complex]):
            raise ArgTypeError("Must be a numerical value.")
        for k in self.values:
            v.append(Vector(*[l / arg for l in k]))
        return Matrix(*v)

    def __floordiv__(self, arg):
        v = []
        if not isinstance(arg, Union[int, float, Decimal, Infinity, Undefined, Complex]):
            raise ArgTypeError("Must be a numerical value.")
        for k in self.values:
            v.append(Vector(*[l // arg for l in k]))
        return Matrix(*v)

    def __pow__(self, p, decimal: bool = False):
        temp = Matrix.identity(len(self.values), decimal)
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
        if self.dimension == "1x1":
            return self.values[0][0]
        if choice == "analytic":
            return Vector.determinant(*[Vector(*k) for k in self.values])
        elif choice == "echelon":
            a = self.echelon()
            sum = 1
            for k in range(0, len(a.values)):
                sum *= a.values[k][k]
            return sum

    def __or__(self, arg):
        v = []
        if not isinstance(arg, Matrix):
            raise ArgTypeError("Must be a numerical value.")
        if not (self.dimension == arg.dimension):
            raise DimensionError(0)
        for k in range(0, len(self.values)):
            m = []
            for l in range(0, len(self.values[k])):
                m.append(self.values[k][l] or arg.values[k][l])
            v.append(Vector(*m))
        return Matrix(*v)

    def __ior__(self, arg):
        v = []
        if not isinstance(arg, Matrix):
            raise ArgTypeError("Must be a matrix.")
        if not (self.dimension == arg.dimension):
            raise DimensionError(0)
        for k in range(0, len(self.values)):
            m = []
            for l in range(0, len(self.values[k])):
                m.append(self.values[k][l] or arg.values[k][l])
            v.append(Vector(*m))
        return Matrix(*v)

    def __and__(self, arg):
        v = []
        if not isinstance(arg, Matrix):
            raise ArgTypeError("Must be a matrix.")
        if not (self.dimension == arg.dimension):
            raise DimensionError(0)
        for k in range(0, len(self.values)):
            m = []
            for l in range(0, len(self.values[k])):
                m.append(self.values[k][l] and arg.values[k][l])
            v.append(Vector(*m))
        return Matrix(*v)

    def __iand__(self, arg):
        v = []
        if not isinstance(arg, Matrix):
            raise ArgTypeError("Must be a matrix.")
        if not (self.dimension == arg.dimension):
            raise DimensionError(0)
        for k in range(0, len(self.values)):
            m = []
            for l in range(0, len(self.values[k])):
                m.append(self.values[k][l] and arg.values[k][l])
            v.append(Vector(*m))
        return Matrix(*v)

    def __xor__(self, arg):
        v = []
        if not isinstance(arg, Matrix):
            raise ArgTypeError("Must be a matrix.")
        if not (self.dimension == arg.dimension):
            raise DimensionError(0)
        for k in range(0, len(self.values)):
            m = []
            for l in range(0, len(self.values[k])):
                m.append(self.values[k][l] ^ arg.values[k][l])
            v.append(Vector(*m))
        return Matrix(*v)

    def __ixor__(self, arg):
        v = []
        if not isinstance(arg, Matrix):
            raise ArgTypeError("Must be a matrix.")
        if not (self.dimension == arg.dimension):
            raise DimensionError(0)
        for k in range(0, len(self.values)):
            m = []
            for l in range(0, len(self.values[k])):
                m.append(self.values[k][l] ^ arg.values[k][l])
            v.append(Vector(*m))
        return Matrix(*v)

    def __invert__(self):
        return Matrix(*[Vector(*[int(not l) for l in k]) for k in self.values])

    def __eq__(self, arg):
        if not isinstance(arg, Matrix):
            raise ArgTypeError("Must be a matrix.")
        if self.values == arg.values:
            return True
        return False

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
        if not len(self.values[0]) == arg.dimension:
            raise DimensionError(0)
        self.values.append(arg.values)
        temp = self.dimension.split("x")
        temp[0] = str(int(temp[0]) + 1)
        self.dimension = "x".join(temp)

    def copy(self):
        """
            Creates a copy of the matrix.

            Returns:
                Matrix: A new matrix object containing a copy of the original matrix.

            Notes:
                The copy method performs a deep copy of the matrix, including its values and dimension.
        """
        return Matrix(*[Vector(*k.copy()) for k in self.values])

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
        temp = self.dimension.split("x")
        temp[0] = str(int(temp[0]) - 1)
        self.dimension = "x".join(temp)
        return Vector(*popped)

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
        if not isinstance(arg, Matrix): raise ArgTypeError("Must be a matrix.")
        if self.dimension != arg.dimension: raise DimensionError(0)
        sum = 0
        for k in range(len(self.values)):
            for l in range(len(self.values[0])):
                sum += self.values[k][l] * arg.values[k][l]
        return sum

    def transpose(self):
        """
            Transposes the matrix by swapping its rows and columns.

            Returns:
                Matrix: The transposed matrix.
        """
        v = []
        for k in range(0, len(self.values[0])):
            m = []
            for l in range(0, len(self.values)):
                m.append(self.values[l][k])
            v.append(Vector(*m))
        return Matrix(*v)

    def conjugate(self):
        """
            Computes the conjugate of the matrix.

            Returns:
                Matrix: The conjugate of the matrix.

            Notes:
                The conjugate of a matrix involves taking the conjugate of each complex
                element in the matrix. For real numbers, the conjugate is the number itself.
        """
        for k in range(len(self.values)):
            for l in range(len(self.values[0])):
                if isinstance(self[k][l], Complex):
                    self.values[k][l] = self.values[k][l].conjugate()
        temp = [Vector(*k) for k in self.values]
        return Matrix(*temp)

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
        if not d:
            return Undefined()
        return Matrix(*[Vector(*[val / d for val in k]) for k in self.values])

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

    def norm(self, resolution: int = 10, decimal: bool = False):
        """
            Computes the norm of the matrix.

            Args:
                resolution (int, optional): The resolution for eigenvalue computation. Defaults to 10.
                decimal (bool, optional): Whether to use decimal precision for computation. Defaults to False.

            Returns:
                Union[int, float, Decimal]: The norm of the matrix.

            Raises:
                ArgTypeError: If the resolution is not an integer.
                RangeError: If the resolution is less than 1.

            Notes:
                The norm of a matrix is the square root of the maximum eigenvalue of the matrix's
                Hermitian conjugate matrix obtained by multiplying the matrix with its conjugate transpose.
                The resolution parameter determines the accuracy of the eigenvalue computation. The calculated
                norm is specifically the Euclidian norm.
        """
        temp = self.hconj() * self
        vals = temp.eigenvalue(resolution=resolution, decimal=decimal)
        return sqrt(maximum(vals))

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
                "gauss", "analytic", "iterative", and "neumann". The resolution parameter affects the accuracy
                of iterative or Neumann series methods. The lowlimit and highlimit parameters are used to control
                the convergence behavior of the gauss method. The inverse matrix is obviously only defined for
                square matrices.
        """
        if method not in ["gauss", "analytic", "iterative", "neumann"]: raise ArgTypeError()
        if resolution < 1: raise RangeError()
        if not ((isinstance(lowlimit, Union[int, float, Decimal]))
                and (isinstance(highlimit, Union[int, float, Decimal]))):
            raise ArgTypeError("Must be a numerical value.")
        if not (self.dimension.split("x")[0] == self.dimension.split("x")[1]):
            raise DimensionError(2)
        if self.dimension == "1x1":
            return 1 / self.values[0][0]

        if method == "analytic":
            det = Matrix.determinant(self)
            if not det:
                return
            end = list()
            for k in range(0, len(self.values)):
                temp = list()
                for l in range(0, len(self.values)):
                    sub = list()
                    for a in range(0, len(self.values)):
                        n = list()
                        for b in range(0, len(self.values)):
                            if (not k == a) and (not l == b):
                                n.append(self.values[a][b])
                        if len(n) > 0:
                            sub.append(Vector(*n))
                    temp.append(pow(-1, k + l) * Matrix.determinant(Matrix(*sub)))
                end.append(temp)
            return Matrix(*[Vector(*k) for k in end]).transpose() / det

        elif method == "gauss":
            if isinstance(self[0][0], Decimal):
                i = Matrix.identity(len(self.values))
            else:
                i = Matrix.identity(len(self.values), decimal)
            i_values = i.values.copy()
            v = self.values.copy()
            taken_list = []
            taken_list_i = []
            counter = 0

            for k in range(0, len(self.values)):
                for l in range(0, len(self.values[0])):
                    if not self.values[k][l] == 0 and l not in taken_list:
                        v[l] = self.values[k]
                        i_values[l] = i.values[k]
                        counter += 1
                        if not l == k and counter % 2 == 0:
                            v[l] = [-z for z in self.values[k]]
                            i_values[l] = [-z for z in i.values[k]]
                        else:
                            v[l] = self.values[k]
                            i_values[l] = i.values[k]
                        taken_list.append(l)
                        taken_list_i.append(l)
                        break
                    elif not self.values[k][l] == 0 and l in taken_list:
                        for m in range(l, len(self.values)):
                            if m not in taken_list:
                                v[m] = self.values[k]
                                i_values[m] = i.values[k]
                                counter += 1
                                if not m == k and counter % 2 == 0:
                                    v[m] = [-z for z in self.values[k]]
                                    i_values[m] = [-z for z in i.values[k]]



            for k in range(0, len(self.values[0])):
                if v[k][k] == 0:
                    continue
                for l in range(0, len(self.values)):
                    if l == k:
                        continue
                    try:
                        factor = (v[l][k]) / (v[k][k])
                        if abs(factor) < lowlimit or abs(factor) > highlimit:
                            factor = 0
                        factored_list = [v[l][m] - (factor * v[k][m]) for m in range(0, len(self.values[0]))]
                        factored_list_i = [i_values[l][m] - (factor * i_values[k][m]) for m in
                                           range(0, len(self.values[0]))]
                        v[l] = factored_list
                        i_values[l] = factored_list_i
                    except ZeroDivisionError:
                        continue

            v = v[::-1]
            iden_values = i_values.copy()
            iden_values = iden_values[::-1]

            for k in range(0, len(self.values[0])):
                if v[k][k] == 0:
                    continue
                for l in range(0, len(self.values)):
                    if l == k:
                        continue
                    try:
                        factor = (v[l][k]) / (v[k][k])
                        if abs(factor) < lowlimit or abs(factor) > highlimit:
                            factor = 0
                        factored_list = [v[l][m] - (factor * v[k][m]) for m in range(0, len(self.values[0]))]
                        factored_list_i = [iden_values[l][m] - (factor * iden_values[k][m]) for m in
                                           range(0, len(self.values[0]))]
                        v[l] = factored_list
                        iden_values[l] = factored_list_i
                    except ZeroDivisionError:
                        continue

            iden_values = iden_values[::-1].copy()
            v = v[::-1].copy()

            for k in range(0, len(self.values[0])):
                if v[k][k] == 0:
                    continue
                for l in range(0, len(self.values)):
                    if l == k:
                        continue
                    try:
                        factor = (v[l][k]) / (v[k][k])
                        if abs(factor) < lowlimit or abs(factor) > highlimit:
                            factor = 0
                        factored_list = [v[l][m] - (factor * v[k][m]) for m in range(0, len(self.values[0]))]
                        factored_list_i = [iden_values[l][m] - (factor * iden_values[k][m]) for m in
                                           range(0, len(self.values[0]))]
                        v[l] = factored_list
                        iden_values[l] = factored_list_i
                    except ZeroDivisionError:
                        continue

            for k in range(len(self.values[0])):
                try:
                    iden_values[k] = list(map(lambda x: x if (abs(x) > lowlimit) else 0,
                                              [iden_values[k][l] / v[k][k] for l in range(len(self.values[0]))]))

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
            max = 0
            for k in sum_list:
                if k > max:
                    max = k

            alpha = 1 / max

            guess = tpose * alpha

            identity = Matrix.identity(len(self.values), decimal) * 2

            for k in range(resolution):
                guess = guess * (identity - self * guess)
                #guess = guess * 2 - guess * self * guess
            return guess

        elif method == "neumann":
            # don't forget to calibrate the resolution here
            i = Matrix.identity(len(self.values), decimal)
            M = self - i

            for k in range(resolution):
                i += pow(-1, k + 1) * pow(M, k + 1, decimal)

            return i

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
        if not isinstance(dim, int): raise ArgTypeError("Must be an integer.")
        if dim <= 0:
            raise RangeError()
        v = []
        if decimal:
            for k in range(0, dim):
                temp = [Decimal(0)] * dim
                temp[k] = Decimal(1)
                v.append(Vector(*temp))
        else:
            for k in range(0, dim):
                temp = [0] * dim
                temp[k] = 1
                v.append(Vector(*temp))
        return Matrix(*v)

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
        if not (isinstance(a, int) and isinstance(b, int)): raise ArgTypeError("Must be an integer.")
        if a <= 0 or b <= 0:
            raise RangeError()

        if decimal:
            return Matrix(*[Vector(*[Decimal(0) for l in range(b)]) for k in range(a)])
        return Matrix(*[Vector(*[0 for l in range(b)]) for k in range(a)])

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
        if not (isinstance(a, int) and isinstance(b, int)): raise ArgTypeError("Must be an integer.")
        if a <= 0 or b <= 0:
            raise RangeError()

        if decimal:
            return Matrix(*[Vector(*[Decimal(1) for l in range(b)]) for k in range(a)])
        return Matrix(*[Vector(*[1 for l in range(b)]) for k in range(a)])

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
            return Matrix(*[Vector(*[Decimal(random.randint(0, 1)) for l in range(n)]) for k in range(m)])
        return Matrix(*[Vector(*[random.randint(a, b) for l in range(n)]) for k in range(m)])

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
        if not (isinstance(m, int) and isinstance(n, int)): raise ArgTypeError("Must be an integer.")
        if not ((isinstance(a, Union[int, float, Decimal]))
                and (isinstance(b, Union[int, float, Decimal]))):
            raise ArgTypeError("Must be a numerical value")
        if m <= 0 or n <= 0:
            raise RangeError()

        if decimal:
            return Matrix(*[Vector(*[Decimal(random.uniform(a, b)) for l in range(n)]) for k in range(m)])
        return Matrix(*[Vector(*[random.uniform(a, b) for l in range(n)]) for k in range(m)])

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
        if not (isinstance(m, int) and isinstance(n, int)): raise ArgTypeError("Must be an integer.")
        if m <= 0 or n <= 0:
            raise RangeError()

        if decimal:
            return Matrix(*[Vector(*[Decimal(random.randint(0, 1)) for l in range(n)]) for k in range(m)])
        return Matrix(*[Vector(*[random.randint(0, 1) for l in range(n)]) for k in range(m)])

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
        if not (isinstance(m, int) and isinstance(n, int)): raise ArgTypeError("Must be an integer.")
        if not ((isinstance(mu, Union[int, float, Decimal]))
                and (isinstance(sigma, Union[int, float, Decimal]))):
            raise ArgTypeError("Must be a numerical value.")
        if m <= 0 or n <= 0:
            raise RangeError()

        if decimal:
            return Matrix(*[Vector(*[Decimal(random.gauss(mu, sigma)) for l in range(n)]) for k in range(m)])
        return Matrix(*[Vector(*[random.gauss(mu, sigma) for l in range(n)]) for k in range(m)])

    def echelon(self):
        """
            Computes the echelon form of the matrix.

            Returns:
                Matrix: The echelon form of the matrix.
        """
        v = self.values.copy()
        taken_list = list()
        counter = 0
        for k in range(0, len(self.values)):
            for l in range(0, len(self.values[0])):
                if not self.values[k][l] == 0 and l not in taken_list:
                    v[l] = self.values[k]
                    counter += 1
                    if not l == k and counter % 2 == 0:
                        v[l] = [-z for z in self.values[k]]
                    else:
                        v[l] = self.values[k]
                    taken_list.append(l)
                    break
                elif not self.values[k][l] == 0 and l in taken_list:
                    for m in range(l, len(self.values)):
                        if m not in taken_list:
                            v[m] = self.values[k]
                            counter += 1
                            if not m == k and counter % 2 == 0:
                                v[m] = [-z for z in self.values[k]]
        for k in range(0, len(self.values[0])):
            if v[k][k] == 0:
                continue
            for l in range(0, len(self.values)):
                if l == k:
                    continue
                try:
                    factor = (v[l][k]) / (v[k][k])
                    if abs(factor) < 0.0000000001:
                        factor = 0
                    factored_list = [v[l][m] - (factor * v[k][m]) for m in range(0, len(self.values[0]))]
                    v[l] = factored_list
                except ZeroDivisionError:
                    continue
        taken_list = list()
        end_list = v.copy()
        for k in range(0, len(self.values)):
            for l in range(0, len(self.values[0])):
                if not v[k][l] == 0 and l not in taken_list:
                    end_list[l] = v[k]
                    counter += 1
                    if not k == l and counter % 2 == 0:
                        end_list[l] = [-z for z in v[k]]
                    taken_list.append(l)
                    break
                elif not v[k][l] == 0 and l in taken_list:
                    for m in range(l, len(self.values)):
                        if m not in taken_list:
                            end_list[m] = v[k]
                            counter += 1
                            if not m == l and counter % 2 == 0:
                                end_list[m] = [-z for z in v[k]]
        return Matrix(*[Vector(*k) for k in end_list])

    def cramer(a, number: int):
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
        if not isinstance(a, Matrix):
            raise ArgTypeError("Must be a numerical value.")
        if not number < len(a.values[0]) - 1 or number < 0:
            raise RangeError()
        v = []
        for k in range(0, len(a.values)):
            m = []
            for l in range(0, len(a.values[0]) - 1):
                if not l == number:
                    m.append(a.values[k][l])
                else:
                    m.append(a.values[k][len(a.values[0]) - 1])
            v.append(Vector(*m))
        first = Matrix(*v).determinant()
        v.clear()
        for k in range(0, len(a.values)):
            m = []
            for l in range(0, len(a.values[0]) - 1):
                m.append(a.values[k][l])
            v.append(Vector(*m))
        second = Matrix(*v).determinant()
        try:
            sol = first/second
        except ZeroDivisionError:
            sol = None
        return sol

    def cumsum(self):
        """
            Computes the cumulative sum of all elements in the matrix.

            Returns:
                float: The cumulative sum of all elements.

            Notes:
                The cumulative sum is the sum of all elements in the matrix.
        """
        sum = 0
        for k in self.values:
            for l in k:
                sum += l

        return sum

    def reshape(self, *args):
        """
            Reshapes the matrix to the specified dimensions.

            Args:
                *args (int): The dimensions to reshape the matrix into.

            Returns:
                Vector: The reshaped matrix.

            Raises:
                AmountError: If the number of arguments is not in the range [1, 2].
                RangeError: If any dimension value is less than or equal to 0, or if the total number of elements does not match the original matrix size.

            Notes:
                This method reshapes the matrix to the specified dimensions while preserving the order of elements.
        """
        if not (0 < len(args) < 3): raise AmountError()
        for k in args:
            if not isinstance(k, int): raise RangeError()
            if k <= 0: raise RangeError()

        temp = []
        for k in self.values:
            for l in k:
                temp.append(l)
        v = Vector(*temp)
        if len(args) == 1:
            if args[0] != len(self.values) * len(self.values[0]): raise RangeError()
            temp = []
            for k in self.values:
                for l in k:
                    temp.append(l)
            return v
        if args[0] * args[1] != len(self.values) * len(self.values[0]): raise RangeError()
        return v.reshape(args[0], args[1])

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
        if self.dimension.split("x")[0] != self.dimension.split("x")[1]: raise DimensionError(2)
        if resolution < 1: raise RangeError()

        to_work = self.copy()
        for k in range(resolution):
            Q, R = to_work.qr(decimal)
            to_work = R * Q
        result = []
        for k in range(len(to_work.values)):
            result.append(to_work.values[k][k])

        return result

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
        if self.dimension.split("x")[0] != self.dimension.split("x")[1]: raise DimensionError(2)
        v_list = []
        for k in self.transpose():
            v_list.append(Vector(*k))
        if not Vector.does_span(*v_list):
            m = Matrix.zero(len(self.values), decimal)
            return m, m
        result_list = [k.unit() for k in Vector.spanify(*v_list)]
        Q = Matrix(*result_list).transpose()
        R = Q.transpose() * self
        return Q, R

    def cholesky(self):
        """
            Computes the Cholesky decomposition of the matrix.

            Returns:
                Matrix: The lower triangular Cholesky factor of the matrix.

            Raises:
                DimensionError: If the matrix is not square.
        """
        if self.dimension.split("x")[0] != self.dimension.split("x")[1]: raise DimensionError(2)
        L = Matrix.zero(len(self.values), False)
        L.values[0][0] = sqrt(self[0][0])

        for i in range(len(self.values)):
            for j in range(i + 1):
                sum = 0
                for k in range(j):
                    sum += L[i][k] * L[j][k]

                if i == j:
                    L.values[i][j] = sqrt(self[i][i] - sum)
                else:
                    L.values[i][j] = (1.0 / L.values[j][j]) * (self[i][j] - sum)
        return Matrix(*[Vector(*k) for k in L.values])

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
        if self.dimension.split("x")[0] != self.dimension.split("x")[1]: raise DimensionError(2)

        v_list = []
        for k in range(len(self.values)):
            temp = [0] * len(self.values)
            for l in range(len(self.values)):
                if l == k:
                    temp[l] = self[k][l]
            v_list.append(Vector(*temp))

        return Matrix(*v_list)

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
        if self.dimension.split("x")[0] != self.dimension.split("x")[1]: raise DimensionError(2)

        v_list = []
        for k in range(len(self.values)):
            temp = [0] * len(self.values)
            for l in range(len(self.values)):
                if l < k:
                    temp[l] = self[k][l]
            v_list.append(Vector(*temp))

        return Matrix(*v_list)

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
        if self.dimension.split("x")[0] != self.dimension.split("x")[1]: raise DimensionError(2)

        v_list = []
        for k in range(len(self.values)):
            temp = [0] * len(self.values)
            for l in range(len(self.values)):
                if l > k:
                    temp[l] = self[k][l]
            v_list.append(Vector(*temp))

        return Matrix(*v_list)

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
        if i >= dim or j >= dim: raise RangeError()
        if resolution < 1: raise RangeError()

        v_list = [[0 for l in range(dim)] for k in range(dim)]
        for k in range(dim):
            v_list[k][k] = 1

        c = cos(angle, resolution=resolution)
        s = sin(angle, resolution=resolution)
        v_list[i][i] = c
        v_list[j][j] = c
        v_list[i][j] = s
        v_list[j][i] = -s
        return Matrix(*[Vector(*k) for k in v_list])

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
        if not (isinstance(a, Matrix) and isinstance(b, Matrix)): raise ArgTypeError("Must be a matrix.")
        if a.dimension != b.dimension: raise DimensionError(0)

        temp = a.copy().conjugate()

        result = 0
        for i in range(len(a.values)):
            for j in range(len(a.values[0])):
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
        if self.dimension.split("x")[0] != self.dimension.split("x")[1]: raise DimensionError(2)
        sum = 0
        for k in range(len(self.values)):
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
        if self.dimension.split("x")[0] != self.dimension.split("x")[1]: raise DimensionError(2)
        diag = []
        for k in range(len(self.values)):
            diag.append(self.values[k][k])
        return diag

    def diagonal_mul(self):
        """
            Computes the product of the diagonal elements of the square matrix.

            Returns:
                float: The product of the diagonal elements.

            Raises:
                DimensionError: If the matrix is not square.
        """
        if self.dimension.split("x")[0] != self.dimension.split("x")[1]: raise DimensionError(2)
        sum = 1
        for k in range(len(self.values)):
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
            if not isinstance(initial, Vector): raise ArgTypeError("Must be a vector.")
        if b.dimension != len(self.values): raise DimensionError(0)
        if resolution < 1: raise RangeError()
        if initial is None:
            initial = Vector.zero(len(self.values), decimal)
            for i in range(initial.dimension):
                initial.values[i] = b[i] / self[i][i]
        for_l = []
        for_u = []
        for k in range(len(self.values)):
            for_l.append(Vector.zero(len(self.values), decimal))
            for_u.append(Vector.zero(len(self.values), decimal))
            for l in range(len(self.values[0])):
                if l <= k:
                    for_l[-1].values[l] = self[k][l]
                else:
                    for_u[-1].values[l] = self[k][l]
        L_inverse = Matrix(*for_l).inverse(resolution=resolution, decimal=decimal)
        U = Matrix(*for_u)
        for k in range(resolution):
            initial = L_inverse * (b - U * initial)
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
        if not isinstance(b, Vector): raise ArgTypeError("Must be a vector.")
        if len(self.values) != b.dimension: raise DimensionError(0)

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
        if not isinstance(b, Vector): raise ArgTypeError("Must be a vector.")
        if len(self.values) != b.dimension: raise DimensionError(0)
        if resolution < 1: raise RangeError()

        D = self.get_diagonal()
        v_list = []
        for k in range(len(D.values)):
            temp = [0] * len(D.values)
            for l in range(len(D.values)):
                if l == k:
                    temp[l] = 1 / D[k][l]
            v_list.append(Vector(*temp))

        D_inverse = Matrix(*v_list)

        T = -D_inverse * (self - D)
        C = D_inverse * b
        del D, D_inverse

        x = Vector(*[1 for k in range(b.dimension)])
        for i in range(resolution):
            x = T * x + C
        return x

    def toInt(self):
        """
            Converts the matrix to an integer matrix.

            Returns:
                Matrix: The integer matrix.
        """
        return Matrix(*[Vector(*[int(item) for item in row]) for row in self.values])

    def toFloat(self):
        """
            Converts the matrix to a float matrix.

            Returns:
                Matrix: The float matrix.
        """
        return Matrix(*[Vector(*[float(item) for item in row]) for row in self.values])

    def toBool(self):
        """
            Converts the matrix to a boolean matrix.

            Returns:
                Matrix: The boolean matrix.
        """
        return Matrix(*[Vector(*[bool(item) for item in row]) for row in self.values])

    def toDecimal(self):
        """
            Converts the matrix to a decimal matrix.

            Returns:
                Matrix: The decimal matrix.
        """
        return Matrix(*[Vector(*[Decimal(item) for item in row]) for row in self.values])

    def map(self, f):
        """
            Applies a function element-wise to the matrix.

            Args:
                f (Callable): The function to apply.

            Returns:
                Matrix: The matrix with the function applied.
        """
        if not isinstance(f, Callable): raise ArgTypeError("f must be a callable.")
        return Matrix(*[Vector(*[f(item) for item in row]) for row in self.values])

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
        if not isinstance(f, Callable): raise ArgTypeError("f must be a callable.")
        vlist = []
        for row in self.values:
            temp = []
            for item in row:
                if f(item):
                    temp.append(item)
            vlist.append(Vector(*temp))
        return Matrix(*vlist)

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
        # a-b is row, c-d is column
        if not (isinstance(a, int) or isinstance(b, int) or isinstance(c, int) or isinstance(d, int)):
            raise ArgTypeError("Must be an integer.")

        vlist = []
        for row in self.values[a:b]:
            vlist.append(Vector(*row[c:d]))
        return Matrix(*vlist)

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
        return sum / (len(self.values) * len(self.values[0]))

def maximum(dataset):
    """
        Finds the maximum value in the given dataset.

        Args:
            dataset (Union[tuple, list, Vector, Matrix]): The dataset to search for the maximum value.

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

def minimum(dataset):
    """
        Finds the minimum value in the given dataset.

        Args:
            dataset (Union[tuple, list, Vector, Matrix]): The dataset to search for the minimum value.

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

def __mul(row: list, m, id: int, target: dict, amount: int):
    """
        Perform multiplication of a row with a matrix.

        Args:
        - row (list): The row of the first matrix to be multiplied.
        - m: The second matrix.
        - id (int): Identifier for the row being processed.
        - target (dict): Dictionary to store the resulting row.
        - amount (int): Number of elements in the row and the number of rows in the second matrix.

        This function multiplies a given row from the first matrix with the second matrix.
        It calculates the dot product of the row with each column of the second matrix and stores the result in the target dictionary.
        The id parameter is used as an identifier for the row being processed.
    """
    length = len(m[0])  # Number of columns for the second matrix
    result = [0] * length

    for k in range(length):
        sum = 0
        for l in range(amount):
            sum += row[l] * m[l][k]
        result[k] = sum

    target[id] = result

def matmul(m1, m2, max: int = 10):
    """
        Perform matrix multiplication between two matrices.

        Args:
        - m1 (Matrix): The first matrix.
        - m2 (Matrix): The second matrix.
        - max (int, optional): The maximum number of threads to use for parallel computation. Defaults to 10.

        Returns:
        - Matrix: The resulting matrix after multiplication.

        Raises:
        - ArgTypeError: If either m1 or m2 is not an instance of the Matrix class.
        - DimensionError: If the dimensions of the matrices are incompatible for matrix multiplication.

        This function performs matrix multiplication between two matrices, m1 and m2.
        It utilizes threading for parallel computation when the number of rows in m1 exceeds a threshold.
        The max parameter determines the maximum number of threads to use concurrently.
        """
    if not (isinstance(m1, Matrix) and isinstance(m2, Matrix)): raise ArgTypeError()
    a, b = [int(k) for k in m1.dimension.split("x")]
    data = {}
    m1values = m1.values

    c, d = [int(k) for k in m2.dimension.split("x")]
    if not b == c: raise DimensionError(0)
    m2values = m2.values
    if a < 5: return m1 * m2

    count = 0
    pool = [0] * max
    for k in range(a):
        if count >= max:
            pool[-1].join()
            count = 0
        pool[count] = threading.Thread(target=__mul, args=[m1values[k], m2values, k, data, a])
        # pool.append(threading.Thread(target=__mul, args=[m1.values[k], m2, k, data, a]))
        pool[count].start()
        count += 1

    for k in pool:
        try:
            k.join()
        except:
            pass
    return Matrix(*[Vector(*data[k]) for k in range(a)])

def cumsum(arg):
    """
        Computes the cumulative sum of numerical elements in the input iterable.

        Args:
            arg (list, tuple, Vector, Matrix): The iterable containing numerical elements for which the cumulative sum is to be computed.

        Returns:
            int, float, Decimal: The cumulative sum of numerical elements in the iterable.

        Raises:
            ArgTypeError: If the argument is not of a valid type or if elements of 'arg' are not numerical.
    """
    if isinstance(arg, Union[list, tuple, Vector]):
        sum = 0
        try:
            for k in arg:
                sum += k
            return sum
        except:
            raise ArgTypeError("Elements of arg must be numerical")
    elif isinstance(arg, Vector):
        vals = arg.values
        if isinstance(vals[0], Decimal):
            sum = Decimal(0)
            for k in vals:
                sum += k
            return sum
        sum = 0
        for k in vals:
            sum += k
        return sum
    elif isinstance(arg, Matrix):
        rows = arg.values
        if isinstance(rows[0][0], Decimal):
            sum = Decimal(0)
            for k in rows:
                for l in k:
                    sum += l
            return sum
        sum = 0
        for k in rows:
            for l in k:
                sum += l
        return sum
    raise ArgTypeError("Must be an iterable.")

def mode(arg):
    """
        Finds the mode in the given dataset.

        Args:
            arg (Union[tuple, list, Vector, Matrix]): The dataset to find the mode.

        Returns:
            Any: The mode value found in the dataset.

        Raises:
            ArgTypeError: If the dataset type is not supported.
    """
    if isinstance(arg, Union[tuple, list]):
        max = [None, Infinity(False)]
        counts = {k: 0 for k in arg}
        for element in arg:
            counts[element] += 1
            if counts[element] >= max[1]:
                max[0] = element
                max[1] = counts[element]
        return max[0]
    if isinstance(arg, Vector):
        max = [None, Infinity(False)]
        counts = {k: 0 for k in arg.values}
        for element in arg.values:
            counts[element] += 1
            if counts[element] >= max[1]:
                max[0] = element
                max[1] = counts[element]
        return max[0]
    if isinstance(arg, Matrix):
        max = [None, Infinity(False)]
        counts = {}
        for k in arg.values:
            for l in k.values:
                counts[l] = 0

        for k in arg.values:
            for l in k.values:
                counts[l] += 1
                if counts[l] >= max[1]:
                    max[0] = l
                    max[1] = l

        return max[0]
    raise ArgTypeError()

def mean(arg):
    """
        Calculates the mean (average) of the given dataset.

        Args:
            arg (Union[list, tuple, Vector, Matrix, dict]): The dataset for which to calculate the mean.

        Returns:
            float: The mean value of the dataset.

        Raises:
            ArgTypeError: If the dataset type is not supported or invalid.
    """
    if isinstance(arg, Union[list, tuple, Vector]):
        sum = 0
        for k in range(len(arg)):
            sum += arg[k]
        return sum / len(arg)
    if isinstance(arg, Matrix):
        sum = 0
        for k in range(len(arg.values)):
            for l in range(len(k)):
                sum += arg[k][l]
        return sum / (len(arg.values) * len(arg.values[0]))
    if isinstance(arg, dict):
        sum = 0
        count = 0
        for k, v in arg:
            sum += v
            count += 1
        return sum / count
    raise ArgTypeError()

def median(data):
    """
        Calculates the median of the given dataset.

        Args:
            data (Union[list, tuple, Vector]): The dataset for which to calculate the median.

        Returns:
            float: The median value of the dataset.

        Raises:
            ArgTypeError: If the dataset type is not supported or invalid.
    """
    if isinstance(data, list):
        arg = data.copy()
    elif isinstance(data, tuple):
        arg = list(data)
    elif isinstance(data, Vector):
        arg = data.values.copy()
    else: raise ArgTypeError()
    arg.sort()
    n = len(arg)
    if n // 2 == n / 2:
        return arg[n // 2]
    else:
        return (arg[n // 2] + arg[(n // 2) + 1]) / 2

def expectation(values, probabilities, moment: int = 1):
    """
        Calculates the expectation of a random variable.

        Args:
            values (Union[list, tuple, Vector]): The values of the random variable.
            probabilities (Union[list, tuple, Vector]): The corresponding probabilities for each value.
            moment (int, optional): The order of the moment. Defaults to 1.

        Returns:
            float: The expectation value of the random variable.

        Raises:
            RangeError: If the moment is negative.
            DimensionError: If the lengths of values and probabilities are different.
            ArgTypeError: If arguments are not one-dimensional iterables.
    """
    if moment < 0: raise RangeError()
    if (isinstance(values, Union[list, tuple, Vector])) \
        and (isinstance(probabilities, Union[list, tuple, Vector])):
        if len(values) != len(probabilities): raise DimensionError(0)

        sum = 0
        for k in range(len(values)):
            sum += (values[k]**moment) * probabilities[k]
        return sum
    raise ArgTypeError("Arguments must be one dimensional iterables")

def variance(values, probabilities):
    """
        Calculates the variance of a random variable.

        Args:
            values (Union[list, tuple, Vector]): The values of the random variable.
            probabilities (Union[list, tuple, Vector]): The corresponding probabilities for each value.

        Returns:
            float: The variance of the random variable.

        Raises:
            DimensionError: If the lengths of values and probabilities are different.
            ArgTypeError: If arguments are not one-dimensional iterables.
    """
    if isinstance(values, Union[list, tuple, Vector]) and isinstance(probabilities, Union[list, tuple, Vector]):
        if len(values) != len(probabilities): raise DimensionError(0)

        sum = 0
        sum2 = 0
        for k in range(len(values)):
            sum += (values[k]**2) * probabilities[k]
            sum2 += values[k] * probabilities[k]
        return sum - sum2**2
    raise ArgTypeError("Arguments must be one dimensional iterables")

def sd(values, probabilities):
    """
        Calculates the standard deviation of a random variable.

        Args:
            values (Union[list, tuple, Vector]): The values of the random variable.
            probabilities (Union[list, tuple, Vector]): The corresponding probabilities for each value.

        Returns:
            float: The standard deviation of the random variable.

        Raises:
            DimensionError: If the lengths of values and probabilities are different.
            ArgTypeError: If arguments are not one-dimensional iterables.
    """
    if isinstance(values, Union[list, tuple, Vector]) and isinstance(probabilities, Union[list, tuple, Vector]):
        if len(values) != len(probabilities): raise DimensionError(0)

        sum = 0
        for k in range(len(values)):
            sum += (values[k]**2) * probabilities[k] - values[k] * probabilities[k]
        return sqrt(sum)
    raise ArgTypeError("Arguments must be one dimensional iterables")

def linear_fit(x, y, rate=0.01, iterations: int = 15) -> tuple:
    """
        Performs linear regression to fit a line to the given data points.

        Args:
            x (Union[list, tuple, Vector]): The independent variable data points.
            y (Union[list, tuple, Vector]): The dependent variable data points corresponding to x.
            rate (Union[int, float, Decimal], optional): The learning rate for gradient descent. Defaults to 0.01.
            iterations (int, optional): The number of iterations for gradient descent. Defaults to 15.

        Returns:
            tuple: A tuple containing the coefficients of the linear model (b0, b1).

        Raises:
            ArgTypeError: If arguments are not one-dimensional iterables or if rate is not a numerical value.
            RangeError: If iterations is less than 1.
            DimensionError: If the lengths of x and y are different.

        Notes:
            - Linear regression is performed using gradient descent to minimize the mean squared error.
            - The algorithm iteratively adjusts the coefficients (b0, b1) to minimize the error.
            - The learning rate controls the step size of each iteration.
            - The number of iterations determines the convergence of the algorithm.
    """
    if not (isinstance(x, Union[list, tuple, Vector]))\
            and (isinstance(y, Union[list, tuple, Vector])):
        raise ArgTypeError("Arguments must be one dimensional iterables")
    if not (isinstance(rate, Union[int, float, Decimal])):
        raise ArgTypeError("Must be a numerical value.")
    if not isinstance(iterations, int): raise ArgTypeError("Must be an integer.")
    if iterations < 1: raise RangeError()
    if len(x) != len(y): raise DimensionError(0)

    N = len(x)
    b0 = 1
    b1 = 1
    for k in range(iterations):
        sum1 = 0
        sum2 = 0
        for i in range(N):
            sum1 += (y[i] - b0 - b1 * x[i])
            sum2 += (y[i] - b0 - b1 * x[i]) * x[i]
        b0 = b0 - rate * (-2 * sum1 / N)
        b1 = b1 - rate * (-2 * sum2 / N)
    return b0, b1

def general_fit(x, y, rate=0.0000002, iterations: int = 15, degree: int = 1) -> Vector:
    """
        Performs polynomial regression to fit a polynomial of specified degree to the given data points.

        Args:
            x (Union[list, tuple, Vector]): The independent variable data points.
            y (Union[list, tuple, Vector]): The dependent variable data points corresponding to x.
            rate (Union[int, float, Decimal], optional): The learning rate for gradient descent. Defaults to 0.0000002.
            iterations (int, optional): The number of iterations for gradient descent. Defaults to 15.
            degree (int, optional): The degree of the polynomial model. Defaults to 1.

        Returns:
            Vector: A vector containing the coefficients of the polynomial model.

        Raises:
            ArgTypeError: If arguments are not one-dimensional iterables or if rate is not a numerical value.
            RangeError: If iterations or degree is less than 1.
            DimensionError: If the lengths of x and y are different.

        Notes:
            - Polynomial regression is performed using gradient descent to minimize the mean squared error.
            - The polynomial model is of the form: b0 + b1*x + b2*x^2 + ... + bn*x^n (where n = degree)
            - The algorithm iteratively adjusts the coefficients to minimize the error.
            - The learning rate controls the step size of each iteration.
            - The number of iterations determines the convergence of the algorithm.
    """
    if (not (isinstance(x, Union[list, tuple, Vector]))
            and (isinstance(y, Union[list, tuple, Vector]))): raise ArgTypeError("Arguments must be one dimensional iterables")
    if not (isinstance(rate, Union[int, float, Decimal])):
        raise ArgTypeError("Must be a numerical value.")
    if not isinstance(iterations, int): raise ArgTypeError("Must be an integer.")
    if iterations < 1 or degree < 1: raise RangeError()
    if len(x) != len(y): raise DimensionError(0)

    # Preprocess
    if not isinstance(x, Vector):
        x = Vector(*[k for k in x])
    if not isinstance(y, Vector):
        y = Vector(*[k for k in y])
    N = len(x)
    b = Vector(*[1 for k in range(degree + 1)])

    # Work
    for k in range(iterations):
        c = Vector(*[0 for p in range(degree + 1)])
        for i in range(N):
            v = Vector(*[x[i]**p for p in range(degree + 1)])
            c += (y[i] - b.dot(v)) * v
        c *= (-2 / N)
        b = b - rate * c

    return b

def kmeans(dataset, k: int = 2, iterations: int = 15, a=0, b=10):
    """
        Performs k-means clustering on the given dataset.

        Args:
            dataset (Union[list, tuple, Vector]): The dataset to be clustered.
            k (int, optional): The number of clusters. Defaults to 2.
            iterations (int, optional): The number of iterations for the k-means algorithm. Defaults to 15.
            a (Union[int, float, Decimal], optional): The lower bound for generating initial cluster centers. Defaults to 0.
            b (Union[int, float, Decimal], optional): The upper bound for generating initial cluster centers. Defaults to 10.

        Returns:
            tuple: A tuple containing the cluster centers and the data assigned to each cluster.

        Raises:
            ArgTypeError: If the dataset is not a one-dimensional iterable or if a or b are not numerical values.
            DimensionError: If the dataset is empty.
            RangeError: If k or iterations are less than 1.
            ArgTypeError: If the elements in the dataset are not of the same type or not of type Vector.

        Notes:
            - The k-means algorithm partitions the dataset into k clusters.
            - It iteratively assigns data points to the nearest cluster center and updates the center.
            - The algorithm terminates when either the maximum number of iterations is reached or the cluster centers converge.
            - The initial cluster centers are randomly generated within the range [a, b].
            - The output is a tuple where the first element is a list of cluster centers (Vectors) and the second element
              is a list of lists containing the data assigned to each cluster.
    """
    if not (isinstance(dataset, Union[list, tuple, Vector])): raise ArgTypeError()
    if len(dataset) == 0: raise DimensionError(1)
    if k < 1: raise RangeError()
    if iterations < 1: raise RangeError()
    if not ((isinstance(a, Union[int, float, Decimal]))
            and (isinstance(b, Union[int, float, Decimal]))): raise ArgTypeError("Must be a numerical value.")

    check = True
    first = type(dataset[0])
    for i in range(1, len(dataset)):
        check &= (first == type(dataset[i]))
    del first

    if not check:
        raise ArgTypeError("All element types must be the same.")

    check = True
    for data in dataset:
        check &= isinstance(data, Vector)

    if not check:
        for i in range(len(dataset)):
            dataset[i] = Vector(*dataset[i])
    del check

    d = len(dataset[0])
    assigns = []
    points = []
    for i in range(k):
        points.append(Vector(*[random.uniform(a, b) for l in range(d)]))

    for i in range(iterations):
        # Main body of the algorithm.
        assigns = [[] for i in range(k)]
        for data in dataset:
            distances = []
            for j in range(k):
                distances.append((data - points[j]).length())
            minima = minimum(distances)
            assigns[distances.index(minima)].append(data)
        for j in range(k):
            amount = len(assigns[j])
            if not amount: continue
            v = Vector.zero(d, False)
            for temp in assigns[j]:
                v += temp
            points[j] = v / amount


    # This will return a 2d list. Both elements are lists.
    # First one is the cluster centers.
    # Second one is data assigned to cluster centers in order.
    return points, assigns

def unique(data):
    """
        Counts the occurrences of unique elements in the given data.

        Args:
            data (Union[list, tuple, Vector, Matrix]): The data to find unique elements and their counts.

        Returns:
            dict: A dictionary containing unique elements as keys and their counts as values.

        Raises:
            ArgTypeError: If the data is not an iterable (list, tuple, Vector, or Matrix).

        Notes:
            - For one-dimensional data (list, tuple, Vector), this function counts the occurrences of each unique element.
            - For a Matrix, the function reshapes it into a one-dimensional structure before counting unique elements.
            - Returns a dictionary where keys represent unique elements and values represent their counts.
        """
    if isinstance(data, Union[list, tuple, Vector]):
        res = {}
        for val in data:
            if val in res:
                res[val] += 1
            else:
                res[val] = 1
        return res
    if isinstance(data, Matrix):
        arg = data.reshape(len(data) * len(data[0]))
        res = {}
        for val in arg:
            if val in res:
                res[val] += 1
            else:
                res[val] = 1
        return res
    raise ArgTypeError("Must be an iterable.")

def isAllUnique(data):
    """
        Checks if all elements in the given data are unique.

        Args:
            data (Union[list, tuple, Vector, Matrix]): The data to check for uniqueness.

        Returns:
            bool: True if all elements are unique, False otherwise.

        Raises:
            ArgTypeError: If the data is not an iterable (list, tuple, Vector, or Matrix).
    """
    if isinstance(data, Union[list, tuple]):
        return len(data) == len(set(data))
    if isinstance(data, Vector):
        return len(data) == len(set(data.values))
    if isinstance(data, Matrix):
        val = len(data) * len(data[0])
        v = data.reshape(val)
        return val == len(set(v.values))
    raise ArgTypeError("Must be an iterable.")

def __permutate(sample, count, length, target):
    """
        Recursive helper function to generate permutations of a given sample.

        Args:
            sample (list): The sample elements to permute.
            count (int): The current count of elements permuted.
            length (int): The total length of the sample.
            target (list): The list to store generated permutations.

        Returns:
            None: The permutations are stored in the target list.

        Notes:
            This function is not intended for direct use. It's called by the permutate function to generate permutations recursively.

    """
    if count == length:
        target.append(sample.copy())
    for k in range(count, length):
        sample[k], sample[count] = sample[count], sample[k]
        __permutate(sample, count + 1, length, target)
        sample[count], sample[k] = sample[k], sample[count]

def permutate(sample):
    """
        Generates all possible permutations of the elements in the given sample.

        Args:
            sample (Union[list, tuple, Vector, Matrix]): The sample elements to permute.

        Returns:
            list: A list containing all possible permutations of the sample elements.

        Raises:
            ArgTypeError: If the sample is not an iterable (list, tuple, Vector, or Matrix).
    """
    if isinstance(sample, Union[list, tuple]):
        arg = list(set(sample))
    elif isinstance(sample, Vector):
        arg = list(set(sample.values))
    elif isinstance(sample, Matrix):
        arg = sample.values
    else: raise ArgTypeError("Must be an iterable.")
    target = []
    __permutate(arg, 0, len(arg), target)
    return target
