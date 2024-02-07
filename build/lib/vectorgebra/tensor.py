from .vectormatrix import *

class Tensor:
    """
        Class representing a multi-dimensional tensor.

        Attributes:
            values (list): List of elements constituting the tensor.
            dimension (list): List representing the dimensionality of the tensor.
    """

    def __init__(self, *items):
        if len(items) == 0:
            self.values = []
            self.dimension = Undefined()
        else:
            first = items[0]
            if isinstance(first, Matrix):
                for k in items[1:]:
                    if not (isinstance(k, Matrix)):
                        raise ArgTypeError("Arguments must be Matrix")
                    if not k.dimension == first.dimension:
                        raise DimensionError(0)
            elif isinstance(first, Tensor):
                for k in items[1:]:
                    if not (isinstance(k, Tensor)):
                        raise ArgTypeError("Arguments must be Tensor")
                    if not k.dimension == first.dimension:
                        raise DimensionError(0)
            else:
                raise ArgTypeError("Arguments must be either Matrix or Tensor")
            self.values = [k for k in items]
            self.dimension = [len(items), *(self.values[0].__subdimension__())]

    def __subdimension__(self):
        return len(self.values), *(self.values[0].__subdimension__())

    def __str__(self):
        result = "[\n"
        if len(self.dimension) <= 3:
            for k in self.values[:-1]:
                result += k.__str__() + "\n,\n"
            result += self.values[-1].__str__()
            result += "\n]"
            return result
        else:
            try:
                return "..." + self.values[0].__str__() + "..."
            except:
                return "[...]"

    def __repr__(self):
        result = "[\n"
        if len(self.dimension) <= 3:
            for k in self.values[:-1]:
                result += k.__str__() + "\n\n"
            result += self.values[-1].__str__()
            result += "\n]"
            return result
        else:
            try:
                return "..." + self.values[0].__str__() + "..."
            except:
                return "[]"

    def __getitem__(self, index):
        return self.values[index]

    def __setitem__(self, index, item):
        if self.values[0].dimension != item.dimension:
            raise DimensionError(0)
        if type(self.values[0]) != type(item):
            raise ArgTypeError("Target and source types must match")
        self.values[index] = item

    def __len__(self):
        return self.dimension[0]

    def append(self, item):
        """
            Appends an item to the tensor.

            Args:
                item: The item to be appended to the tensor.

            Raises:
                DimensionError: If the dimension of the item being appended does not match
                    the dimension of existing items in the tensor.
                ArgTypeError: If the item being appended is not compatible with the tensor.

            Returns:
                None

            Notes:
                This method adds a new item to the tensor. If the tensor is initially empty,
                the dimension of the tensor is updated to match the dimension of the appended item.
                If the tensor is not empty, the dimension of the item being appended must match
                the dimension of the existing items in the tensor.
        """
        if self.dimension == Undefined():
            # This is the same as projection onto higher dimensional space
            self.values.append(item)
            self.dimension = [1, *(item.dimension)]
            return
        # Below code do an automatic type checking...
        if self.values[0].__subdimension__() != item.__subdimension__():
            raise DimensionError(0)
        self.values.append(item)
        self.dimension[0] += 1

    def pop(self, ord=-1):
        """
            Removes and returns the item at the specified position in the tensor.

            Args:
                ord (int): Optional. The index of the item to be removed and returned. If not specified,
                    it defaults to -1.

            Returns:
                The removed item from the tensor.

            Raises:
                IndexError: If the index specified is out of range.

        """
        if self.dimension[0] > 0:
            self.dimension[0] -= 1
            return self.values.pop(ord)
        raise IndexError()

    def copy(self):
        """
            Creates a deep copy of the tensor.

            Returns:
                Tensor: A new tensor containing deep copies of the values of the original tensor.

            Notes:
                This method performs a deep copy, meaning that it recursively copies all elements within the tensor
                and constructs a new tensor with the same structure and values as the original tensor.

        """
        # This is again a recursive call that is done until it reaches Matrix level
        vlist = []
        for k in self.values:
            vlist.append(k.copy())
        return Tensor(*vlist)

    def __add__(self, arg):
        # We don't need to check for dimension errors.
        # Since no real addition operation is going to
        # be done until we reach the Matrix level, if the
        # dimensions are incorrect, the same error will be raised there.

        if isinstance(arg, Tensor) or isinstance(arg, Matrix):
            vlist = []
            # This is a recursive call. Values of a Tensor could be lower
            # dimensional tensors.

            # For the Matrix case, since this is recursive, recursion will
            # continue until .values reaches a list of matrices. Otherwise
            # it would continue until it reaches a matching dimensional object.
            if self.__subdimension__() == arg.__subdimension__():
                for k in range(self.dimension[0]):
                    vlist.append(self.values[k] + arg.values[k])
            else:
                for k in range(self.dimension[0]):
                    vlist.append(self.values[k] + arg)
            return Tensor(*vlist)
        if isinstance(arg, Vector): # We start to project items into higher dimensional spaces
            mlist = []
            for k in range(self.dimension[-2]):
                mlist.append(arg)  # We don't need to copy. This will be a temporary object already
            M = Matrix(*mlist)
            return self + M
        if isinstance(arg, Union[int, float, Infinity, Undefined]):
                M = arg * Matrix(*[Vector.one(self.dimension[-1], False) for k in range(self.dimension[-2])])
                return self + M
        if isinstance(arg, Decimal):
            M = arg * Matrix(*[Vector.one(self.dimension[-1], True) for k in range(self.dimension[-2])])
            return self + M
        raise ArgTypeError()

    def __iadd__(self, arg):
        if isinstance(arg, Tensor) or isinstance(arg, Matrix):
            if self.__subdimension__() == arg.__subdimension__():
                for k in range(self.dimension[0]):
                    self.values[k] += arg.values[k]
            else:
                for k in range(self.dimension[0]):
                    self.values[k] += arg
            return self
        if isinstance(arg, Vector):  # We start to project items into higher dimensional spaces
            mlist = []
            for k in range(self.dimension[-2]):
                mlist.append(arg)  # We don't need to copy. This will be a temporary object already
            M = Matrix(*mlist)
            self += M
            return self
        if isinstance(arg, Union[int, float, Infinity, Undefined]):
            M = arg * Matrix(*[Vector.one(self.dimension[-1], False) for k in range(self.dimension[-2])])
            self += M
            return self
        if isinstance(arg, Decimal):
            M = arg * Matrix(*[Vector.one(self.dimension[-1], True) for k in range(self.dimension[-2])])
            self += M
            return self
        raise ArgTypeError()

    def __radd__(self, arg):
        if isinstance(arg, Matrix):
            vlist = []
            for k in range(self.dimension[0]):
                vlist.append(self.values[k] + arg)
            return Tensor(*vlist)
        if isinstance(arg, Vector):  # We start to project items into higher dimensional spaces
            mlist = []
            for k in range(self.dimension[-2]):
                mlist.append(arg)  # We don't need to copy. This will be a temporary object already
            M = Matrix(*mlist)
            return self + M
        if isinstance(arg, Union[int, float, Infinity, Undefined]):
            M = arg * Matrix(*[Vector.one(self.dimension[-1], False) for k in range(self.dimension[-2])])
            return self + M
        if isinstance(arg, Decimal):
            M = arg * Matrix(*[Vector.one(self.dimension[-1], True) for k in range(self.dimension[-2])])
            return self + M
        raise ArgTypeError()

    def __sub__(self, arg):
        if isinstance(arg, Tensor) or isinstance(arg, Matrix):
            vlist = []
            if self.__subdimension__() == arg.__subdimension__():
                for k in range(self.dimension[0]):
                    vlist.append(self.values[k] - arg.values[k])
            else:
                for k in range(self.dimension[0]):
                    vlist.append(self.values[k] - arg)
            return Tensor(*vlist)
        if isinstance(arg, Vector):
            mlist = []
            for k in range(self.dimension[-2]):
                mlist.append(arg)
            M = Matrix(*mlist)
            return self - M
        if isinstance(arg, Union[int, float, Infinity, Undefined]):
            M = arg * Matrix(*[Vector.one(self.dimension[-1], False) for k in range(self.dimension[-2])])
            return self - M
        if isinstance(arg, Decimal):
            M = arg * Matrix(*[Vector.one(self.dimension[-1], True) for k in range(self.dimension[-2])])
            return self - M
        raise ArgTypeError()

    def __isub__(self, arg):
        if isinstance(arg, Tensor) or isinstance(arg, Matrix):
            if self.__subdimension__() == arg.__subdimension__():
                for k in range(self.dimension[0]):
                    self.values[k] -= arg.values[k]
            else:
                for k in range(self.dimension[0]):
                    self.values[k] -= arg
            return self
        if isinstance(arg, Vector):
            mlist = []
            for k in range(self.dimension[-2]):
                mlist.append(arg)
            M = Matrix(*mlist)
            self -= M
            return self
        if isinstance(arg, Union[int, float, Infinity, Undefined]):
            M = arg * Matrix(*[Vector.one(self.dimension[-1], False) for k in range(self.dimension[-2])])
            self -= M
            return self
        if isinstance(arg, Decimal):
            M = arg * Matrix(*[Vector.one(self.dimension[-1], True) for k in range(self.dimension[-2])])
            self -= M
            return self
        raise ArgTypeError()

    def __rsub__(self, arg):
        if isinstance(arg, Matrix):
            vlist = []
            for k in range(self.dimension[0]):
                vlist.append(arg - self.values[k])
            return Tensor(*vlist)
        if isinstance(arg, Vector):
            mlist = []
            for k in range(self.dimension[-2]):
                mlist.append(arg)
            M = Matrix(*mlist)
            return M - self
        if isinstance(arg, Union[int, float, Infinity, Undefined]):
            M = arg * Matrix(*[Vector.one(self.dimension[-1], False) for k in range(self.dimension[-2])])
            return M - self
        if isinstance(arg, Decimal):
            M = arg * Matrix(*[Vector.one(self.dimension[-1], True) for k in range(self.dimension[-2])])
            return M - self
        raise ArgTypeError()

    def __truediv__(self, arg):
        if isinstance(arg, Union[int, float, Decimal]):
            vlist = []
            for k in range(self.dimension[0]):
                vlist.append(self.values[k] / arg)
            return Tensor(*vlist)
        raise ArgTypeError("Must be a scalar.")

    def __itruediv__(self, arg):
        if isinstance(arg, Union[int, float, Decimal]):
            for k in range(self.dimension[0]):
                self.values[k] /= arg
            return self
        raise ArgTypeError("Must be a scalar.")

    def __rtruediv__(self, arg):
        if isinstance(arg, Union[int, float, Decimal]):
            vlist = []
            for k in range(self.dimension[0]):
                vlist.append(arg / self.values[k])
            return Tensor(*vlist)
        raise ArgTypeError("Must be a scalar.")

    def __floordiv__(self, arg):
        if isinstance(arg, Union[int, float, Decimal]):
            vlist = []
            for k in range(self.dimension[0]):
                vlist.append(self.values[k] // arg)
            return Tensor(*vlist)
        raise ArgTypeError("Must be a scalar.")

    def __ifloordiv__(self, arg):
        if isinstance(arg, Union[int, float, Decimal]):
            for k in range(self.dimension[0]):
                self.values[k] //= arg
            return self
        raise ArgTypeError("Must be a scalar.")

    def __rfloordiv__(self, arg):
        if isinstance(arg, Union[int, float, Decimal]):
            vlist = []
            for k in range(self.dimension[0]):
                vlist.append(arg // self.values[k])
            return Tensor(*vlist)
        raise ArgTypeError("Must be a scalar.")

    def __mul__(self, arg):
        if isinstance(arg, Union[int, float, Decimal]):
            vlist = []
            for k in range(self.dimension[0]):
                vlist.append(self.values[k] * arg)
            return Tensor(*vlist)
        """
        
        Here, I feel the need to explain what I am trying to accomplish.
        The implementation here thinks of the tensor * tensor just as a
        matrix multiplication. However, there are differences. A scalar,
        0 dimensional tensor, yields a matrix when multiplied with a matrix. 
        A vector, 1 dimensional tensor, yields a vector when multiplied with 
        a matrix. But a matrix multiplied with another yields, a matrix.
        
        matrix @ matrix = matrix  --> No dimension reduction
        matrix @ vector = vector  --> Dimension reduction of 1
        matrix @ scalar = matrix  --> No dimension reduction
        
        The developed algorithm projects this idea onto n dimensions.
        
        We assume that (n dimensional tensor) @ (n dimensional tensor) will
        always yield in an n dimensional tensor.
        
        We assume that (n dimensional tensor) @ (n-1 dimensional tensor) will
        always yield in an n-1 dimensional tensor. 
        
        We assume that an n-a dimensional tensor where a >= 2 is a "scalar" from 
        the perspective of an n dimensional tensor. A lower dimensional object
        will seem just like a point from a much higher dimensional perspective
        no matter the "dimension". The resulting operation therefore must yield 
        an n dimensional tensor.
        
        Tensor algebra is complicated. Rules and definitions here are chosen from
        a functional and recursive programming style. 
        
        We will define the base cases here, and redirect higher dimensional cases
        to lower dimensional predefined-hardcoded cases via recursion. This is
        going to be really, really heavy on the RAM. There isn't a humane way
        of coding this.
        
        Creating a threaded version of this will be much easier compared to the
        matrix counterpart. We can just carry recursive calls to threads.
        
        """

        if isinstance(arg, Vector):
            if len(self.values) != len(arg.values): raise DimensionError(0)
            vlist = []
            for k in range(self.dimension[0]):
                # Recursive call will yield in the above Union[int, float, Decimal]
                # case being called. Below operations is totally safe.
                vlist.append(self.values[k] * arg.values[k])
            return Tensor(*vlist)
        if isinstance(arg, Matrix):
            # This case is indeed the same as above.
            if len(self.dimension) >= 4:
                vlist = []
                for k in range(self.dimension[0]):
                    # The recursive call will continue until left hand side
                    # hits a matrix
                    vlist.append(self.values[k] * arg)
                return Tensor(*vlist)
            sum = self.values[0] * arg
            for k in range(1, self.dimension[0]):
                sum += self.values[k] * arg
            return sum
        if isinstance(arg, Tensor):
            # We won't do dimension checking here. If it is problematic,
            # at some point in the recursive calls it will be raised

            vlist = []
            l1 = len(self.dimension)
            l2 = len(arg.dimension)
            if l1 - l2 >= 2:
                for k in range(self.dimension[0]):
                    # Do not lower arg's dimension space
                    vlist.append(self.values[k] * arg)
                return Tensor(*vlist)
            elif l1 - l2 == 1:
                sum = self.values[0] * arg
                for k in range(1, self.dimension[0]):
                    sum += self.values[k] * arg
                return sum
            elif l1 == l2:
                for k in range(self.dimension[0]):
                    # Lower both operands dimension space
                    vlist.append(self.values[k] * arg.values[k])
                return Tensor(*vlist)
            elif l2 - l1 == 1:
                sum = self * arg.values[0]
                for k in range(1, arg.dimension[0]):
                    sum += self * arg.values[k]
                return sum
            else:
                for k in range(arg.dimension[0]):
                    vlist.append(self * arg.values[k])
                return Tensor(*vlist)
        raise ArgTypeError("Must be a numerical value.")

    # There will be no __imul__ for now.
    # It requires too much dimension transformation
    # Which I deem makes it useless.

    def __rmul__(self, arg):
        if isinstance(arg, Union[int, float, Decimal]):
            vlist = []
            for k in range(self.dimension[0]):
                vlist.append(self.values[k] * arg)
            return Tensor(*vlist)
        if isinstance(arg, Vector):
            if len(self.values) != len(arg.values): raise DimensionError(0)
            vlist = []
            for k in range(self.dimension[0]):
                vlist.append(self.values[k] * arg.values[k])
            return Tensor(*vlist)
        if isinstance(arg, Matrix):
            if len(self.dimension) >= 4:
                vlist = []
                for k in range(self.dimension[0]):
                    vlist.append(self.values[k] * arg)
                return Tensor(*vlist)
            sum = self.values[0] * arg
            for k in range(1, self.dimension[0]):
                sum += self.values[k] * arg
            return sum
        if isinstance(arg, Tensor):
            vlist = []
            l1 = len(self.dimension)
            l2 = len(arg.dimension)
            if l1 - l2 >= 2:
                for k in range(self.dimension[0]):
                    vlist.append(self.values[k] * arg)
                return Tensor(*vlist)
            elif l1 - l2 == 1:
                sum = self.values[0] * arg
                for k in range(1, self.dimension[0]):
                    sum += self.values[k] * arg
                return sum
            elif l1 == l2:
                for k in range(self.dimension[0]):
                    vlist.append(self.values[k] * arg.values[k])
                return Tensor(*vlist)
            elif l2 - l1 == 1:
                sum = self * arg.values[0]
                for k in range(1, arg.dimension[0]):
                    sum += self * arg.values[k]
                return sum
            else:
                for k in range(arg.dimension[0]):
                    vlist.append(self * arg.values[k])
                return Tensor(*vlist)
        raise ArgTypeError("Must be a numerical value.")

    def dot(self, arg):
        """
            Computes the dot product between self and another tensor.

            Args:
                arg (Tensor): The tensor with which to compute the dot product.

            Returns:
                Union[int, float, Decimal]: The dot product of the two tensors.

            Raises:
                ArgTypeError: If the argument is not a tensor.
                DimensionError: If the dimensions of the two tensors do not match.

        """
        if not isinstance(arg, Tensor): raise ArgTypeError("Must be a tensor.")
        if self.dimension != arg.dimension: raise DimensionError(0)
        sum = 0
        for k in range(self.dimension[0]):
            sum += self.values[k].dot(arg.values[k])  # Recursive call
        return sum

    def zero(dim, decimal: bool = False):
        """
            Generates a tensor filled with zeros.

            Args:
                dim (Union[list, tuple]): A list or tuple specifying the dimensions of the tensor.
                decimal (bool, optional): A boolean indicating whether the tensor should contain Decimal numbers.
                    Defaults to False.

            Returns:
                Tensor: A tensor filled with zeros according to the specified dimensions.

            Raises:
                ArgTypeError: If the dimension expression is not a list or tuple.
        """
        if not isinstance(dim, Union[list, tuple]): raise ArgTypeError("Dimension expression must be a list or tuple.")
        if len(dim) == 2:
            return Matrix.zero(dim[0], dim[1], decimal)

        vlist = []
        for k in range(dim[0]):
            vlist.append(Tensor.zero(dim[1:], decimal))
        return Tensor(*vlist)

    def one(dim, decimal: bool = False):
        """
            Generates a tensor filled with ones.

            Args:
                dim (Union[list, tuple]): A list or tuple specifying the dimensions of the tensor.
                decimal (bool, optional): A boolean indicating whether the tensor should contain Decimal numbers.
                            Defaults to False.

            Returns:
                Tensor: A tensor filled with ones according to the specified dimensions.

            Raises:
                ArgTypeError: If the dimension expression is not a list or tuple.
        """
        if not isinstance(dim, Union[list, tuple]): raise ArgTypeError("Dimension expression must be a list or tuple.")
        if len(dim) == 2:
            return Matrix.one(dim[0], dim[1], decimal)

        vlist = []
        for k in range(dim[0]):
            vlist.append(Tensor.one(dim[1:], decimal))
        return Tensor(*vlist)

    def identity(dim, p, decimal: bool = False):
        """
            Generates an identity tensor of the specified dimensions and power.

            Args:
                dim (int): The dimension of the tensor.
                p (int): The power of the tensor.
                decimal (bool, optional): A boolean indicating whether the tensor should contain Decimal numbers.
                    Defaults to False.

            Returns:
                Tensor: An identity tensor of dimension dim raised to the power p.

            Raises:
                ArgTypeError: If dim or p is not an integer.
                RangeError: If dim is less than or equal to 0 or if p is less than 1.

            Notes:
                The `identity` method generates an identity tensor of dimension dim raised to the power p.
                The identity tensor is a tensor with 1's along the all-dimensional diagonal and 0's elsewhere.
                If the `decimal` parameter is set to True, the tensor will contain Decimal numbers.
        """
        # [dim^p] dimensional tensor
        if not (isinstance(dim, int) and isinstance(p, int)): raise ArgTypeError("Must be an integer.")
        if dim <= 0 or p <= 1: raise RangeError()
        if p == 2: return Matrix.identity(2, decimal)
        t = Tensor.zero([dim for k in range(p)], decimal)
        if decimal:
            for i in range(dim):
                holder = t.values
                for k in range(p - 1):
                    holder = holder[i]
                holder[i] = Decimal(1)
        else:
            for i in range(dim):
                holder = t.values
                for k in range(p - 1):
                    holder = holder[i]
                holder[i] = 1
        return t

    def map(self, f):
        """
            Applies a function element-wise to each element of the tensor.

            Args:
                f (Callable): A callable function that accepts a single argument.

            Returns:
                Tensor: A new tensor with the function applied to each element.

            Raises:
                ArgTypeError: If f is not a callable function.

            Notes:
                The `map` method applies the provided function `f` element-wise to each element of the tensor.
                It returns a new tensor containing the results of applying the function to each element.
                The provided function `f` should accept a single argument, representing an element of the tensor.
                This method is particularly useful for applying mathematical operations or transformations to tensors.
        """
        if not isinstance(f, Callable): raise ArgTypeError("f must be a callable.")
        vlist = []
        for k in range(self.dimension[0]):
            vlist.append(self.values[k].filter(f))
        return Tensor(*vlist)

    def avg(self):
        """
            Computes the average value of all elements in the tensor.

            Returns:
                float: The average value of all elements in the tensor.
        """
        sum = 0
        for k in range(self.dimension[0]):
            sum += self.values[0].avg()
        factor = 1
        for k in self.dimension:
            factor *= k
        return sum / factor

    def flatten(self):
        """
            Returns a flattened version of the tensor.

            Returns:
                Union[Matrix, Tensor]: A flattened version of the tensor.

            Notes:
                The `flatten` method returns a flattened version of the tensor. If the tensor has more than two dimensions,
                it returns a new `Matrix` object where each row contains the average values of the corresponding rows
                in the original tensor. If the tensor has less than or equal to 2 dimensions, a deep copy of self is
                returned. This method is useful for reducing the dimensionality of tensor data while retaining important
                information about the distribution of values.
        """
        if len(self.dimension) <= 2:
            return self.copy()
        vlist = []
        for k in range(self.dimension[0]):
            row = []
            for l in range(self.dimension[1]):
                row.append(self.values[k][l].avg())
            vlist.append(Vector(*row))
        return Matrix(*vlist)

