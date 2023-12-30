import random
import threading
import logging
from decimal import *

logger = logging.getLogger("root log")
handler = logging.StreamHandler()
format = logging.Formatter('%(asctime)s %(name)-12s %(levelname)-8s %(message)s')

handler.setFormatter(format)
logger.addHandler(handler)
logger.setLevel(logging.INFO)

PI = 3.14159265359
E = 2.718281828459
ln2 = 0.6931471805599569
log2E = 1.4426950408889392
log2_10 = 3.3219280948873626
sqrtpi = 1.77245385091
sqrt2 = 1.41421356237
sqrt2pi = 2.50662827463

__results = {}

# I leave the DimensionError be...
class DimensionError(Exception):
    def __init__(self, code):
        if code == 0:
            super().__init__("Dimensions must match")
        elif code == 1:
            super().__init__("Number of dimensions cannot be zero")
        elif code == 2:
            super().__init__("Matrix must be a square")

class AmountError(Exception):
    def __init__(self):
        super().__init__("Not the correct amount of args")

class RangeError(Exception):
    def __init__(self, hint: str = ""):
        super().__init__(f"Argument(s) out of range{(': ' + hint) if hint else ''}")

class ArgTypeError(Exception):
    def __init__(self, hint: str = ""):
        super().__init__(f"Argument elements are of the wrong type{(': ' + hint) if hint else ''}")


class Vector:
    def __init__(self, *args):
        for k in args:
            if ((not isinstance(k, int)) and (not isinstance(k, float)) and (not isinstance(k, bool) and (not isinstance(k, Decimal)))
                    and (not isinstance(k, Infinity)) and (not isinstance(k, Undefined)) and (not isinstance(k, complex))):
                raise ArgTypeError("Arguments must be numeric or boolean.")
        self.dimension = len(args)
        self.values = [_ for _ in args]

    def __str__(self):
        return str(self.values)

    def __getitem__(self, index):
        return self.values[index]

    def __setitem__(self, key, value):
        self.values[key] = value

    def __len__(self):
        return len(self.values)

    def __add__(self, arg):
        if (isinstance(arg, int) or isinstance(arg, float) or isinstance(arg, complex)
                or isinstance(arg, Infinity) or isinstance(arg, Undefined) or isinstance(arg, Decimal)):
            return Vector(*[self.values[k] + arg for k in range(0, self.dimension)])
        if not isinstance(arg, Vector):
            raise ArgTypeError("Must be a numerical value.")
        if not (self.dimension == arg.dimension):
            raise DimensionError(0)
        return Vector(*[self.values[k] + arg.values[k] for k in range(0, self.dimension)])

    def __radd__(self, arg):
        if (isinstance(arg, int) or isinstance(arg, float) or isinstance(arg, complex)
                or isinstance(arg, Infinity) or isinstance(arg, Undefined) or isinstance(arg, Decimal)):
            return Vector(*[self.values[k] + arg for k in range(0, self.dimension)])
        raise ArgTypeError("Must be a numerical value.")

    def __sub__(self, arg):
        if (isinstance(arg, int) or isinstance(arg, float) or isinstance(arg, complex)
                or isinstance(arg, Infinity) or isinstance(arg, Undefined) or isinstance(arg, Decimal)):
            return Vector(*[self.values[k] - arg for k in range(0, self.dimension)])
        if not isinstance(arg, Vector):
            raise ArgTypeError("Must be a numerical value.")
        if not (self.dimension == arg.dimension):
            raise DimensionError(0)
        return Vector(*[self.values[k] - arg.values[k] for k in range(0, self.dimension)])

    def __rsub__(self, arg):
        if (isinstance(arg, int) or isinstance(arg, float) or isinstance(arg, complex)
                or isinstance(arg, Infinity) or isinstance(arg, Undefined) or isinstance(arg, Decimal)):
            return -Vector(*[self.values[k] - arg for k in range(0, self.dimension)])
        raise ArgTypeError("Must be a numerical value.")

    def dot(self, arg):
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
        if not (isinstance(arg, int) or isinstance(arg, float) or isinstance(arg, complex)
                or isinstance(arg, Infinity) or isinstance(arg, Undefined)  or isinstance(arg, Decimal)):
            raise ArgTypeError("Must be a numerical value.")
        return Vector(*[self.values[k] * arg for k in range(0, self.dimension)])

    def __rmul__(self, arg):
        if not (isinstance(arg, int) or isinstance(arg, float) or isinstance(arg, complex)
                or isinstance(arg, Infinity) or isinstance(arg, Undefined)  or isinstance(arg, Decimal)):
            raise ArgTypeError("Must be a numerical value.")
        return Vector(*[self.values[k] * arg for k in range(0, self.dimension)])

    def __truediv__(self, arg):
        if not (isinstance(arg, int) or isinstance(arg, float) or isinstance(arg, complex)
                or isinstance(arg, Infinity) or isinstance(arg, Undefined)  or isinstance(arg, Decimal)):
            raise ArgTypeError("Must be a numerical value.")
        return Vector(*[self.values[k] / arg for k in range(0, self.dimension)])


    def __floordiv__(self, arg):
        if not (isinstance(arg, int) or isinstance(arg, float) or isinstance(arg, complex)
                or isinstance(arg, Infinity) or isinstance(arg, Undefined) or isinstance(arg, Decimal)):
            raise ArgTypeError("Must be a numerical value.")
        return Vector(*[self.values[k] // arg for k in range(0, self.dimension)])

    def __iadd__(self, arg):
        if (isinstance(arg, int) or isinstance(arg, float) or isinstance(arg, complex)
                or isinstance(arg, Infinity) or isinstance(arg, Undefined) or isinstance(arg, Decimal)):
            return Vector(*[self.values[k] + arg for k in range(0, self.dimension)])
        if not isinstance(arg, Vector):
            raise ArgTypeError("Must be a numerical value.")
        if not (self.dimension == arg.dimension):
            raise DimensionError(0)
        return Vector(*[self.values[k] + arg.values[k] for k in range(0, self.dimension)])

    def __isub__(self, arg):
        if (isinstance(arg, int) or isinstance(arg, float) or isinstance(arg, complex)
                or isinstance(arg, Infinity) or isinstance(arg, Undefined) or isinstance(arg, Decimal)):
            return Vector(*[self.values[k] - arg for k in range(0, self.dimension)])
        if not isinstance(arg, Vector):
            raise ArgTypeError("Must be a numerical value.")
        if not (self.dimension == arg.dimension):
            raise DimensionError(0)
        return Vector(*[self.values[k] - arg.values[k] for k in range(0, self.dimension)])

    def __gt__(self, arg):
        if (isinstance(arg, int) or isinstance(arg, float) or isinstance(arg, complex)
                or isinstance(arg, Infinity) or isinstance(arg, Undefined) or isinstance(arg, Decimal)):
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
        if (isinstance(arg, int) or isinstance(arg, float) or isinstance(arg, complex)
                or isinstance(arg, Infinity) or isinstance(arg, Undefined) or isinstance(arg, Decimal)):
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
        if (isinstance(arg, int) or isinstance(arg, float) or isinstance(arg, complex)
                or isinstance(arg, Infinity) or isinstance(arg, Undefined) or isinstance(arg, Decimal)):
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
        if (isinstance(arg, int) or isinstance(arg, float) or isinstance(arg, complex)
                or isinstance(arg, Infinity) or isinstance(arg, Undefined) or isinstance(arg, Decimal)):
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
        if (isinstance(arg, int) or isinstance(arg, float) or isinstance(arg, complex)
                or isinstance(arg, Infinity) or isinstance(arg, Undefined) or isinstance(arg, Decimal)):
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
        if (isinstance(arg, int) or isinstance(arg, float) or isinstance(arg, complex)
                or isinstance(arg, Infinity) or isinstance(arg, Undefined) or isinstance(arg, Decimal)):
            self.values.append(arg)
            self.dimension += 1
            return
        if not isinstance(arg, Vector):
            raise ArgTypeError("Must be a numerical value.")
        for k in arg.values:
            self.values.append(k)
        self.dimension += arg.dimension

    def copy(self):
        return Vector(*self.values.copy())

    def pop(self, ord=-1):
        try:
            self.values[ord]
        except IndexError:
            raise RangeError()
        popped = self.values.pop(ord)
        self.dimension -= 1
        return popped

    def length(self):
        sum = 0
        for k in self.values:
            sum += k*k
        return sqrt(sum)

    def proj(self, arg):
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
        l = self.length()
        if l:
            temp = [k/l for k in self.values]
        else:
            temp = [Infinity()] * self.dimension
        return Vector(*temp)

    def spanify(*args):
        v_list = list()
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
        v_list = Vector.spanify(*args)
        for k in range(0, len(v_list)):
            for l in range(0, len(v_list)):
                if not v_list[k].dot(v_list[l]) < 0.0000000001 and not k == l:
                    return False
        return True

    def randVint(dim: int, a: int, b: int, decimal: bool = True):
        if not (isinstance(dim, int) and isinstance(a, int) and isinstance(b, int)):
            raise ArgTypeError("Must be an integer.")
        if not (dim > 0):
            raise RangeError
        if decimal:
            return Vector(*[Decimal(random.randint(a, b)) for k in range(0, dim)])
        return Vector(*[random.randint(a, b) for k in range(0, dim)])

    def randVfloat(dim, a: float, b: float, decimal: bool = True):
        if not (isinstance(dim, int) and
                (isinstance(a, int) or isinstance(a, float) or isinstance(a, Decimal)) and
                (isinstance(b, int) or isinstance(b, float) or isinstance(b, Decimal))):
            raise ArgTypeError("Must be a numerical value.")
        if not (dim > 0):
            raise RangeError
        if decimal:
            return Vector(*[Decimal(random.uniform(a, b)) for k in range(0, dim)])
        return Vector(*[random.uniform(a, b) for k in range(0, dim)])

    def randVbool(dim, decimal: bool = True):
        if not isinstance(dim, int): raise ArgTypeError("Must be an integer.")
        if not (dim > 0): raise RangeError
        if decimal:
            return Vector(*[Decimal(random.randrange(0, 2)) for k in range(0, dim)])
        return Vector(*[random.randrange(0, 2) for k in range(0, dim)])

    def randVgauss(dim, mu=0, sigma=0, decimal: bool = True):
        if not isinstance(dim, int): raise ArgTypeError("Must be an integer.")
        if not ((isinstance(mu, int) or isinstance(mu, float) or isinstance(mu, Decimal)) and
                (isinstance(sigma, int) or isinstance(sigma, float) or isinstance(sigma, Decimal))):
            raise ArgTypeError("Must be a numerical value.")
        if not (dim > 0): raise RangeError
        if decimal:
            return Vector(*[Decimal(random.gauss(mu, sigma)) for k in range(dim)])
        return Vector(*[random.gauss(mu, sigma) for k in range(dim)])

    def determinant(*args):
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
        sum = 0
        for k in self.values:
            sum += k
        return sum

    def zero(dim: int, decimal: bool = True):
        # We use the RangeError because dimension can be 0.
        if dim < 0: raise RangeError()
        if decimal:
            return Vector(*[Decimal(0) for k in range(dim)])
        else:
            return Vector(*[0 for k in range(dim)])

    def one(dim: int, decimal: bool = True):
        if dim < 0: raise RangeError()
        if decimal:
            return Vector(*[Decimal(1) for k in range(dim)])
        else:
            return Vector(*[1 for k in range(dim)])

    def reshape(self, m: int, n: int):
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
        return Matrix.givens(self.dimension, i, j, angle, resolution) * self

    def softmax(self, resolution: int = 15):
        temp = Vector(*[e(k, resolution) for k in self.values])
        temp /= temp.cumsum()
        return temp

    def minmax(self):
        minima = minimum(self)
        maxima = maximum(self)
        val = maxima - minima
        if val == 0:
            return self
        return Vector(*[(k - minima) / val for k in self.values])

    def relu(self, leak=0, cutoff=0):
        if not ((isinstance(leak, int) or isinstance(leak, float) or isinstance(leak, Decimal)
                 or isinstance(leak, Infinity) or isinstance(leak, Undefined))
                and (isinstance(cutoff, int) or isinstance(cutoff, float) or isinstance(cutoff, Decimal)
                     or isinstance(cutoff, Infinity) or isinstance(cutoff, Undefined))):
            raise ArgTypeError("Must be a numerical value.")

        return Vector(*[ReLU(k, leak, cutoff) for k in self.values])

    def sig(self, a=1, cutoff=None):
        if not (isinstance(cutoff, int) or isinstance(cutoff, float) or isinstance(cutoff, Decimal)
                or isinstance(cutoff, Infinity) or isinstance(cutoff, Undefined) or (cutoff is None)):
            raise ArgTypeError("Must be a numerical value.")
        # The reason i do that, i want this to be as fast as possible. I restrain myself to use almost always comprehensions.
        # This is not a code that needs to be readable or sth.
        if cutoff is not None:
            return Vector(*[1 if (x > cutoff) else 0 if (x < -cutoff) else sigmoid(x, a) for x in self.values])
        return Vector(*[sigmoid(x, a) for x in self.values])

    def toInt(self):
        return Vector(*[int(k) for k in self.values])

    def toFloat(self):
        return Vector(*[float(k) for k in self.values])

    def toBool(self):
        return Vector(*[bool(k) for k in self.values])

    def toDecimal(self):
        return Vector(*[Decimal(k) for k in self.values])

    def map(self, f):
        return Vector(*[f(k) for k in self.values])

    def filter(self, f):
        vals = []
        for k in self.values:
            if f(k):
                vals.append(k)
        return Vector(*vals)

    def sort(self, reverse: bool=False):
        self.values.sort(reverse=reverse)
        return self

class Matrix:
    def __init__(self, *args):
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

    def __getitem__(self, index):
        return self.values[index]

    def __setitem__(self, key, value):
        self.values[key] = value

    def __add__(self, arg):
        v = []
        if (isinstance(arg, int) or isinstance(arg, float) or isinstance(arg, complex)
                or isinstance(arg, Infinity) or isinstance(arg, Undefined) or isinstance(arg, Decimal)):
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
        if (isinstance(arg, int) or isinstance(arg, float) or isinstance(arg, complex)
                or isinstance(arg, Infinity) or isinstance(arg, Undefined) or isinstance(arg, Decimal)):
            for k in self.values:
                v.append(Vector(*[l + arg for l in k]))
            return Matrix(*v)
        raise ArgTypeError("Must be a numerical value.")

    def __iadd__(self, arg):
        v = []
        if (isinstance(arg, int) or isinstance(arg, float) or isinstance(arg, complex)
                or isinstance(arg, Infinity) or isinstance(arg, Undefined) or isinstance(arg, Decimal)):
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
        if (isinstance(arg, int) or isinstance(arg, float) or isinstance(arg, complex)
                or isinstance(arg, Infinity) or isinstance(arg, Undefined) or isinstance(arg, Decimal)):
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
        if (isinstance(arg, int) or isinstance(arg, float) or isinstance(arg, complex)
                or isinstance(arg, Infinity) or isinstance(arg, Undefined) or isinstance(arg, Decimal)):
            for k in self.values:
                v.append(Vector(*[l - arg for l in k]))
            return -Matrix(*v)
        raise ArgTypeError("Must be a numerical value.")

    def __isub__(self, arg):
        v = []
        if (isinstance(arg, int) or isinstance(arg, float) or isinstance(arg, complex)
                or isinstance(arg, Infinity) or isinstance(arg, Undefined) or isinstance(arg, Decimal)):
            for k in self.values:
                v.append(Vector(*[l - arg for l in k]))
            return Matrix(*v)
        if not (type(arg) == Matrix):
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
        if (isinstance(arg, int) or isinstance(arg, float) or isinstance(arg, complex)
                or isinstance(arg, Infinity) or isinstance(arg, Undefined) or isinstance(arg, Decimal)):
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
        if (isinstance(arg, int) or isinstance(arg, float) or isinstance(arg, complex)
                or isinstance(arg, Infinity) or isinstance(arg, Undefined) or isinstance(arg, Decimal)):
            for k in self.values:
                v.append(Vector(*[l * arg for l in k]))
            return Matrix(*v)
        raise ArgTypeError("Must be a numerical value.")

    def __neg__(self):
        return Matrix(*[Vector(*[-l for l in k]) for k in self.values])

    def __truediv__(self, arg):
        v = []
        if not (isinstance(arg, int) or isinstance(arg, float) or isinstance(arg, complex)
                or isinstance(arg, Infinity) or isinstance(arg, Undefined) or isinstance(arg, Decimal)):
            raise ArgTypeError("Must be a numerical value.")
        for k in self.values:
            v.append(Vector(*[l / arg for l in k]))
        return Matrix(*v)

    def __floordiv__(self, arg):
        v = []
        if not (isinstance(arg, int) or isinstance(arg, float) or isinstance(arg, complex)
                or isinstance(arg, Infinity) or isinstance(arg, Undefined) or isinstance(arg, Decimal)):
            raise ArgTypeError("Must be a numerical value.")
        for k in self.values:
            v.append(Vector(*[l // arg for l in k]))
        return Matrix(*v)

    def __pow__(self, p, decimal: bool = True):
        temp = Matrix.identity(len(self.values), decimal)
        for k in range(p):
            temp *= self
        return temp

    def determinant(self, choice: str = "echelon"):
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
        if not isinstance(arg, Vector):
            raise ArgTypeError("Must be a vector.")
        if not len(self.values[0]) == arg.dimension:
            raise DimensionError(0)
        self.values.append(arg.values)
        temp = self.dimension.split("x")
        temp[0] = str(int(temp[0]) + 1)
        self.dimension = "x".join(temp)

    def copy(self):
        return Matrix(*[Vector(*k.copy()) for k in self.values])

    def pop(self, ord=-1):
        try:
            self.values[ord]
        except IndexError:
            raise RangeError()
        popped = self.values.pop(ord)
        temp = self.dimension.split("x")
        temp[0] = str(int(temp[0]) - 1)
        self.dimension = "x".join(temp)
        return Vector(*popped)

    def transpose(self):
        v = []
        for k in range(0, len(self.values[0])):
            m = []
            for l in range(0, len(self.values)):
                m.append(self.values[l][k])
            v.append(Vector(*m))
        return Matrix(*v)

    def conjugate(self):
        for k in range(len(self.values)):
            for l in range(len(self.values[0])):
                if isinstance(self[k][l], complex):
                    self.values[k][l] = self.values[k][l].conjugate()
        temp = [Vector(*k) for k in self.values]
        return Matrix(*temp)

    def normalize(self, d_method: str = "echelon"):
        d = self.determinant(d_method)
        if not d:
            return Undefined()
        return Matrix(*[Vector(*[val / d for val in k]) for k in self.values])

    def hconj(self):  # Hermitian conjugate
        return self.transpose().conjugate()

    def norm(self, resolution: int = 10, decimal: bool = True):
        temp = self.hconj() * self
        vals = temp.eigenvalue(resolution=resolution, decimal=decimal)
        return sqrt(maximum(vals))


    def inverse(self, method: str = "iterative", resolution: int = 10, lowlimit=0.0000000001, highlimit=100000, decimal: bool = True):
        if method not in ["gauss", "analytic", "iterative", "neumann"]: raise ArgTypeError()
        if resolution < 1: raise RangeError()
        if not ((isinstance(lowlimit, int) or isinstance(lowlimit, float) or isinstance(lowlimit, Decimal))
                and (isinstance(highlimit, int) or isinstance(highlimit, float) or isinstance(highlimit, Decimal))):
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

    def identity(dim, decimal: bool = True):
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

    def zero(dim, decimal: bool = True):
        if not isinstance(dim, int): raise ArgTypeError("Must be an integer.")
        if dim <= 0:
            raise RangeError()

        if decimal:
            return Matrix(*[Vector(*[Decimal(0) for l in range(dim)]) for k in range(dim)])
        return Matrix(*[Vector(*[0 for l in range(dim)]) for k in range(dim)])

    def one(dim, decimal: bool = True):
        if not isinstance(dim, int): raise ArgTypeError("Must be an integer.")
        if dim <= 0:
            raise RangeError()

        if decimal:
            return Matrix(*[Vector(*[Decimal(1) for l in range(dim)]) for k in range(dim)])
        return Matrix(*[Vector(*[1 for l in range(dim)]) for k in range(dim)])

    def randMint(m, n, a, b, decimal: bool = True):
        if not (isinstance(m, int) and isinstance(n, int) and isinstance(a, int) and isinstance(b, int)):
            raise ArgTypeError("Must be an integer.")
        if m <= 0 or n <= 0:
            raise RangeError()

        if decimal:
            return Matrix(*[Vector(*[Decimal(random.randint(0, 1)) for l in range(n)]) for k in range(m)])
        return Matrix(*[Vector(*[random.randint(a, b) for l in range(n)]) for k in range(m)])

    def randMfloat(m, n, a, b, decimal: bool = True):
        if not (isinstance(m, int) and isinstance(n, int)): raise ArgTypeError("Must be an integer.")
        if not ((isinstance(a, int) or isinstance(a, float) or isinstance(a, Decimal))
                and (isinstance(b, int) or isinstance(b, float) or isinstance(b, Decimal))):
            raise ArgTypeError("Must be a numerical value")
        if m <= 0 or n <= 0:
            raise RangeError()

        if decimal:
            return Matrix(*[Vector(*[Decimal(random.uniform(a, b)) for l in range(n)]) for k in range(m)])
        return Matrix(*[Vector(*[random.uniform(a, b) for l in range(n)]) for k in range(m)])

    def randMbool(m, n, decimal: bool = True):
        if not (isinstance(m, int) and isinstance(n, int)): raise ArgTypeError("Must be an integer.")
        if m <= 0 or n <= 0:
            raise RangeError()

        if decimal:
            return Matrix(*[Vector(*[Decimal(random.randint(0, 1)) for l in range(n)]) for k in range(m)])
        return Matrix(*[Vector(*[random.randint(0, 1) for l in range(n)]) for k in range(m)])

    def randMgauss(m, n, mu, sigma, decimal: bool = True):
        if not (isinstance(m, int) and isinstance(n, int)): raise ArgTypeError("Must be an integer.")
        if not ((isinstance(mu, int) or isinstance(mu, float) or isinstance(mu, Decimal))
                and (isinstance(sigma, int) or isinstance(sigma, float) or isinstance(sigma, Decimal))):
            raise ArgTypeError("Must be a numerical value.")
        if m <= 0 or n <= 0:
            raise RangeError()

        if decimal:
            return Matrix(*[Vector(*[Decimal(random.gauss(mu, sigma)) for l in range(n)]) for k in range(m)])
        return Matrix(*[Vector(*[random.gauss(mu, sigma) for l in range(n)]) for k in range(m)])

    def echelon(self):
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
        sum = 0

        for k in self.values:
            for l in k:
                sum += l

        return sum

    def reshape(self, *args):
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

    def eigenvalue(self, resolution: int = 10, decimal: bool = True):
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

    def qr(self, decimal: bool = True):
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
        if not (isinstance(a, Matrix) and isinstance(b, Matrix)): raise ArgTypeError("Must be a matrix.")
        if a.dimension != b.dimension: raise DimensionError(0)

        temp = a.copy().conjugate()

        result = 0
        for i in range(len(a.values)):
            for j in range(len(a.values[0])):
                result += temp[i][j] * b[i][j]

        return result

    def trace(self):
        if self.dimension.split("x")[0] != self.dimension.split("x")[1]: raise DimensionError(2)
        sum = 0
        for k in range(len(self.values)):
            sum += self.values[k][k]
        return sum

    def diagonals(self):
        if self.dimension.split("x")[0] != self.dimension.split("x")[1]: raise DimensionError(2)
        diag = []
        for k in range(len(self.values)):
            diag.append(self.values[k][k])
        return diag

    def diagonal_mul(self):
        if self.dimension.split("x")[0] != self.dimension.split("x")[1]: raise DimensionError(2)
        sum = 1
        for k in range(len(self.values)):
            sum *= self.values[k][k]
        return sum

    def gauss_seidel(self, b: Vector, initial=None, resolution: int = 20, decimal: bool = True):
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

    def least_squares(self, b, method: str = "iterative", resolution: int = 10, lowlimit=0.0000000001, highlimit=100000, decimal: bool = True):
        if not isinstance(b, Vector): raise ArgTypeError("Must be a vector.")
        if len(self.values) != b.dimension: raise DimensionError(0)

        t = self.transpose()
        temp = (t * self).inverse(method=method, resolution=resolution, lowlimit=lowlimit, highlimit=highlimit, decimal=decimal)
        return temp * (t * b)

    def jacobi_solve(self, b, resolution: int = 15):
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
        return Matrix(*[Vector(*[int(item) for item in row]) for row in self.values])

    def toFloat(self):
        return Matrix(*[Vector(*[float(item) for item in row]) for row in self.values])

    def toBool(self):
        return Matrix(*[Vector(*[bool(item) for item in row]) for row in self.values])

    def toDecimal(self):
        return Matrix(*[Vector(*[Decimal(item) for item in row]) for row in self.values])

    def map(self, f):
        return Matrix(*[Vector(*[f(item) for item in row]) for row in self.values])

    def filter(self, f):
        # This can easily raise a DimensionError
        vlist = []
        for row in self.values:
            temp = []
            for item in row:
                if f(item):
                    temp.append(item)
            vlist.append(Vector(*temp))
        return Matrix(*vlist)

    def submatrix(self, a, b, c, d):
        # a-b is row, c-d is column
        if not (isinstance(a, int) or isinstance(b, int) or isinstance(c, int) or isinstance(d, int)):
            raise ArgTypeError("Must be an integer.")

        vlist = []
        for row in self.values[a:b]:
            vlist.append(Vector(*row[c:d]))
        return Matrix(*vlist)

class Graph:

    def __init__(self, vertices=None, edges=None, weights=None, matrix=None, directed: bool = False):

        if not ((isinstance(vertices, tuple) or isinstance(vertices, list) or (vertices is None))
                and (isinstance(edges, tuple) or isinstance(edges, list) or (edges is None))
                and (isinstance(weights, tuple) or isinstance(weights, list) or (weights is None))):
            raise ArgTypeError()

        if edges is not None:
            for pair in edges:
                if len(pair) != 2:
                    raise AmountError()
                if vertices is not None:
                    if pair[0] not in vertices or pair[1] not in vertices:
                        raise KeyError()

        if weights is not None:
            for val in weights:
                if not (isinstance(val, int) or isinstance(val, float) or isinstance(val, Decimal)
                        or isinstance(val, Infinity) or isinstance(val, Undefined)):
                    raise ArgTypeError("Must be a numerical value.")

        self.directed = directed

        if matrix is None:
            # let "vertices" and "edges" be only lists or tuples.
            # weights and edges can also be None.
            if vertices is None:
                self.vertices = []
            else:
                self.vertices = [k for k in vertices]

            if edges is None:
                self.edges = []
            else:
                self.edges = [list(k) for k in edges]

            if weights is None:
                self.weights = [1 for k in self.edges]
            else:
                self.weights = [k for k in weights]

            general = [[0 for k in range(len(self.vertices))] for l in range(len(self.vertices))]
            for k in range(len(self.vertices)):
                if weights is None:
                    for pair in edges:
                        i = self.vertices.index(pair[0])
                        j = self.vertices.index(pair[1])
                        general[i][j] = 1

                else:
                    counter = 0
                    for pair in edges:
                        i = self.vertices.index(pair[0])
                        j = self.vertices.index(pair[1])
                        general[i][j] = weights[counter]
                        counter += 1
                    del counter
            if self.vertices:
                self.matrix = Matrix(*[Vector(*k) for k in general])
            else:
                self.matrix = Matrix(Vector())

        else:
            dim = matrix.dimension.split("x")
            if dim[0] != dim[1]:
                raise DimensionError(2)
            if vertices is not None:
                if int(dim[0]) != len(vertices):
                    raise DimensionError(0)
                self.vertices = [k for k in vertices]  # You can still name your vertices like "a", "b", "c"...
            else:
                self.vertices = [k for k in range(len(matrix.values))]
            # "edges" and "weights" option will be ignored here

            self.weights = []
            self.edges = []
            for k in range(int(dim[0])):
                for l in range(int(dim[1])):
                    val = matrix[k][l]
                    if val != 0:
                        self.edges.append([self.vertices[k], self.vertices[l]])
                        self.weights.append(val)

            self.matrix = matrix.copy()

    def __str__(self):
        m = maximum(self.weights)
        beginning = maximum([len(str(k)) for k in self.vertices]) + 1
        length = len(str(m)) + 1
        length = beginning if beginning >= length else length

        result = beginning * " "
        each = []
        for item in self.vertices:
            the_string = str(item)
            emptyness = length - len(the_string)
            emptyness = emptyness if emptyness >= 0 else 0
            val = " " * emptyness + the_string
            result += val
            each.append(val)
        result += "\n"

        for i in range(len(each)):
            emptyness = beginning - len(str(self.vertices[i]))
            emptyness = emptyness if emptyness >= 0 else 0

            result += " " * emptyness + str(self.vertices[i])
            for j in range(len(each)):
                val = str(self.matrix[i][j])
                emptyness = length - len(val)
                emptyness = emptyness if emptyness >= 0 else 0
                result += " " * emptyness + val
            result += "\n"

        return result

    def addedge(self, label, weight=1):
        if not (isinstance(label, tuple) or isinstance(label, list)): raise ArgTypeError()
        if len(label) != 2: raise AmountError()
        if not (isinstance(weight, int) or isinstance(weight, float)
                or isinstance(weight, Decimal) or isinstance(weight, Infinity) or isinstance(weight, Undefined)):
            raise ArgTypeError("Must be a numerical value.")

        i = self.vertices.index(label[0])
        j = self.vertices.index(label[1])
        self.matrix[i][j] = weight

        self.edges.append(list(label))
        self.weights.append(weight)

        return self

    def popedge(self, label):
        if not (isinstance(label, tuple) or isinstance(label, list)): raise ArgTypeError()
        if len(label) != 2: raise AmountError()

        k = self.edges.index(label)
        self.edges.pop(k)
        self.weights.pop(k)

        i = self.vertices.index(label[0])
        j = self.vertices.index(label[1])
        self.matrix[i][j] = 0
        return label

    def addvertex(self, v):
        self.vertices.append(v)

        general = [[0 for k in range(len(self.vertices))] for l in range(len(self.vertices))]
        for k in range(len(self.vertices)):
            if not self.weights:
                for pair in self.edges:
                    i = self.vertices.index(pair[0])
                    j = self.vertices.index(pair[1])
                    general[i][j] = 1
            else:
                counter = 0
                for pair in self.edges:
                    i = self.vertices.index(pair[0])
                    j = self.vertices.index(pair[1])
                    general[i][j] = self.weights[counter]
                    counter += 1
                del counter

        self.matrix = Matrix(*[Vector(*k) for k in general])
        return self

    def popvertex(self, v):
        k = self.vertices.index(v)
        self.vertices.pop(k)

        while True:
            control = True
            for item in self.edges:
                if v in item:
                    control = False
                    i = self.edges.index(item)
                    self.edges.pop(i)
                    self.weights.pop(i)
                    break

            if control:
                break

        general = [[0 for k in range(len(self.vertices))] for l in range(len(self.vertices))]
        for k in range(len(self.vertices)):
            if not self.weights:
                for pair in self.edges:
                    i = self.vertices.index(pair[0])
                    j = self.vertices.index(pair[1])
                    general[i][j] = 1
            else:
                counter = 0
                for pair in self.edges:
                    i = self.vertices.index(pair[0])
                    j = self.vertices.index(pair[1])
                    general[i][j] = self.weights[counter]
                    counter += 1
                del counter

        self.matrix = Matrix(*[Vector(*k) for k in general])
        return v

    def getdegree(self, vertex):
        if vertex not in self.vertices: raise KeyError()

        d = 0
        for item in self.edges:
            if vertex in item:
                d += 1
        return d

    def getindegree(self, vertex):
        if vertex not in self.vertices: raise KeyError()

        d = 0
        if self.directed:
            for item in self.edges:
                if vertex == item[1]:
                    d += 1
        else:
            for item in self.edges:
                if vertex in item:
                    d += 1
        return d

    def getoutdegree(self, vertex):
        if vertex not in self.vertices: raise KeyError()

        d = 0
        if self.directed:
            for item in self.edges:
                if vertex == item[0]:
                    d += 1
        else:
            for item in self.edges:
                if vertex in item:
                    d += 1
        return d

    def getdegrees(self):
        result = {}
        for v in self.vertices:
            d = self.getdegree(v)
            if d not in result:
                result[d] = 1
            else:
                result[d] += 1

        return result

    def getweight(self, label):
        if not (isinstance(label, tuple) or isinstance(label, list)): raise ArgTypeError()
        if len(label) != 2: raise AmountError()

        i = self.edges.index(label)
        return self.weights[i]

    def isIsomorphic(g, h):
        if not (isinstance(g, Graph) or isinstance(h, Graph)): raise ArgTypeError("Must be a Graph.")
        return g.getdegrees() == h.getdegrees()

    def isEuler(self):
        degrees = self.getdegrees()
        for k in degrees:
            if k % 2:
                return False
        return True

def Range(low, high, step=Decimal(1)):
    if not ((isinstance(low, int) or isinstance(low, float) or isinstance(low, Decimal) or isinstance(low, Infinity) or isinstance(low, Undefined))
            and (isinstance(high, int) or isinstance(high, float) or isinstance(high, Decimal) or isinstance(high, Infinity) or isinstance(high, Undefined))):
        raise ArgTypeError("Must be a numerical value.")
    if not ((high < low) ^ (step > 0)): raise RangeError()

    while (high < low) ^ (step > 0) and not high == low:
        yield low
        low += step

def abs(arg):
    if not (isinstance(arg, int) or isinstance(arg, float) or isinstance(arg, Decimal) or isinstance(arg, Infinity) or isinstance(arg, Undefined)):
        raise ArgTypeError("Must be a numerical value.")

    return arg if (arg >= 0) else -arg

def sqrt(arg, resolution: int = 10):
    if isinstance(arg, int) or isinstance(arg, float) or isinstance(arg, Decimal):
        if resolution < 1: raise RangeError("Resolution must be a positive integer")
        c = True if arg >= 0 else False
        arg = abs(arg)
        digits = 0
        temp = arg
        first_digit = 0
        while temp != 0:
            first_digit = temp
            temp //= 10
            digits += 1

        estimate = (first_digit // 2 + 1) * pow(10, digits // 2)

        for k in range(resolution):
            estimate = (estimate + arg / estimate) / 2

        # Yes we can return the negatives too.
        if c: return estimate
        return complex(0, estimate)
    if isinstance(arg, complex):
        return arg.sqrt()
    if isinstance(arg, Infinity):
        if arg.sign: return Infinity()
        return sqrt(complex(0, 1)) * Infinity()
    if isinstance(arg, Undefined):
        return Undefined()
    raise ArgTypeError()

def cumsum(arg) -> int or float:
    if isinstance(arg, list) or isinstance(arg, tuple) or isinstance(arg, Vector):
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

def __cumdiv(x, power: int):
    if not (isinstance(x, int) or isinstance(x, float) or isinstance(x, Decimal) or isinstance(x, Infinity) or isinstance(x, Undefined)):
        raise ArgTypeError("Must be a numerical value.")

    result: float = 1
    for k in range(power, 0, -1):
        result *= x / k
    return result

def e(exponent, resolution=15):
    if not (isinstance(resolution, int) and resolution >= 1): raise RangeError("Resolution must be a positive integer")
    if not (isinstance(exponent, int) or isinstance(exponent, float) or isinstance(exponent, Decimal) or isinstance(exponent, Infinity) or isinstance(exponent, Undefined)):
        raise ArgTypeError("Must be a numerical value.")

    sum = 1
    for k in range(resolution, 0, -1):
        sum += __cumdiv(exponent, k)
    return sum

def sin(angle, resolution=15):
    if not (isinstance(resolution, int) and resolution >= 1): raise RangeError("Resolution must be a positive integer")
    if not (isinstance(angle, int) or isinstance(angle, float) or isinstance(angle, Decimal) or isinstance(angle, Infinity) or isinstance(angle, Undefined)):
        raise ArgTypeError("Must be a numerical value.")

    radian: float = (2 * PI * (angle % 360 / 360)) % (2 * PI)
    result: float = 0
    if not resolution % 2:
        resolution += 1
    for k in range(resolution, 0, -2):
        result = result + __cumdiv(radian, k) * pow(-1, (k - 1) / 2)

    return result

def cos(angle, resolution=16):
    if not (isinstance(resolution, int) and resolution >= 1): raise RangeError("Resolution must be a positive integer")
    if not (isinstance(angle, int) or isinstance(angle, float) or isinstance(angle, Decimal) or isinstance(angle, Infinity) or isinstance(angle, Undefined)):
        raise ArgTypeError("Must be a numerical value.")

    radian: float = (2 * PI * (angle % 360 / 360)) % (2 * PI)
    result: float = 1

    if resolution % 2:
        resolution += 1

    for k in range(resolution, 0, -2):
        result = result + __cumdiv(radian, k) * pow(-1, k / 2)
    return result

def tan(angle, resolution=16):
    if not (isinstance(resolution, int) and resolution >= 1): raise RangeError("Resolution must be a positive integer")
    try:
        return sin(angle, resolution - 1) / cos(angle, resolution)
        # Because of the error amount, probably cos will never be zero.
    except ZeroDivisionError:
        # sin * cos is positive in this area
        if 90 >= (angle % 360) >= 0 or 270 >= (angle % 360) >= 180: return Infinity()
        return Infinity(False)

def cot(angle, resolution: int = 16):
    try:
        return 1 / tan(angle, resolution)
    except ZeroDivisionError:
        return None

def sinh(x, resolution: int = 15):
    return (e(x, resolution) - e(-x, resolution)) / 2

def cosh(x, resolution: int = 15):
    return (e(x, resolution) + e(-x, resolution)) / 2

def tanh(x, resolution: int = 15):
    try:
        return sinh(x, resolution) / cosh(x, resolution)
        # Indeed cosh is non-negative
    except ZeroDivisionError:
        return None

def coth(x, resolution: int = 15):
    try:
        return cosh(x, resolution) / sinh(x, resolution)
    except ZeroDivisionError:
        if x >= 0: return Infinity()
        return Infinity(False)

def arcsin(x, resolution: int = 20):
    if not (isinstance(x, int) or isinstance(x, float) or isinstance(x, Decimal) or isinstance(x, Infinity) or isinstance(x, Undefined)):
        raise ArgTypeError("Must be a numerical value.")
    if not (-1 <= x <= 1): raise RangeError()
    if resolution < 1: raise RangeError("Resolution must be a positive integer")
    c = 1
    sol = x
    for k in range(1, resolution):
        c *= (2 * k - 1) / (2 * k)
        sol += c * pow(x, 2 * k + 1) / (2 * k + 1)
    return sol * 360 / (2 * PI)

def arccos(x: int or float, resolution: int = 20) -> float:
    if not (-1 <= x <= 1): raise RangeError()
    if resolution < 1: raise RangeError("Resolution must be a positive integer")

    return 90 - arcsin(x, resolution)

def log2(x, resolution: int = 15):
    if not (isinstance(x, int) or isinstance(x, float) or isinstance(x, Decimal) or isinstance(x, Infinity) or isinstance(x, Undefined)):
        raise ArgTypeError("Must be a numerical value.")
    if x <= 0: raise RangeError()
    if resolution < 1: raise RangeError()
    # finally...
    count = 0
    while x > 2:
        x = x / 2
        count += 1

    # x can be a decimal
    for i in range(1, resolution + 1):
        x = x**2
        if x >= 2:
            count += 1 / (2**i)
            x = x / 2

    return count

def ln(x, resolution: int = 15):
    return log2(x, resolution) / log2E

def log10(x, resolution: int = 15):
    return log2(x, resolution) / log2_10

def log(x, base=2, resolution: int = 15):
    if not (isinstance(base, int) or isinstance(base, float) or isinstance(base, Decimal)):
        raise ArgTypeError("Must be a numerical value.")
    if base <= 0 or base == 1: raise RangeError()
    return log2(x, resolution) / log2(base, resolution)

def __find(f, low, high, search_step, res=Decimal(0.0001)):
    if not ((isinstance(low, int) or isinstance(low, float) or isinstance(low, Decimal))
            and (isinstance(high, int) or isinstance(high, float) or isinstance(high, Decimal))
            and (isinstance(search_step, int) or isinstance(search_step, float) or isinstance(search_step, Decimal))
            and (isinstance(res, int) or isinstance(res, float) or isinstance(res, Decimal))):
        raise ArgTypeError("Must be a numerical value.")

    global __results
    last_sign: bool = True if (f(low) >= 0) else False
    for x in Range(low, high, search_step):
        value = f(x)
        temp_sign = True if (value >= 0) else False
        if temp_sign and last_sign and (value // 1 != 0 or value // 1 != -1):
            last_sign = temp_sign
        elif abs(value) < res * 100:
            __results[x] = True
            return x
        elif search_step > res:
            get = __find(f, x - search_step, x, search_step / 10)
            if get is not None:
                __results[get] = True
                return get

def solve(f, low=-50, high=50, search_step=0.1,
          res=Decimal(0.0001)) -> list:
    if not ((isinstance(low, int) or isinstance(low, float) or isinstance(low, Decimal))
            and (isinstance(high, int) or isinstance(high, float) or isinstance(high, Decimal))
            and (isinstance(search_step, int) or isinstance(search_step, float) or isinstance(search_step, Decimal))
            and (isinstance(res, int) or isinstance(res, float) or isinstance(res, Decimal))):
        raise ArgTypeError("Must be a numerical value.")

    # I couldn't find any other way to check it
    if str(type(f)) != "<class 'function'>" and str(
        type(f)) != "<class 'builtin_function_or_method'>": raise ArgTypeError("f must be a callable")

    if high <= low: raise RangeError()
    if search_step <= 0: raise RangeError()
    if res <= 0 or res >= 1: raise RangeError()

    zeroes: list = []
    thread_list: list = []
    last_sign: bool = True if (f(low) >= 0) else False
    for x in Range(low, high, search_step):
        value = f(x)
        temp_sign = True if (value >= 0) else False
        if temp_sign and last_sign and value // 1 != 0 and value // 1 != -1:
            last_sign = temp_sign
        elif abs(value) < 0.001:
            zeroes.append(x)
            last_sign = temp_sign
        else:
            try:
                thread_list.append(threading.Thread(target=__find, args=[f, x - search_step, x, search_step / 10, res]))
                thread_list[-1].start()
            except RuntimeError:
                logger.info(f"Thread count limit reached at {len(thread_list)}.")
                break
            last_sign = temp_sign

    logger.debug("Main loop end reached, waiting for joins.")
    for k in thread_list:
        k.join()
    logger.debug("Joins ended.")
    for k in __results:
        zeroes.append(k)

    __results.clear()

    zeroes = list(map(lambda x: x if (abs(x) > 0.00001) else 0, zeroes))

    return zeroes

def derivative(f, x, h=0.0000000001):
    if not ((isinstance(x, int) or isinstance(x, float) or isinstance(x, Decimal))
            and (isinstance(h, int) or isinstance(h, float) or isinstance(h, Decimal))):
        raise ArgTypeError("Must be a numerical value.")
    if str(type(f)) != "<class 'function'>" and str(
        type(f)) != "<class 'builtin_function_or_method'>": raise ArgTypeError("f must be a callable")

    return (f(x + h) - f(x)) / h

def integrate(f, a, b, delta=0.01):
    if not ((isinstance(a, int) or isinstance(a, float) or isinstance(a, Decimal))
            and (isinstance(b, int) or isinstance(b, float) or isinstance(b, Decimal))
            and (isinstance(delta, int) or isinstance(delta, float) or isinstance(delta, Decimal))):
        raise ArgTypeError("Must be a numerical value.")

    if str(type(f)) != "<class 'function'>" and str(
        type(f)) != "<class 'builtin_function_or_method'>": raise ArgTypeError("f must be a callable")

    if a == b:
        return .0
    half = delta / 2
    sum = 0

    if a > b:
        while a > b:
            sum += f(b + half)
            b += delta
        return sum * delta

    while b > a:
        sum += f(a + half)
        a += delta
    return sum * delta

def __mul(row: list, m, id: int, target: dict, amount: int):
    length = len(m[0])  # Number of columns for the second matrix
    result = [0] * length

    for k in range(length):
        sum = 0
        for l in range(amount):
            sum += row[l] * m[l][k]
        result[k] = sum

    target[id] = result

def matmul(m1, m2, max: int = 10):
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

def findsol(f, x=0, resolution: int = 15):
    """
    Finds a singular solution at a time with Newton's method.

    :param f: Function
    :param x: Starting value
    :param resolution: Number of iterations
    :return: The found/guessed solution of the function
    """
    if str(type(f)) != "<class 'function'>" and str(
        type(f)) != "<class 'builtin_function_or_method'>": raise ArgTypeError("f must be a callable")
    if resolution < 1: raise RangeError("Resolution must be a positive integer")

    for k in range(resolution):
        try:
            x = x - (f(x) / derivative(f, x))
        except ZeroDivisionError:
            if f(x) >= 0:
                x = Infinity(False)
            else:
                x = Infinity()

    return x

def sigmoid(x, a=1):
    if not((isinstance(a, int) or isinstance(a, float) or isinstance(a, Decimal) or isinstance(a, Infinity) or isinstance(a, Undefined))
           and (isinstance(x, int) or isinstance(x, float) or isinstance(x, Decimal) or isinstance(x, Infinity) or isinstance(x, Undefined))):
        raise ArgTypeError("Must be a numerical value.")
    return 1 / (1 + e(-a*x))

def ReLU(x, leak=0, cutoff=0):
    if not ((isinstance(x, int) or isinstance(x, float) or isinstance(x, Decimal) or isinstance(x, Infinity) or isinstance(x, Undefined))
            and (isinstance(leak, int) or isinstance(leak, float) or isinstance(leak, Decimal)
             or isinstance(leak, Infinity) or isinstance(leak, Undefined))
            and (isinstance(cutoff, int) or isinstance(cutoff, float) or isinstance(cutoff, Decimal)
                 or isinstance(cutoff, Infinity) or isinstance(cutoff, Undefined))):
        raise ArgTypeError("Must be a numerical value.")

    if x >= cutoff:
        return x
    elif x < 0:
        return leak * x
    else:
        return cutoff

def deriv_relu(x, leak=0, cutoff=0):
    return 1 if x >= cutoff else 0 if x >= 0 else leak

def Sum(f, a, b, step=Decimal(0.01), control: bool = False, limit=Decimal(0.000001)):
    # Yes, with infinities this can blow up. It is the users problem to put infinities in.
    if not ((isinstance(a, int) or isinstance(a, float) or isinstance(a, Decimal) or isinstance(a, Infinity))
            and (isinstance(b, int) or isinstance(b, float) or isinstance(b, Decimal) or isinstance(b, Infinity))
            and (isinstance(step, int) or isinstance(step, float) or isinstance(step, Decimal) or isinstance(step, Infinity))
            and (isinstance(limit, int) or isinstance(limit, float) or isinstance(limit, Decimal) or isinstance(limit, Infinity) or isinstance(limit, Undefined))):
        raise ArgTypeError("Must be a numerical value.")
    if not b >= a:
        raise RangeError()

    result = 0
    if control:
        for x in Range(a, b, step):
            if abs(derivative(f, x, x + step)) <= limit:
                return result
            result += f(x)
    else:
        for x in Range(a, b, step):
            result += f(x)
        return result

def mode(arg):
    if isinstance(arg, tuple) or isinstance(arg, list):
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
    if isinstance(arg, tuple) or isinstance(arg, list) or isinstance(arg, Vector):
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
    if moment < 0: raise RangeError()
    if (isinstance(values, list) or isinstance(values, tuple) or isinstance(values, Vector)) \
        and (isinstance(probabilities, list) or isinstance(probabilities, tuple) or isinstance(probabilities, Vector)):
        if len(values) != len(probabilities): raise DimensionError(0)

        sum = 0
        for k in range(len(values)):
            sum += (values[k]**moment) * probabilities[k]
        return sum
    raise ArgTypeError("Arguments must be one dimensional iterables")

def variance(values, probabilities):
    if (isinstance(values, list) or isinstance(values, tuple) or isinstance(values, Vector)) \
            and (
            isinstance(probabilities, list) or isinstance(probabilities, tuple) or isinstance(probabilities, Vector)):
        if len(values) != len(probabilities): raise DimensionError(0)

        sum = 0
        sum2 = 0
        for k in range(len(values)):
            sum += (values[k]**2) * probabilities[k]
            sum2 += values[k] * probabilities[k]
        return sum - sum2**2
    raise ArgTypeError("Arguments must be one dimensional iterables")

def sd(values, probabilities) -> float:
    if (isinstance(values, list) or isinstance(values, tuple) or isinstance(values, Vector)) \
            and (
            isinstance(probabilities, list) or isinstance(probabilities, tuple) or isinstance(probabilities, Vector)):
        if len(values) != len(probabilities): raise DimensionError(0)

        sum = 0
        for k in range(len(values)):
            sum += (values[k]**2) * probabilities[k] - values[k] * probabilities[k]
        return sqrt(sum)
    raise ArgTypeError("Arguments must be one dimensional iterables")

def maximum(dataset):
    maxima = Infinity(False)
    if isinstance(dataset, tuple) or isinstance(dataset, list):
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
    minima = Infinity(True)
    if isinstance(dataset, tuple) or isinstance(dataset, list):
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

def factorial(x: int = 0) -> int:
    if x < 0: raise RangeError()
    if x <= 1: return 1
    mul = 1
    for k in range(2, x):
        mul *= k
    return mul * x

def permutation(x: int = 1, y: int = 1) -> int:
    if x < 1 or y < 1 or y > x: raise RangeError()
    result = 1
    for v in range(y + 1, x + 1):
        result *= v
    return result

def combination(x: int = 0, y: int = 0) -> int:
    if x < 0 or y < 0 or y > x: raise RangeError()
    result = 1
    count = 1
    for v in range(y + 1, x + 1):
        result *= v / count
        count += 1
    return result

def multinomial(n: int = 0, *args) -> int:
    sum = 0
    for k in args:
        if not isinstance(k, int): raise ArgTypeError()
        if k < 0: raise RangeError()
        sum += k
    c = [k for k in args]
    if sum != n: raise RangeError("Sum of partitions must be equal to n")
    result = 1
    while n != 1:
        result *= n
        for k in range(len(c)):
            if not c[k]: continue
            result /= c[k]
            c[k] -= 1
        n -= 1
    return result

def binomial(n: int, k: int, p):
    if not (isinstance(p, int) or isinstance(p, float) or isinstance(p, Decimal)):
        raise ArgTypeError("Must be a numerical value.")
    if p < 0 or p > 1: raise RangeError("Probability cannot be negative or bigger than 1")
    return combination(n, k) * pow(p, k) * pow(1 - p, k)

def geometrical(n: int, p):
    if not (isinstance(p, int) or isinstance(p, float) or isinstance(p, Decimal)):
        raise ArgTypeError("Must be a numerical value.")
    if p < 0 or p > 1: raise RangeError("Probability cannot be negative or bigger than 1")
    if n < 0: raise RangeError("Trial number cannot be negative")
    if n == 0: return Undefined()
    return p * pow(1 - p, n - 1)

def poisson(k, l):
    if not ((isinstance(k, int) or isinstance(k, float) or isinstance(k, Decimal))
            and (isinstance(l, int) or isinstance(l, float) or isinstance(l, Decimal))):
        raise ArgTypeError("Must be a numerical value.")
    if l < 0 or k < 0: raise RangeError()
    return pow(l, k) * e(-l) / factorial(k)

def normal(x, resolution: int = 15):
    if not (isinstance(x, int) or isinstance(x, float) or isinstance(x, Decimal) or isinstance(x, Infinity)): raise ArgTypeError("Must be a numerical value.")
    return e(-(x**2) / 2, resolution=resolution) / sqrt2pi

def gaussian(x, mean, sigma, resolution: int = 15):
    if not ((isinstance(x, int) or isinstance(x, float) or isinstance(x, Decimal) or isinstance(x, Infinity))
            and (isinstance(mean, int) or isinstance(mean, float) or isinstance(mean, Decimal) or isinstance(mean, Infinity))
            and (isinstance(sigma, int) or isinstance(sigma, float) or isinstance(sigma, Decimal) or isinstance(sigma, Infinity))):
        raise ArgTypeError("Must be a numerical value.")

    coef = 1 / (sqrt2pi * sigma)
    power = - pow(x - mean, 2) / (2 * pow(sigma, 2))
    return coef * e(power, resolution=resolution)

def laplace(x, sigma, resolution: int = 15):
    if not ((isinstance(x, int) or isinstance(x, float) or isinstance(x, Decimal) or isinstance(x, Infinity))
            and (isinstance(sigma, int) or isinstance(sigma, float) or isinstance(sigma, Decimal) or isinstance(sigma, Infinity))):
        raise ArgTypeError("Must be a numerical value.")

    coef = 1 / (sqrt2 * sigma)
    power = - (sqrt2 / sigma) * abs(x)
    return coef * e(power, resolution=resolution)

def linear_fit(x, y, rate=Decimal(0.01), iterations: int = 15) -> tuple:
    if not (isinstance(x, list) or isinstance(x, tuple) or isinstance(x, Vector))\
            and (isinstance(y, list) or isinstance(y, tuple) or isinstance(y, Vector)):
        raise ArgTypeError("Arguments must be one dimensional iterables")
    if not (isinstance(rate, int) or isinstance(rate, float) or isinstance(rate, Decimal)):
        raise ArgTypeError("Must be a numerical value.")
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

def general_fit(x, y, rate=Decimal(0.0000002), iterations: int = 15, degree: int = 1) -> Vector:
    if (not (isinstance(x, list) or isinstance(x, tuple) or isinstance(x, Vector))
            and (isinstance(y, list) or isinstance(y, tuple) or isinstance(y, Vector))): raise ArgTypeError("Arguments must be one dimensional iterables")
    if not (isinstance(rate, int) or isinstance(rate, float) or isinstance(rate, Decimal)):
        raise ArgTypeError("Must be a numerical value.")
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
    if not (isinstance(dataset, tuple) or isinstance(dataset, list) or isinstance(dataset, Vector)): raise ArgTypeError()
    if len(dataset) == 0: raise DimensionError(1)
    if k < 1: raise RangeError()
    if iterations < 1: raise RangeError()
    if not ((isinstance(a, int) or isinstance(a, float) or isinstance(a, Decimal))
            and (isinstance(b, int) or isinstance(b, float) or isinstance(b, Decimal))): raise ArgTypeError("Must be a numerical value.")

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
    if isinstance(data, tuple) or isinstance(data, list) or isinstance(data, Vector):
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
    if isinstance(data, tuple) or isinstance(data, list):
        return len(data) == len(set(data))
    if isinstance(data, Vector):
        return len(data) == len(set(data.values))
    if isinstance(data, Matrix):
        val = len(data) * len(data[0])
        v = data.reshape(val)
        return val == len(set(v.values))
    raise ArgTypeError("Must be an iterable.")

def __permutate(sample, count, length, target):
    if count == length:
        target.append(sample.copy())
    for k in range(count, length):
        sample[k], sample[count] = sample[count], sample[k]
        __permutate(sample, count + 1, length, target)
        sample[count], sample[k] = sample[k], sample[count]

def permutate(sample):
    if isinstance(sample, list) or isinstance(sample, tuple):
        arg = list(set(sample))
    elif isinstance(sample, Vector):
        arg = list(set(sample.values))
    elif isinstance(sample, Matrix):
        arg = sample.values
    else: raise ArgTypeError("Must be an iterable.")
    target = []
    __permutate(arg, 0, len(arg), target)
    return target

class complex:

    def __init__(self, real=0, imaginary=0):
        # Initialization is erroneous when expressions with infinities result in NoneTypes
        if ((not isinstance(real, int)) and (not isinstance(real, float)) and (not isinstance(imaginary, int))
                and (not isinstance(imaginary, float)) and (not isinstance(real, Infinity)) and (not isinstance(real, Decimal))
                and (not isinstance(imaginary, Infinity)) and (not isinstance(real, Undefined)) and (not isinstance(imaginary, Undefined))
                and (not isinstance(imaginary, Decimal))): raise ArgTypeError()
        self.real = real
        self.imaginary = imaginary

    def __str__(self):
        if self.imaginary >= 0: return f"{self.real} + {self.imaginary}i"

        return f"{self.real} - {-self.imaginary}i"

    def __repr__(self):
        if self.imaginary >= 0: return f"{self.real} + {self.imaginary}i"

        return f"{self.real} - {-self.imaginary}i"

    def __add__(self, arg):
        if ((not isinstance(arg, int)) and (not isinstance(arg, float)) and (not isinstance(arg, complex))
                and (not isinstance(arg, Infinity)) and (not isinstance(arg, Undefined)) and (not isinstance(arg, Decimal))): raise ArgTypeError()

        if isinstance(arg, complex):
            return complex(self.real + arg.real, self.imaginary + arg.imaginary)
        return complex(self.real + arg, self.imaginary)

    def __radd__(self, arg):
        if ((not isinstance(arg, int)) and (not isinstance(arg, float)) and (not isinstance(arg, Infinity))
                and (not isinstance(arg, Undefined)) and (not isinstance(arg, Decimal))): raise ArgTypeError()
        return complex(self.real + arg, self.imaginary)

    def __iadd__(self, arg):
        if ((not isinstance(arg, int)) and (not isinstance(arg, float)) and (not isinstance(arg, complex))
                and (not isinstance(arg, Infinity)) and (not isinstance(arg, Undefined)) and (not isinstance(arg, Decimal))): raise ArgTypeError()

        if isinstance(arg, complex):
            self.real += arg.real
            self.imaginary += arg.imaginary
            return self
        self.real += arg
        return self

    def __sub__(self, arg):
        if ((not isinstance(arg, int)) and (not isinstance(arg, float)) and (not isinstance(arg, complex))
                and (not isinstance(arg, Infinity)) and (not isinstance(arg, Undefined)) and (not isinstance(arg, Decimal))): raise ArgTypeError()

        if isinstance(arg, complex):
            return complex(self.real - arg.real, self.imaginary - arg.imaginary)
        return complex(self.real - arg, self.imaginary)

    def __rsub__(self, arg):
        if ((not isinstance(arg, int)) and (not isinstance(arg, float)) and (not isinstance(arg, Infinity))
                and (not isinstance(arg, Undefined)) and (not isinstance(arg, Decimal))): raise ArgTypeError()
        return -complex(self.real - arg, self.imaginary)

    def __isub__(self, arg):
        if ((not isinstance(arg, int)) and (not isinstance(arg, float)) and (not isinstance(arg, complex))
                and (not isinstance(arg, Infinity)) and (not isinstance(arg, Undefined)) and (not isinstance(arg, Decimal))): raise ArgTypeError()

        if isinstance(arg, complex):
            self.real -= arg.real
            self.imaginary -= arg.imaginary
            return self
        self.real -= arg
        return self

    def __mul__(self, arg):
        if ((not isinstance(arg, int)) and (not isinstance(arg, float)) and (not isinstance(arg, complex))
                and (not isinstance(arg, Infinity)) and (not isinstance(arg, Undefined)) and (not isinstance(arg, Decimal))): raise ArgTypeError()

        if isinstance(arg, complex):
            return complex(self.real * arg.real - self.imaginary * arg.imaginary,
                           self.real * arg.imaginary + self.imaginary * arg.real)
        return complex(self.real * arg, self.imaginary * arg)

    def __rmul__(self, arg):
        if ((not isinstance(arg, int)) and (not isinstance(arg, float)) and (not isinstance(arg, Infinity))
                and (not isinstance(arg, Undefined)) and (not isinstance(arg, Decimal))): raise ArgTypeError()
        return complex(self.real * arg, self.imaginary * arg)

    def __imul__(self, arg):
        if ((not isinstance(arg, int)) and (not isinstance(arg, float)) and (not isinstance(arg, complex))
                and (not isinstance(arg, Infinity)) and (not isinstance(arg, Undefined)) and (not isinstance(arg, Decimal))): raise ArgTypeError()

        if isinstance(arg, complex):
            self.real = self.real * arg.real - self.imaginary * arg.imaginary
            self.imaginary = self.real * arg.imaginary + self.imaginary * arg.real
            return self
        self.real *= arg
        self.imaginary *= arg
        return self

    def __truediv__(self, arg):
        if ((not isinstance(arg, int)) and (not isinstance(arg, float)) and (not isinstance(arg, complex))
                and (not isinstance(arg, Infinity)) and (not isinstance(arg, Undefined)) and (not isinstance(arg, Decimal))): raise ArgTypeError()

        if isinstance(arg, complex):
            return self * arg.inverse()
        if arg: return complex(self.real / arg, self.imaginary / arg)
        return complex(Infinity(self.real >= 0), Infinity(self.imaginary >= 0))

    def __rtruediv__(self, arg):
        if ((not isinstance(arg, int)) and (not isinstance(arg, float)) and (not isinstance(arg, Infinity))
                and (not isinstance(arg, Undefined)) and (not isinstance(arg, Decimal))): raise ArgTypeError()
        return arg * self.inverse()


    def __idiv__(self, arg):
        if ((not isinstance(arg, int)) and (not isinstance(arg, float)) and (not isinstance(arg, complex))
                and (not isinstance(arg, Infinity)) and (not isinstance(arg, Undefined)) and (not isinstance(arg, Decimal))): raise ArgTypeError()

        if isinstance(arg, complex):
            temp = self * arg.inverse()
            self.real, self.imaginary = temp.real, temp.imaginary
            return self
        self.real /= arg
        self.imaginary /= arg
        return self

    def __floordiv__(self, arg):
        if ((not isinstance(arg, int)) and (not isinstance(arg, float)) and (not isinstance(arg, complex))
                and (not isinstance(arg, Infinity)) and (not isinstance(arg, Undefined)) and (not isinstance(arg, Decimal))): raise ArgTypeError()

        if isinstance(arg, complex):
            temp = self * arg.inverse()
            return complex(temp.real // 1, temp.imaginary // 1)
        return complex(self.real // arg, self.imaginary // arg)

    def __ifloordiv__(self, arg):
        if ((not isinstance(arg, int)) and (not isinstance(arg, float)) and (not isinstance(arg, complex))
                and (not isinstance(arg, Infinity)) and (not isinstance(arg, Undefined)) and (not isinstance(arg, Decimal))): raise ArgTypeError()

        if isinstance(arg, complex):
            temp = self * arg.inverse()
            self.real, self.imaginary = temp.real // 1, temp.imaginary // 1
            return self
        self.real //= arg
        self.imaginary //= arg
        return self

    def __neg__(self):
        return complex(-self.real, -self.imaginary)

    def __eq__(self, arg):
        if isinstance(arg, complex):
            if self.real == arg.real and self.imaginary == arg.imaginary:
                return True
            else:
                return False
        return False

    def __ne__(self, arg):
        if isinstance(arg, complex):
            if self.real != arg.real or self.imaginary != arg.imaginary:
                return True
            else:
                return False
        return True

    def __gt__(self, arg):
        if isinstance(arg, complex):
            return self.length() > arg.length()
        if isinstance(arg, int) or isinstance(arg, float) or isinstance(arg, Decimal) or isinstance(arg, Infinity): return self.length() > arg
        raise ArgTypeError()

    def __ge__(self, arg):
        if isinstance(arg, complex):
            return self.length() >= arg.length()
        if isinstance(arg, int) or isinstance(arg, float) or isinstance(arg, Decimal) or isinstance(arg, Infinity): return self.length() >= arg
        raise ArgTypeError()

    def __lt__(self, arg):
        if isinstance(arg, complex):
            return self.length() < arg.length()
        if isinstance(arg, int) or isinstance(arg, float) or isinstance(arg, Decimal) or isinstance(arg, Infinity): return self.length() < arg
        raise ArgTypeError()

    def __le__(self, arg):
        if isinstance(arg, complex):
            return self.length() <= arg.length()
        if isinstance(arg, int) or isinstance(arg, float) or isinstance(arg, Decimal) or isinstance(arg, Infinity): return self.length() <= arg
        raise ArgTypeError()

    def __pow__(self, p):
        temp = 1
        for k in range(p):
            temp = temp * self
        return temp

    def conjugate(self):
        return complex(self.real, -self.imaginary)

    def length(self):
        return (self * self.conjugate()).real

    def unit(self):
        return self / sqrt(self.length())

    def sqrt(arg, resolution: int = 200):
        if isinstance(arg, complex):
            temp = arg.unit()
            angle = arcsin(temp.imaginary, resolution=resolution) / 2
            return complex(cos(angle), sin(angle)) * sqrt(sqrt(arg.length()))
        raise ArgTypeError()

    # noinspection PyMethodFirstArgAssignment
    def range(lowreal, highreal, lowimg, highimg, step1=Decimal(1), step2=Decimal(1)):
        if not ((isinstance(lowreal, int) or isinstance(lowreal, float) or isinstance(lowreal, Decimal) or isinstance(lowreal, Infinity))
                and (isinstance(highreal, int) or isinstance(highreal, float) or isinstance(highreal, Decimal) or isinstance(highreal, Infinity))
                and (isinstance(lowimg, int) or isinstance(lowimg, float) or isinstance(lowimg, Decimal) or isinstance(lowimg, Infinity))
                and (isinstance(highimg, int) or isinstance(highimg, float) or isinstance(highimg, Decimal) or isinstance(highimg, Infinity))
                and (isinstance(step1, int) or isinstance(step1, float) or isinstance(step1, Decimal))
                and (isinstance(step2, int) or isinstance(step2, float) or isinstance(step2, Decimal))):
            raise ArgTypeError("Must be a numerical value.")
        if (highreal < lowreal ^ step1 > 0) or (highimg < lowimg ^ step2 > 0): raise RangeError()
        reset = lowimg
        while (not highreal < lowreal ^ step1 > 0) and not highreal == lowreal:
            lowimg = reset
            while (not highimg < lowimg ^ step2 > 0) and not highimg == lowimg:
                yield complex(lowreal, lowimg)
                lowimg += step2
            lowreal += step1

    def inverse(self):
        divisor = self.length()
        if divisor: return complex(self.real / divisor, -self.imaginary / divisor)
        else: return complex(Infinity(self.real >= 0), Infinity(-self.imaginary >= 0))

    def rotate(self, angle):
        return self * complex(cos(angle), sin(angle))

    def rotationFactor(self, angle):
        return complex(cos(angle), sin(angle))

class Infinity:
    def __init__(self, sign: bool = True):
        self.sign = sign

    def __str__(self):
        return f"Infinity({'positive' if self.sign else 'negative'})"

    def __repr__(self):
        return f"Infinity({'positive' if self.sign else 'negative'})"

    def __add__(self, arg):
        if isinstance(arg, int) or isinstance(arg, float) or isinstance(arg, Decimal): return Infinity(self.sign)
        if isinstance(arg, Infinity): return Undefined() if (self.sign ^ arg.sign) else Infinity(self.sign)
        if isinstance(arg, complex): return complex(Infinity(self.sign) + arg.real, arg.imaginary)
        if isinstance(arg, Undefined): return Undefined()
        raise ArgTypeError()

    def __radd__(self, arg):
        if isinstance(arg, int) or isinstance(arg, float) or isinstance(arg, Decimal): return Infinity(self.sign)
        if isinstance(arg, complex): return complex(Infinity(self.sign) + arg.real, arg.imaginary)
        raise ArgTypeError()

    def __iadd__(self, arg):
        if isinstance(arg, int) or isinstance(arg, float) or isinstance(arg, Decimal): return Infinity(self.sign)
        if isinstance(arg, Infinity): return Undefined() if (self.sign ^ arg.sign) else self
        if isinstance(arg, complex): return complex(Infinity(self.sign) + arg.real, arg.imaginary)
        if isinstance(arg, Undefined): return Undefined()
        raise ArgTypeError()

    def __sub__(self, arg):
        if isinstance(arg, int) or isinstance(arg, float) or isinstance(arg, Decimal): return Infinity(self.sign)
        if isinstance(arg, Infinity): return Undefined() if not (self.sign ^ arg.sign) else Infinity(self.sign)
        if isinstance(arg, complex): return complex(Infinity(self.sign) - arg.real, -arg.imaginary)
        if isinstance(arg, Undefined): return Undefined()
        raise ArgTypeError()

    def __rsub__(self, arg):
        if isinstance(arg, int) or isinstance(arg, float) or isinstance(arg, Decimal): return Infinity(not self.sign)
        if isinstance(arg, complex): return complex(-self + arg.real, arg.imaginary)
        raise ArgTypeError()

    def __isub__(self, arg):
        if isinstance(arg, int) or isinstance(arg, float) or isinstance(arg, Decimal): return Infinity(self.sign)
        if isinstance(arg, Infinity): return Undefined() if not (self.sign ^ arg.sign) else Infinity(self.sign)
        if isinstance(arg, complex): return complex(self - arg.real, -arg.imaginary)
        if isinstance(arg, Undefined): return Undefined()
        raise ArgTypeError()

    def __mul__(self, arg):
        if isinstance(arg, Infinity): return Infinity(not (self.sign ^ arg.sign))
        if isinstance(arg, int) or isinstance(arg, float) or isinstance(arg, Decimal): return Infinity(self.sign and arg > 0) if arg != 0 else Undefined()
        if isinstance(arg, complex): return complex(arg.real * self, arg.imaginary * self)
        if isinstance(arg, Undefined): return Undefined()
        raise ArgTypeError()

    def __rmul__(self, arg):
        if isinstance(arg, int) or isinstance(arg, float) or isinstance(arg, Decimal): return Infinity(self.sign and arg > 0) if arg != 0 else Undefined()
        if isinstance(arg, complex): return complex(arg.real * self, arg.imaginary * self)
        raise ArgTypeError()

    def __imul__(self, arg):
        if isinstance(arg, Infinity): return Infinity(not (self.sign ^ arg.sign))
        if isinstance(arg, int) or isinstance(arg, float) or isinstance(arg, Decimal): return Infinity(self.sign and arg > 0) if arg != 0 else Undefined()
        if isinstance(arg, complex): return complex(arg.real * self, arg.imaginary * self)
        if isinstance(arg, Undefined): return Undefined()
        raise ArgTypeError()

    def __truediv__(self, arg):
        if isinstance(arg, Infinity) or isinstance(arg, Undefined): return Undefined()
        if isinstance(arg, int) or isinstance(arg, float) or isinstance(arg, Decimal): return Infinity(self.sign ^ arg >= 0)
        if isinstance(arg, complex): return complex(Infinity(self.sign ^ arg.real >= 0), Infinity(self.sign ^ arg.imaginary >= 0))
        raise ArgTypeError()

    def __floordiv__(self, arg):
        if isinstance(arg, Infinity) or isinstance(arg, Undefined): return Undefined()
        if isinstance(arg, int) or isinstance(arg, float) or isinstance(arg, Decimal): return Infinity(self.sign ^ arg >= 0)
        if isinstance(arg, complex): return complex(Infinity(self.sign ^ arg.real >= 0), Infinity(self.sign ^ arg.imaginary >= 0))
        raise ArgTypeError()

    def __rtruediv__(self, arg):
        if isinstance(arg, Infinity): return Undefined()
        if isinstance(arg, int) or isinstance(arg, float) or isinstance(arg, Decimal): return 0
        raise ArgTypeError()

    def __rfloordiv__(self, arg):
        if isinstance(arg, Infinity): return Undefined()
        if isinstance(arg, int) or isinstance(arg, float) or isinstance(arg, Decimal): return 0
        raise ArgTypeError()

    def __idiv__(self, arg):
        if isinstance(arg, Infinity) or isinstance(arg, Undefined): return Undefined()
        if isinstance(arg, int) or isinstance(arg, float) or isinstance(arg, Decimal): return self
        if isinstance(arg, complex):
            temp = 1 / arg
            return complex(self * temp.real, self * temp.imaginary)
        raise ArgTypeError()

    def __ifloordiv__(self, arg):
        if isinstance(arg, Infinity) or isinstance(arg, Undefined): return Undefined()
        if isinstance(arg, int) or isinstance(arg, float) or isinstance(arg, Decimal): return self
        if isinstance(arg, complex):
            temp = 1 / arg
            return complex(self * temp.real, self * temp.imaginary)
        raise ArgTypeError()

    def __neg__(self):
        return Infinity(not self.sign)

    def __pow__(self, p):
        if not p: return Undefined()
        if not (isinstance(p, int) or isinstance(p, float) or isinstance(p, Decimal)): raise ArgTypeError("Must be a numerical value.")
        if p > 0: return Infinity(True) if self.sign else ((-1)**p) * Infinity(False)
        return 0

    def __invert__(self):
        # 0111111... -> 10000000... -> -0
        # 1111111... -> 00000000... -> +0
        return -0 if self.sign else 0

    def __eq__(self, arg):
        if isinstance(arg, Infinity): return not (self.sign ^ arg.sign)
        return False

    def __ne__(self, arg):
        if isinstance(arg, Infinity): return self.sign ^ arg.sign
        return True

    def __or__(self, arg):
        return True

    def __ior__(self, arg):
        return True

    def __ror__(self, arg):
        return True

    def __and__(self, arg):
        return True and arg

    def __iand__(self, arg):
        return True and arg

    def __rand__(self, arg):
        return True and arg

    def __xor__(self, arg):
        return True ^ arg

    def __ixor__(self, arg):
        return True ^ arg

    def __rxor__(self, arg):
        return True ^ arg

    def __gt__(self, arg):
        if isinstance(arg, Infinity): return (self.sign ^ arg.sign) and self.sign
        return self.sign
    def __ge__(self, arg):
        if isinstance(arg, Infinity): return (self.sign ^ arg.sign) and self.sign
        return self.sign

    def __lt__(self, arg):
        if isinstance(arg, Infinity): return not (self.sign ^ arg.sign) or ~self.sign
        return not self.sign

    def __le__(self, arg):
        if isinstance(arg, Infinity): return not (self.sign ^ arg.sign) or ~self.sign
        return not self.sign

class Undefined:

    def __str__(self):
        return "Undefined()"

    def __repr__(self):
        return "Undefined()"

    def __add__(self, other):
        return Undefined()

    def __radd__(self, other):
        return Undefined()

    def __iadd__(self, other):
        return self

    def __sub__(self, other):
        return Undefined()

    def __rsub__(self, other):
        return Undefined()

    def __isub__(self, other):
        return self

    def __mul__(self, other):
        return Undefined()

    def __rmul__(self, other):
        return Undefined()

    def __imul__(self, other):
        return self

    def __truediv__(self, other):
        return Undefined()

    def __floordiv__(self, other):
        return Undefined()

    def __rtruediv__(self, other):
        return Undefined()

    def __rfloordiv__(self, other):
        return Undefined()

    def __idiv__(self, other):
        return self

    def __ifloordiv__(self, other):
        return self

    def __neg__(self):
        return self

    def __invert__(self):
        return self

    # Normally boolean resulting operations are defined as "False"
    def __eq__(self, other):
        return False

    # nothing is equal to undefined
    def __ne__(self, other):
        return True

    def __gt__(self, other):
        return False

    def __ge__(self, other):
        return False

    def __lt__(self, other):
        return False

    def __le__(self):
        return False

    def __and__(self, other):
        return False

    def __rand__(self, other):
        return False

    def __iand__(self, other):
        return False

    # This is important
    def __or__(self, other):
        return other

    def __ror__(self, other):
        return other

    def __ior__(self, other):
        return other

    def __xor__(self, other):
        return False

    def __rxor__(self, other):
        return False

    def __ixor__(self, other):
        return False


