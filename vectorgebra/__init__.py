import math
import random
import threading
import logging

logger = logging.getLogger("root log")
handler = logging.StreamHandler()
format = logging.Formatter('%(asctime)s %(name)-12s %(levelname)-8s %(message)s')

handler.setFormatter(format)
logger.addHandler(handler)
logger.setLevel(logging.INFO)

PI = 3.14159265359
E = 2.718281828459
ln2 = 0.6931471805599569

results = {}

class MathArgError(Exception):
    def __init__(self, hint: str = ""):
        super().__init__(f"Argument elements are of the wrong type{(': ' + hint) if hint else ''}")

class MathRangeError(Exception):
    def __init__(self, hint: str = ""):
        super().__init__(f"Argument(s) out of range{(': ' + hint) if hint else ''}")

class DimensionError(Exception):
    def __init__(self, code):
        if code == 0:
            super().__init__("Dimensions must match")
        elif code == 1:
            super().__init__("Number of dimensions cannot be zero")
        elif code == 2:
            super().__init__("Matrix must be a square")

class ArgTypeError(Exception):
    def __init__(self, code):
        if code == "f":
            super().__init__("Args must be of type float")
        elif code == "vif":
            super().__init__("Arg must be of type Vector, int, float")
        elif code == "if":
            super().__init__("Arg must be of type int, float")
        elif code == "v":
            super().__init__("Arg must be of type Vector")
        elif code == "i":
            super().__init__("Args must be of type int")
        elif code == "ifm":
            super().__init__("Arg must be of type int, float, Matrix")
        elif code == "ifmv":
            super().__init__("Arg must be of type int, float, Matrix, Vector")
        elif code == "m":
            super().__init__("Arg must be of type Matrix")
        elif code == "im":
            super().__init__("Arg must be of type int, Matrix")

class ArgumentError(Exception):
    def __init__(self):
        super().__init__("Not the correct amount of args")

class RangeError(Exception):
    def __init__(self):
        super().__init__("Argument out of range")


class Vector:
    def __init__(self, *args):
        for k in args:
            if ((not isinstance(k, int)) and (not isinstance(k, float)) and (not isinstance(k, bool))
                    and (not isinstance(k, Infinity)) and (not isinstance(k, Undefined)) and (not isinstance(k, complex))):
                raise MathArgError("Arguments must be numeric or boolean.")
        self.dimension = len(args)
        self.values = [_ for _ in args]

    def __str__(self):
        return str(self.values)

    def __getitem__(self, index):
        return self.values[index]

    def __len__(self):
        return len(self.values)

    def __add__(self, arg):
        if isinstance(arg, int) or isinstance(arg, float) or isinstance(arg, complex) or isinstance(arg, Infinity) or isinstance(arg, Undefined):
            return Vector(*[self.values[k] + arg for k in range(0, self.dimension)])
        if not isinstance(arg, Vector):
            raise ArgTypeError("vif")
        if not (self.dimension == arg.dimension):
            raise DimensionError(0)
        return Vector(*[self.values[k] + arg.values[k] for k in range(0, self.dimension)])

    def __radd__(self, arg):
        if isinstance(arg, int) or isinstance(arg, float) or isinstance(arg, complex) or isinstance(arg, Infinity) or isinstance(arg, Undefined):
            return Vector(*[self.values[k] + arg for k in range(0, self.dimension)])
        raise ArgTypeError("vif")

    def __sub__(self, arg):
        if isinstance(arg, int) or isinstance(arg, float) or isinstance(arg, complex) or isinstance(arg, Infinity) or isinstance(arg, Undefined):
            return Vector(*[self.values[k] - arg for k in range(0, self.dimension)])
        if not isinstance(arg, Vector):
            raise ArgTypeError("vif")
        if not (self.dimension == arg.dimension):
            raise DimensionError(0)
        return Vector(*[self.values[k] - arg.values[k] for k in range(0, self.dimension)])

    def __rsub__(self, arg):
        if isinstance(arg, int) or isinstance(arg, float) or isinstance(arg, complex) or isinstance(arg, Infinity) or isinstance(arg, Undefined):
            return -Vector(*[self.values[k] - arg for k in range(0, self.dimension)])
        raise ArgTypeError("vif")

    def dot(self, arg):
        if not isinstance(arg, Vector):
            raise ArgTypeError("v")
        if not (self.dimension == arg.dimension):
            raise DimensionError(0)
        mul = [self.values[k] * arg.values[k] for k in range(0, self.dimension)]
        sum = 0
        for k in mul:
            sum += k
        return sum

    def __mul__(self, arg):
        if not (isinstance(arg, int) or isinstance(arg, float) or isinstance(arg, complex) or isinstance(arg, Infinity) or isinstance(arg, Undefined)):
            raise ArgTypeError("if")
        return Vector(*[self.values[k] * arg for k in range(0, self.dimension)])

    def __rmul__(self, arg):
        if not (isinstance(arg, int) or isinstance(arg, float) or isinstance(arg, complex) or isinstance(arg, Infinity) or isinstance(arg, Undefined)):
            raise ArgTypeError("if")
        return Vector(*[self.values[k] * arg for k in range(0, self.dimension)])

    def __truediv__(self, arg):
        if not (isinstance(arg, int) or isinstance(arg, float) or isinstance(arg, complex) or isinstance(arg, Infinity) or isinstance(arg, Undefined)):
            raise ArgTypeError("if")
        return Vector(*[self.values[k] / arg for k in range(0, self.dimension)])


    def __floordiv__(self, arg):
        if not (isinstance(arg, int) or isinstance(arg, float) or isinstance(arg, complex) or isinstance(arg, Infinity) or isinstance(arg, Undefined)):
            raise ArgTypeError("if")
        return Vector(*[self.values[k] // arg for k in range(0, self.dimension)])

    def __iadd__(self, arg):
        if isinstance(arg, int) or isinstance(arg, float) or isinstance(arg, complex) or isinstance(arg, Infinity) or isinstance(arg, Undefined):
            return Vector(*[self.values[k] + arg for k in range(0, self.dimension)])
        if not isinstance(arg, Vector):
            raise ArgTypeError("vif")
        if not (self.dimension == arg.dimension):
            raise DimensionError(0)
        return Vector(*[self.values[k] + arg.values[k] for k in range(0, self.dimension)])

    def __isub__(self, arg):
        if isinstance(arg, int) or isinstance(arg, float) or isinstance(arg, complex) or isinstance(arg, Infinity) or isinstance(arg, Undefined):
            return Vector(*[self.values[k] - arg for k in range(0, self.dimension)])
        if not isinstance(arg, Vector):
            raise ArgTypeError("vif")
        if not (self.dimension == arg.dimension):
            raise DimensionError(0)
        return Vector(*[self.values[k] - arg.values[k] for k in range(0, self.dimension)])

    def __gt__(self, arg):
        if isinstance(arg, int) or isinstance(arg, float) or isinstance(arg, complex) or isinstance(arg, Infinity) or isinstance(arg, Undefined):
            sum = 0
            for k in self.values:
                sum += k * k
            if sum > arg * arg:
                return True
            return False
        if not isinstance(arg, Vector):
            raise ArgTypeError("vif")
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
        if isinstance(arg, int) or isinstance(arg, float) or isinstance(arg, complex) or isinstance(arg, Infinity) or isinstance(arg, Undefined):
            sum = 0
            for k in self.values:
                sum += k * k
            if sum >= arg * arg:
                return True
            return False
        if not isinstance(arg, Vector):
            raise ArgTypeError("vif")
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
        if isinstance(arg, int) or isinstance(arg, float) or isinstance(arg, complex) or isinstance(arg, Infinity) or isinstance(arg, Undefined):
            sum = 0
            for k in self.values:
                sum += k * k
            if sum > arg * arg:
                return True
            return False
        if not isinstance(arg, Vector):
            raise ArgTypeError("vif")
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
        if isinstance(arg, int) or isinstance(arg, float) or isinstance(arg, complex) or isinstance(arg, Infinity) or isinstance(arg, Undefined):
            sum = 0
            for k in self.values:
                sum += k * k
            if sum <= arg * arg:
                return True
            return False
        if not isinstance(arg, Vector):
            raise ArgTypeError("vif")
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
        if isinstance(arg, int) or isinstance(arg, float) or isinstance(arg, complex) or isinstance(arg, Infinity) or isinstance(arg, Undefined):
            for k in self.values:
                if not (k == arg):
                    return False
            return True
        if not isinstance(arg, Vector):
            raise ArgTypeError("vif")
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
            raise ArgTypeError("v")
        if not (self.dimension == arg.dimension):
            raise DimensionError(0)
        return Vector(*[(self.values[k] and arg.values[k]) for k in range(0, self.dimension)])

    def __iand__(self, arg):
        if not isinstance(arg, Vector):
            raise ArgTypeError("v")
        if not (self.dimension == arg.dimension):
            raise DimensionError(0)
        return Vector(*[(self.values[k] and arg.values[k]) for k in range(0, self.dimension)])

    def __or__(self, arg):
        if not isinstance(arg, Vector):
            raise ArgTypeError("v")
        if not (self.dimension == arg.dimension):
            raise DimensionError(0)
        return Vector(*[(self.values[k] or arg.values[k]) for k in range(0, self.dimension)])

    def __ior__(self, arg):
        if not isinstance(arg, Vector):
            raise ArgTypeError("v")
        if not (self.dimension == arg.dimension):
            raise DimensionError(0)
        return Vector(*[(self.values[k] or arg.values[k]) for k in range(0, self.dimension)])

    def __xor__(self, arg):
        if not isinstance(arg, Vector):
            raise ArgTypeError("v")
        if not (self.dimension == arg.dimension):
            raise DimensionError(0)
        return Vector(*[(self.values[k] ^ arg.values[k]) for k in range(0, self.dimension)])

    def __ixor__(self, arg):
        if not isinstance(arg, Vector):
            raise ArgTypeError("v")
        if not (self.dimension == arg.dimension):
            raise DimensionError(0)
        return Vector(*[(self.values[k] ^ arg.values[k]) for k in range(0, self.dimension)])

    def __invert__(self):
        return Vector(*[int(not self.values[k]) for k in range(0, self.dimension)])

    def append(self, arg):
        if isinstance(arg, int) or isinstance(arg, float) or isinstance(arg, complex) or isinstance(arg, Infinity) or isinstance(arg, Undefined):
            self.values.append(arg)
            self.dimension += 1
            return
        if not isinstance(arg, Vector):
            raise ArgTypeError("vif")
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
        return math.sqrt(sum)

    def proj(self, arg):
        if not isinstance(arg, Vector):
            raise ArgTypeError("v")
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
                raise ArgTypeError("v")
            if not (k.dimension == (len(args))):
                raise ArgumentError
            v_list.append(k)
        for k in range(1, len(v_list)):
            temp = v_list[k]
            for l in range(0, k):
                temp -= v_list[k].proj(v_list[l])
            v_list[k] = temp.unit()
        return v_list

    def does_span(*args):
        v_list = Vector.spanify(*args)
        for k in range(0, len(v_list)):
            for l in range(0, len(v_list)):
                if not v_list[k].dot(v_list[l]) < 0.0000000001 and not k == l:
                    return False
        return True

    def randVint(dim: int, a: int, b: int):
        if not (isinstance(dim, int) and isinstance(a, int) and isinstance(b, int)):
            raise ArgTypeError("i")
        if not (dim > 0):
            raise RangeError
        return Vector(*[random.randint(a, b) for k in range(0, dim)])

    def randVfloat(dim: int, a: float, b: float):
        if not (isinstance(dim, int) and (isinstance(a, int) or isinstance(a, float)) and (isinstance(b, int) or isinstance(b, float))):
            raise ArgTypeError("if")
        if not (dim > 0):
            raise RangeError
        return Vector(*[random.uniform(a, b) for k in range(0, dim)])

    def randVbool(dim: int):
        if not isinstance(dim, int): raise ArgTypeError("i")
        if not (dim > 0): raise RangeError
        return Vector(*[random.randrange(0, 2) for k in range(0, dim)])

    def determinant(*args):
        for k in args:
            if not isinstance(k, Vector): raise ArgTypeError("v")
            if not (args[0].dimension == k.dimension): raise DimensionError(0)
        if not (len(args) == args[0].dimension): raise ArgumentError

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
            if not isinstance(k, Vector): raise ArgTypeError("v")
            if not (args[0].dimension == k.dimension): raise DimensionError(0)

        if len(args) == 2 and args[0].dimension == 2:
            return args[0].values[0] * args[1].values[1] - args[0].values[1] * args[1].values[0]
        if not (len(args) == args[0].dimension - 1): raise ArgumentError

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

    def cumsum(self):
        sum = 0
        for k in self.values:
            sum += k
        return sum

    def zero(dim: int):
        if dim < 0: raise RangeError()
        return Vector(*[0 for k in range(dim)])

    def one(dim: int):
        if dim < 0: raise RangeError()
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


class Matrix:
    def __init__(self, *args):
        for k in args:
            if not isinstance(k, Vector):
                raise ArgTypeError("v")
            if not (args[0].dimension == k.dimension):
                raise DimensionError(0)
        self.values = [k.values for k in args]
        self.dimension = f"{args[0].dimension}x{len(args)}"
        self.string = [str(k) for k in self.values]
        self.string = "\n".join(self.string)


    def __str__(self):
        return self.string

    def __getitem__(self, index):
        return self.values[index]

    def __add__(self, arg):
        v = list()
        if isinstance(arg, int) or isinstance(arg, float) or isinstance(arg, complex) or isinstance(arg, Infinity) or isinstance(arg, Undefined):
            for k in self.values:
                v.append(Vector(*[l + arg for l in k]))
            return Matrix(*v)
        if not isinstance(arg, Matrix):
            raise ArgTypeError("ifm")
        if not (self.dimension == arg.dimension):
            raise DimensionError(0)
        for k in range(0, len(self.values)):
            m = list()
            for l in range(0, len(self.values[0])):
                m.append(self.values[k][l] + arg.values[k][l])
            v.append(m)
        return Matrix(*[Vector(*k) for k in v])

    def __radd__(self, arg):
        v = list()
        if isinstance(arg, int) or isinstance(arg, float) or isinstance(arg, complex) or isinstance(arg, Infinity) or isinstance(arg, Undefined):
            for k in self.values:
                v.append(Vector(*[l + arg for l in k]))
            return Matrix(*v)
        raise ArgTypeError("ifm")

    def __iadd__(self, arg):
        v = list()
        if isinstance(arg, int) or isinstance(arg, float) or isinstance(arg, complex) or isinstance(arg, Infinity) or isinstance(arg, Undefined):
            for k in self.values:
                v.append(Vector(*[l + arg for l in k]))
            return Matrix(*v)
        if not isinstance(arg, Matrix):
            raise ArgTypeError("ifm")
        if not (self.dimension == arg.dimension):
            raise DimensionError(0)
        for k in range(0, len(self.values)):
            m = list()
            for l in range(0, len(self.values[0])):
                m.append(self.values[k][l] + arg.values[k][l])
            v.append(m)
        return Matrix(*[Vector(*k) for k in v])

    def __sub__(self, arg):
        v = list()
        if isinstance(arg, int) or isinstance(arg, float) or isinstance(arg, complex) or isinstance(arg, Infinity) or isinstance(arg, Undefined):
            for k in self.values:
                v.append(Vector(*[l - arg for l in k]))
            return Matrix(*v)
        if not isinstance(arg, Matrix):
            raise ArgTypeError("ifm")
        if not (self.dimension == arg.dimension):
            raise DimensionError(0)
        for k in range(0, len(self.values)):
            m = list()
            for l in range(0, len(self.values[0])):
                m.append(self.values[k][l] - arg.values[k][l])
            v.append(m)
        return Matrix(*[Vector(*k) for k in v])

    def __rsub__(self, arg):
        v = list()
        if isinstance(arg, int) or isinstance(arg, float) or isinstance(arg, complex) or isinstance(arg, Infinity) or isinstance(arg, Undefined):
            for k in self.values:
                v.append(Vector(*[l - arg for l in k]))
            return -Matrix(*v)
        raise ArgTypeError("ifm")

    def __isub__(self, arg):
        v = list()
        if isinstance(arg, int) or isinstance(arg, float) or isinstance(arg, complex) or isinstance(arg, Infinity) or isinstance(arg, Undefined):
            for k in self.values:
                v.append(Vector(*[l - arg for l in k]))
            return Matrix(*v)
        if not (type(arg) == Matrix):
            raise ArgTypeError("ifm")
        if not (self.dimension == arg.dimension):
            raise DimensionError(0)
        for k in range(0, len(self.values)):
            m = list()
            for l in range(0, len(self.values[0])):
                m.append(self.values[k][l] - arg.values[k][l])
            v.append(m)
        return Matrix(*[Vector(*k) for k in v])

    def __mul__(self, arg):
        v = list()
        if isinstance(arg, int) or isinstance(arg, float) or isinstance(arg, complex) or isinstance(arg, Infinity) or isinstance(arg, Undefined):
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
            raise ArgTypeError("ifmv")
        if not (self.dimension.split("x")[1] == arg.dimension.split("x")[0]):
            raise DimensionError(0)
        for k in range(0, len(self.values)):
            n = list()
            for l in range(0, len(arg.values[0])):
                sum = 0
                for m in range(0, len(arg.values)):
                    sum += self.values[k][m] * arg.values[m][l]
                n.append(sum)
            v.append(n)
        return Matrix(*[Vector(*k) for k in v])

    def __rmul__(self, arg):
        v = list()
        if isinstance(arg, int) or isinstance(arg, float) or isinstance(arg, complex) or isinstance(arg, Infinity) or isinstance(arg, Undefined):
            for k in self.values:
                v.append(Vector(*[l * arg for l in k]))
            return Matrix(*v)
        raise ArgTypeError("ifmv")

    def __neg__(self):
        return Matrix(*[Vector(*[-l for l in k]) for k in self.values])

    def __truediv__(self, arg):
        v = list()
        if not (isinstance(arg, int) or isinstance(arg, float) or isinstance(arg, complex) or isinstance(arg, Infinity) or isinstance(arg, Undefined)):
            raise ArgTypeError("if")
        for k in self.values:
            v.append(Vector(*[l / arg for l in k]))
        return Matrix(*v)

    def __floordiv__(self, arg):
        v = list()
        if not (isinstance(arg, int) or isinstance(arg, float) or isinstance(arg, complex) or isinstance(arg, Infinity) or isinstance(arg, Undefined)):
            raise ArgTypeError("if")
        for k in self.values:
            v.append(Vector(*[l // arg for l in k]))
        return Matrix(*v)

    def __pow__(self, p):
        temp = Matrix.identity(len(self.values))
        for k in range(p):
            temp *= self
        return temp

    def determinant(arg):
        if not isinstance(arg, Matrix):
            raise ArgTypeError("m")
        if arg.dimension == "1x1":
            return arg.values[0][0]
        return Vector.determinant(*[Vector(*k) for k in arg.values])

    def __or__(self, arg):
        v = list()
        if not isinstance(arg, Matrix):
            raise ArgTypeError("ifm")
        if not (self.dimension == arg.dimension):
            raise DimensionError(0)
        for k in range(0, len(self.values)):
            m = list()
            for l in range(0, len(self.values[k])):
                m.append(self.values[k][l] or arg.values[k][l])
            v.append(Vector(*m))
        return Matrix(*v)

    def __ior__(self, arg):
        v = list()
        if not isinstance(arg, Matrix):
            raise ArgTypeError("m")
        if not (self.dimension == arg.dimension):
            raise DimensionError(0)
        for k in range(0, len(self.values)):
            m = list()
            for l in range(0, len(self.values[k])):
                m.append(self.values[k][l] or arg.values[k][l])
            v.append(Vector(*m))
        return Matrix(*v)

    def __and__(self, arg):
        v = list()
        if not isinstance(arg, Matrix):
            raise ArgTypeError("m")
        if not (self.dimension == arg.dimension):
            raise DimensionError(0)
        for k in range(0, len(self.values)):
            m = list()
            for l in range(0, len(self.values[k])):
                m.append(self.values[k][l] and arg.values[k][l])
            v.append(Vector(*m))
        return Matrix(*v)

    def __iand__(self, arg):
        v = list()
        if not isinstance(arg, Matrix):
            raise ArgTypeError("m")
        if not (self.dimension == arg.dimension):
            raise DimensionError(0)
        for k in range(0, len(self.values)):
            m = list()
            for l in range(0, len(self.values[k])):
                m.append(self.values[k][l] and arg.values[k][l])
            v.append(Vector(*m))
        return Matrix(*v)

    def __xor__(self, arg):
        v = list()
        if not isinstance(arg, Matrix):
            raise ArgTypeError("m")
        if not (self.dimension == arg.dimension):
            raise DimensionError(0)
        for k in range(0, len(self.values)):
            m = list()
            for l in range(0, len(self.values[k])):
                m.append(self.values[k][l] ^ arg.values[k][l])
            v.append(Vector(*m))
        return Matrix(*v)

    def __ixor__(self, arg):
        v = list()
        if not isinstance(arg, Matrix):
            raise ArgTypeError("m")
        if not (self.dimension == arg.dimension):
            raise DimensionError(0)
        for k in range(0, len(self.values)):
            m = list()
            for l in range(0, len(self.values[k])):
                m.append(self.values[k][l] ^ arg.values[k][l])
            v.append(Vector(*m))
        return Matrix(*v)

    def __invert__(self):
        return Matrix(*[Vector(*[int(not l) for l in k]) for k in self.values])

    def __eq__(self, arg):
        if not isinstance(arg, Matrix):
            raise ArgTypeError("m")
        if self.values == arg.values:
            return True
        return False

    def append(self, arg):
        if not isinstance(arg, Vector):
            raise ArgTypeError("v")
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
        v = list()
        for k in range(0, len(self.values[0])):
            m = list()
            for l in range(0, len(self.values)):
                m.append(self.values[l][k])
            v.append(Vector(*m))
        return Matrix(*v)

    def inverse(self, method: str = "iterative", resolution: int = 10):
        if method not in ["gauss", "analytic", "iterative", "neumann"]: raise MathArgError()
        if resolution < 1: raise RangeError()
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
            i = Matrix.identity(len(self.values))
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
                        if abs(factor) < 0.0000000001:
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
                        if abs(factor) < 0.0000000001:
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
                        if abs(factor) < 0.0000000001:
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
                    iden_values[k] = list(map(lambda x: x if (abs(x) > 0.00000001) else 0,
                                              [iden_values[k][l] / v[k][k] for l in range(len(self.values[0]))]))

                except ZeroDivisionError:
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

            identity = Matrix.identity(len(self.values)) * 2

            for k in range(resolution):
                guess = guess * (identity - self * guess)
                #guess = guess * 2 - guess * self * guess
            return guess

        elif method == "neumann":
            # dont forget to calibrate the resolution here
            sum = Matrix.zero(len(self.values))
            block = Matrix.identity(len(self.values)) - self

            for k in range(resolution):
                temp = Matrix.identity(len(self.values))
                for l in range(k):
                    temp = temp * block
                sum += temp

            return sum

    def identity(dim: int):
        if dim <= 0:
            raise RangeError()
        v = list()
        for k in range(0, dim):
            temp = [0] * dim
            temp[k] = 1
            v.append(Vector(*temp))
        return Matrix(*v)

    def zero(dim: int):
        if dim <= 0:
            raise RangeError()
        v = list()
        for k in range(0, dim):
            temp = [0] * dim
            v.append(Vector(*temp))
        return Matrix(*v)

    def one(dim: int):
        if dim <= 0:
            raise RangeError()
        v = list()
        for k in range(0, dim):
            temp = [1] * dim
            v.append(Vector(*temp))
        return Matrix(*v)

    def randMint(m: int, n: int, a: int, b: int):
        if m <= 0 or n <= 0:
            raise RangeError
        v = list()
        for k in range(0, m):
            temp = list()
            for l in range(0, n):
                temp.append(random.randint(a, b))
            v.append(Vector(*temp))
        return Matrix(*v)

    def randMfloat(m: int, n: int, a: float, b: float):
        if m <= 0 or n <= 0:
            raise RangeError
        v = list()
        for k in range(0, m):
            temp = list()
            for l in range(0, n):
                temp.append(random.uniform(a, b))
            v.append(Vector(*temp))
        return Matrix(*v)

    def randMbool(m: int, n: int):
        if m <= 0 or n <= 0:
            raise RangeError
        v = list()
        for k in range(0, m):
            temp = list()
            for l in range(0, n):
                temp.append(random.randint(0, 1))
            v.append(Vector(*temp))
        return Matrix(*v)

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

    def det_echelon(self):
        a = self.echelon()
        if not a.dimension.split("x")[0] == a.dimension.split("x")[1]:
            return
        sum = 1
        for k in range(0, len(a.values)):
            sum *= a.values[k][k]
        return sum

    def fast_inverse(self):
        i = Matrix.identity(len(self.values))
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
                    if abs(factor) < 0.0000000001:
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
                    if abs(factor) < 0.0000000001:
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
                    if abs(factor) < 0.0000000001:
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
                iden_values[k] = list(map(lambda x: x if (abs(x) > 0.00000001) else 0,
                                          [iden_values[k][l] / v[k][k] for l in range(len(self.values[0]))]))

            except ZeroDivisionError:
                pass

        return Matrix(*[Vector(*k) for k in iden_values])

    def cramer(a, number: int):
        if not isinstance(a, Matrix):
            raise ArgTypeError("im")
        if not number < len(a.values[0]) - 1 or number < 0:
            raise RangeError
        v = list()
        for k in range(0, len(a.values)):
            m = list()
            for l in range(0, len(a.values[0]) - 1):
                if not l == number:
                    m.append(a.values[k][l])
                else:
                    m.append(a.values[k][len(a.values[0]) - 1])
            v.append(Vector(*m))
        first = Matrix(*v).det_echelon()
        v.clear()
        for k in range(0, len(a.values)):
            m = list()
            for l in range(0, len(a.values[0]) - 1):
                m.append(a.values[k][l])
            v.append(Vector(*m))
        second = Matrix(*v).det_echelon()
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
        if not (0 < len(args) < 3): raise ArgumentError()
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

    def eigenvalue(self, resolution: int = 10):
        if self.dimension.split("x")[0] != self.dimension.split("x")[1]: raise DimensionError(2)
        if resolution < 1: raise RangeError()

        to_work = self.copy()
        for k in range(resolution):
            Q, R = to_work.qr()
            to_work = R * Q
        result = []
        for k in range(len(to_work.values)):
            result.append(to_work.values[k][k])

        return result

    def qr(self):
        if self.dimension.split("x")[0] != self.dimension.split("x")[1]: raise DimensionError(2)
        v_list = []
        for k in self.transpose():
            v_list.append(Vector(*k))
        if not Vector.does_span(*v_list):
            m = Matrix.zero(len(self.values))
            return m, m
        result_list = [k.unit() for k in Vector.spanify(*v_list)]
        Q = Matrix(*result_list).transpose()
        R = Q.transpose() * self
        return Q, R

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
        sum = 0
        for k in range(len(self.values)):
            sum *= self.values[k][k]
        return sum



def Range(low: int or float, high: int or float, step: float = 1):
    if not ((high < low) ^ (step > 0)): raise MathRangeError()

    """
    A lazy implementation for creating ranges.

    This works almost at the exact speed as built 
    in range() when given a function inside the 
    loop such as print().

    :param low: low limit
    :param high: high limit
    :return: yields
    """
    while (high < low) ^ (step > 0) and not high == low:
        yield low
        low += step

def abs(arg: int or float) -> int or float:
    """
    Basic absolute value function

    :param arg: float or int
    :return: Absolute value of the arg
    """
    return arg if (arg >= 0) else -arg

def sqrt(arg, resolution: int = 10):
    """
    Square root with Newton's method. Speed is very close to
    the math.sqrt(). This may be due to both complex number
    allowance and digit counting while loop, I don't know the
    built-in algorithm.

    :param arg: Number, can be negative
    :param resolution: Number of iterations
    :return: float or complex
    """
    if isinstance(arg, int) or isinstance(arg, float):
        if resolution < 1: raise MathRangeError("Resolution must be a positive integer")
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
    raise MathArgError()

def cumsum(arg: list or tuple) -> int or float:
    """
    Cumulative sum of iterables. To use with vectors,
    put vector.values as the argument.

    :param arg: Number valued iterable
    :return: Cumulative sum
    """
    sum: float = 0
    try:
        for k in arg:
            sum += k
        return sum
    except:
        raise MathArgError("Elements of arg must be numerical")

def __cumdiv(x: int or float, power: int) -> float:
    """
    This is for lossless calculation of Taylor
    series.

    :param x: Number
    :param power: power
    :return: Returns x^power/(power!)
    """
    result: float = 1
    for k in range(power, 0, -1):
        result *= x / k
    return result

def e(exponent: int or float, resolution: int = 15) -> float:
    """
    e^x function.

    :param exponent: x value
    :param resolution: Up to which exponent Taylor series will continue.
    :return: e^x
    """
    if resolution < 1: raise MathRangeError("Resolution must be a positive integer")
    sum = 1
    for k in range(resolution, 0, -1):
        sum += __cumdiv(exponent, k)
    return sum

def sin(angle: int or float, resolution: int = 15) -> float:
    """
    sin(x) using Taylor series. Input is in degrees.

    :param angle: degrees
    :param resolution: Up to which exponent Taylor series will continue.
    :return: sin(angle)
    """
    if not (resolution % 2) or resolution < 1: raise MathRangeError("Resolution must be a positive integer")

    radian: float = (2 * PI * (angle % 360 / 360)) % (2 * PI)
    result: float = 0

    for k in range(resolution, 0, -2):
        result = result + __cumdiv(radian, k) * pow(-1, (k - 1) / 2)
    return result

def cos(angle: int or float, resolution: int = 16) -> float:
    """
    cos(x) using Taylor series. Input is in degrees.

    :param angle: degrees
    :param resolution: Up to which exponent Taylor series will continue.
    :return: cos(angle)
    """
    if (resolution % 2) or resolution < 1: raise MathRangeError("Resolution must be a positive integer")
    radian: float = (2 * PI * (angle % 360 / 360)) % (2 * PI)
    result: float = 1

    for k in range(resolution, 0, -2):
        result = result + __cumdiv(radian, k) * pow(-1, k / 2)
    return result

def tan(angle: int or float, resolution: int = 16):
    """
    tan(x) using Taylor series. Input is in degrees.

    :param angle: degrees
    :param resolution: Up to which exponent Taylor series will continue.
    :return: tan(angle)
    """
    if (resolution % 2) or resolution < 1: raise MathRangeError("Resolution must be a positive integer")
    try:
        return sin(angle, resolution - 1) / cos(angle, resolution)
        # Because of the error amount, probably cos will never be zero.
    except ZeroDivisionError:
        # sin * cos is positive in this area
        if 90 >= (angle % 360) >= 0 or 270 >= (angle % 360) >= 180: return Infinity()
        return Infinity(False)

def cot(angle: int or float, resolution: int = 16):
    """
    cot(x) using Taylor series. Input is in degrees.

    :param angle: degrees
    :param resolution: Up to which exponent Taylor series will continue.
    :return: cot(angle)
    """
    try:
        return 1 / tan(angle, resolution)
    except ZeroDivisionError:
        return None

def sinh(x: int or float, resolution: int = 15) -> float:
    return (e(x, resolution) - e(-x, resolution)) / 2

def cosh(x: int or float, resolution: int = 15) -> float:
    return (e(x, resolution) + e(-x, resolution)) / 2

def tanh(x: int or float, resolution: int = 15):
    try:
        return sinh(x, resolution) / cosh(x, resolution)
        # Indeed cosh is non-negative
    except ZeroDivisionError:
        return None

def coth(x: int or float, resolution: int = 15):
    try:
        return cosh(x, resolution) / sinh(x, resolution)
    except ZeroDivisionError:
        if x >= 0: return Infinity()
        return Infinity(False)

def arcsin(x: int or float, resolution: int = 20) -> float:
    """
    A Taylor series implementation of arcsin.

    :param x: sin(angle)
    :param resolution: Resolution of operation
    :return: angle
    """
    if not (-1 <= x <= 1): raise MathRangeError()
    if resolution < 1: raise MathRangeError("Resolution must be a positive integer")
    c = 1
    sol = x
    for k in range(1, resolution):
        c *= (2 * k - 1) / (2 * k)
        sol += c * pow(x, 2 * k + 1) / (2 * k + 1)
    return sol * 360 / (2 * PI)

def arccos(x: int or float, resolution: int = 20) -> float:
    if not (-1 <= x <= 1): raise MathRangeError()
    if resolution < 1: raise MathRangeError("Resolution must be a positive integer")

    return 90 - arcsin(x, resolution)

def __find(f, low: int or float, high: int or float, search_step: int or float, res: float = 0.0001):
    global results
    last_sign: bool = True if (f(low) >= 0) else False
    for x in Range(low, high, search_step):
        value = f(x)
        temp_sign = True if (value >= 0) else False
        if temp_sign and last_sign and (value // 1 != 0 or value // 1 != -1):
            last_sign = temp_sign
        elif abs(value) < res * 100:
            results[x] = True
            return x
        elif search_step > res:
            get = __find(f, x - search_step, x, search_step / 10)
            if get is not None:
                results[get] = True
                return get

def solve(f, low: int or float = -50, high: int or float = 50, search_step: int or float = 0.1,
          res: float = 0.0001) -> list:
    """
    Solves the equation f(x) = 0. Utilizes thread pool and some logging.

    :param f: Function
    :param low: Low limit for search
    :param high: High limit for search
    :param search_step: Search steps for the range
    :param res: Search resolution
    :return: List of zeroes (may be incomplete)
    """
    # I couldn't find any other way to check it
    if str(type(f)) != "<class 'function'>" and str(
        type(f)) != "<class 'builtin_function_or_method'>": raise MathArgError("f must be a callable")

    if high <= low: raise MathRangeError()
    if search_step <= 0: raise MathRangeError()
    if res <= 0 or res >= 1: raise MathRangeError()

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
            last_sign = temp_sign

    logger.debug("Main loop end reached, waiting for joins.")
    for k in thread_list:
        k.join()
    logger.debug("Joins ended.")
    for k in results:
        zeroes.append(k)

    results.clear()

    zeroes = list(map(lambda x: x if (abs(x) > 0.00001) else 0, zeroes))

    return zeroes

def derivative(f, x: int or float, h: float = 0.0000000001) -> float:
    if str(type(f)) != "<class 'function'>" and str(
        type(f)) != "<class 'builtin_function_or_method'>": raise MathArgError("f must be a callable")

    return (f(x + h) - f(x)) / h

def integrate(f, a: int or float, b: int or float, delta: float = 0.01) -> float:
    if str(type(f)) != "<class 'function'>" and str(
        type(f)) != "<class 'builtin_function_or_method'>": raise MathArgError("f must be a callable")

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
    """
    This function is for efficient multiplication of matrices (and vectors).
    It makes use of threads. Efficiency is depended on number of rows.

    :param m1: Matrix
    :param m2: Matrix or Vector
    :return: m1 * m2
    """
    if not (isinstance(m1, Matrix) and isinstance(m2, Matrix)): raise MathArgError()
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

def findsol(f, x: int = 0, resolution: int = 15) -> float:
    """
    Finds a singular solution at a time with Newton's method.

    :param f: Function
    :param x: Starting value
    :param resolution: Number of iterations
    :return: The found/guessed solution of the function
    """
    if str(type(f)) != "<class 'function'>" and str(
        type(f)) != "<class 'builtin_function_or_method'>": raise MathArgError("f must be a callable")
    if resolution < 1: raise MathRangeError("Resolution must be a positive integer")

    for k in range(resolution):
        try:
            x = x - (f(x) / derivative(f, x))
        except ZeroDivisionError:
            if f(x) >= 0:
                x = Infinity(False)
            else:
                x = Infinity()

    return x

def mean(arg) -> float:
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
    raise MathArgError()

def expectation(values, probabilities, moment: int = 1) -> float:
    if moment < 0: raise MathRangeError()
    if (isinstance(values, list) or isinstance(values, tuple) or isinstance(values, Vector)) \
        and (isinstance(probabilities, list) or isinstance(probabilities, tuple) or isinstance(probabilities, Vector)):
        if len(values) != len(probabilities): raise DimensionError(0)

        sum = 0
        for k in range(len(values)):
            sum += (values[k]**moment) * probabilities[k]
        return sum
    raise MathArgError("Arguments must be one dimensional iterables")

def variance(values, probabilities) -> float:
    if (isinstance(values, list) or isinstance(values, tuple) or isinstance(values, Vector)) \
            and (
            isinstance(probabilities, list) or isinstance(probabilities, tuple) or isinstance(probabilities, Vector)):
        if len(values) != len(probabilities): raise DimensionError(0)

        sum = 0
        for k in range(len(values)):
            sum += (values[k]**2) * probabilities[k] - values[k] * probabilities[k]
        return sum
    raise MathArgError("Arguments must be one dimensional iterables")

def sd(values, probabilities) -> float:
    if (isinstance(values, list) or isinstance(values, tuple) or isinstance(values, Vector)) \
            and (
            isinstance(probabilities, list) or isinstance(probabilities, tuple) or isinstance(probabilities, Vector)):
        if len(values) != len(probabilities): raise DimensionError(0)

        sum = 0
        for k in range(len(values)):
            sum += (values[k]**2) * probabilities[k] - values[k] * probabilities[k]
        return sqrt(sum)
    raise MathArgError("Arguments must be one dimensional iterables")

def factorial(x: int = 0) -> int:
    if x < 0: raise MathRangeError()
    if x <= 1: return 1
    return x * factorial(x - 1)

def permutation(x: int = 1, y: int = 1) -> int:
    if x < 1 or y < 1 or y > x: raise MathRangeError()
    result = 1
    for v in range(y + 1, x + 1):
        result *= v
    return result

def combination(x: int = 0, y: int = 0) -> int:
    if x < 0 or y < 0 or y > x: raise MathRangeError()
    result = 1
    count = 1
    for v in range(y + 1, x + 1):
        result *= v / count
        count += 1
    return result

def multinomial(n: int = 0, *args) -> int:
    sum = 0
    for k in args:
        if not isinstance(k, int): raise MathArgError()
        if k < 0: raise MathRangeError()
        sum += k
    if sum != n: raise MathRangeError("Sum of partitions must be equal to n")
    result = 1
    while n != 1:
        result *= n
        for k in args:
            result /= k
            k -= 1
        n -= 1
    return result

def binomial(n: int, k: int, p: float) -> float:
    if p < 0 or p > 1: raise MathRangeError("Probability cannot be negative or bigger than 1")
    return combination(n, k) * pow(p, k) * pow(1 - p, k)

def geometrical(n: int, p: float):
    if p < 0 or p > 1: raise MathRangeError("Probability cannot be negative or bigger than 1")
    if n < 0: raise MathRangeError("Trial number cannot be negative")
    if n == 0: return Undefined()
    return p * pow(1 - p, n - 1)

def poisson(k: int or float, l: int or float) -> float:
    if l < 0 or k < 0: raise MathRangeError()
    return pow(l, k) * e(-l) / factorial(k)

def linear_fit(x, y, rate: int or float = 0.01, iterations: int = 15) -> tuple:
    if not (isinstance(x, list) or isinstance(x, tuple) or isinstance(x, Vector))\
            and (isinstance(y, list) or isinstance(y, tuple) or isinstance(y, Vector)): raise MathArgError("Arguments must be one dimensional iterables")

    if iterations < 1: raise MathRangeError()
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

def general_fit(x, y, rate: int or float = 0.0000002, iterations: int = 15, degree: int = 1) -> Vector:
    if (not (isinstance(x, list) or isinstance(x, tuple) or isinstance(x, Vector))
            and (isinstance(y, list) or isinstance(y, tuple) or isinstance(y, Vector))): raise MathArgError("Arguments must be one dimensional iterables")
    if iterations < 1 or degree < 1: raise MathRangeError()
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



class complex:

    def __init__(self, real=0, imaginary=0):
        # Initialization is erroneous when expressions with infinities result in NoneTypes
        if ((not isinstance(real, int)) and (not isinstance(real, float)) and (not isinstance(imaginary, int))
                and (not isinstance(imaginary, float)) and (not isinstance(real, Infinity))
                and (not isinstance(imaginary, Infinity)) and (not isinstance(real, Undefined)) and (not isinstance(imaginary, Undefined))): raise MathArgError()
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
                and (not isinstance(arg, Infinity)) and (not isinstance(arg, Undefined))): raise MathArgError()

        if isinstance(arg, complex):
            return complex(self.real + arg.real, self.imaginary + arg.imaginary)
        return complex(self.real + arg, self.imaginary)

    def __radd__(self, arg):
        if (not isinstance(arg, int)) and (not isinstance(arg, float)) and (not isinstance(arg, Infinity)) and (not isinstance(arg, Undefined)): raise MathArgError()
        return complex(self.real + arg, self.imaginary)

    def __iadd__(self, arg):
        if ((not isinstance(arg, int)) and (not isinstance(arg, float)) and (not isinstance(arg, complex))
                and (not isinstance(arg, Infinity)) and (not isinstance(arg, Undefined))): raise MathArgError()

        if isinstance(arg, complex):
            self.real += arg.real
            self.imaginary += arg.imaginary
            return self
        self.real += arg
        return self

    def __sub__(self, arg):
        if ((not isinstance(arg, int)) and (not isinstance(arg, float)) and (not isinstance(arg, complex))
                and (not isinstance(arg, Infinity)) and (not isinstance(arg, Undefined))): raise MathArgError()

        if isinstance(arg, complex):
            return complex(self.real - arg.real, self.imaginary - arg.imaginary)
        return complex(self.real - arg, self.imaginary)

    def __rsub__(self, arg):
        if (not isinstance(arg, int)) and (not isinstance(arg, float)) and (not isinstance(arg, Infinity)) and (not isinstance(arg, Undefined)): raise MathArgError()
        return -complex(self.real - arg, self.imaginary)

    def __isub__(self, arg):
        if ((not isinstance(arg, int)) and (not isinstance(arg, float)) and (not isinstance(arg, complex))
                and (not isinstance(arg, Infinity)) and (not isinstance(arg, Undefined))): raise MathArgError()

        if isinstance(arg, complex):
            self.real -= arg.real
            self.imaginary -= arg.imaginary
            return self
        self.real -= arg
        return self

    def __mul__(self, arg):
        if ((not isinstance(arg, int)) and (not isinstance(arg, float)) and (not isinstance(arg, complex))
                and (not isinstance(arg, Infinity)) and (not isinstance(arg, Undefined))): raise MathArgError()

        if isinstance(arg, complex):
            return complex(self.real * arg.real - self.imaginary * arg.imaginary,
                           self.real * arg.imaginary + self.imaginary * arg.real)
        return complex(self.real * arg, self.imaginary * arg)

    def __rmul__(self, arg):
        if (not isinstance(arg, int)) and (not isinstance(arg, float)) and (not isinstance(arg, Infinity)) and (not isinstance(arg, Undefined)): raise MathArgError()
        return complex(self.real * arg, self.imaginary * arg)

    def __imul__(self, arg):
        if ((not isinstance(arg, int)) and (not isinstance(arg, float)) and (not isinstance(arg, complex))
                and (not isinstance(arg, Infinity)) and (not isinstance(arg, Undefined))): raise MathArgError()

        if isinstance(arg, complex):
            self.real = self.real * arg.real - self.imaginary * arg.imaginary
            self.imaginary = self.real * arg.imaginary + self.imaginary * arg.real
            return self
        self.real *= arg
        self.imaginary *= arg
        return self

    def __truediv__(self, arg):
        if ((not isinstance(arg, int)) and (not isinstance(arg, float)) and (not isinstance(arg, complex))
                and (not isinstance(arg, Infinity)) and (not isinstance(arg, Undefined))): raise MathArgError()

        if isinstance(arg, complex):
            return self * arg.inverse()
        if arg: return complex(self.real / arg, self.imaginary / arg)
        return complex(Infinity(self.real >= 0), Infinity(self.imaginary >= 0))

    def __rtruediv__(self, arg):
        if (not isinstance(arg, int)) and (not isinstance(arg, float)) and (not isinstance(arg, Infinity)) and (not isinstance(arg, Undefined)): raise MathArgError()
        return arg * self.inverse()


    def __idiv__(self, arg):
        if ((not isinstance(arg, int)) and (not isinstance(arg, float)) and (not isinstance(arg, complex))
                and (not isinstance(arg, Infinity)) and (not isinstance(arg, Undefined))): raise MathArgError()

        if isinstance(arg, complex):
            temp = self * arg.inverse()
            self.real, self.imaginary = temp.real, temp.imaginary
            return self
        self.real /= arg
        self.imaginary /= arg
        return self

    def __floordiv__(self, arg):
        if ((not isinstance(arg, int)) and (not isinstance(arg, float)) and (not isinstance(arg, complex))
                and (not isinstance(arg, Infinity)) and (not isinstance(arg, Undefined))): raise MathArgError()

        if isinstance(arg, complex):
            temp = self * arg.inverse()
            return complex(temp.real // 1, temp.imaginary // 1)
        return complex(self.real // arg, self.imaginary // arg)

    def __ifloordiv__(self, arg):
        if ((not isinstance(arg, int)) and (not isinstance(arg, float)) and (not isinstance(arg, complex))
                and (not isinstance(arg, Infinity)) and (not isinstance(arg, Undefined))): raise MathArgError()

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
        if isinstance(arg, int) or isinstance(arg, float): return self.length() > arg
        raise MathArgError()

    def __ge__(self, arg):
        if isinstance(arg, complex):
            return self.length() >= arg.length()
        if isinstance(arg, int) or isinstance(arg, float): return self.length() >= arg
        raise MathArgError()

    def __lt__(self, arg):
        if isinstance(arg, complex):
            return self.length() < arg.length()
        if isinstance(arg, int) or isinstance(arg, float): return self.length() < arg
        raise MathArgError()

    def __le__(self, arg):
        if isinstance(arg, complex):
            return self.length() <= arg.length()
        if isinstance(arg, int) or isinstance(arg, float): return self.length() <= arg
        raise MathArgError()

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
        raise MathArgError()

    # noinspection PyMethodFirstArgAssignment
    def range(lowreal: int or float, highreal: int or float, lowimg: int or float, highimg: int or float,
              step1: float = 1, step2: float = 1):
        if (highreal < lowreal ^ step1 > 0) or (highimg < lowimg ^ step2 > 0): raise MathRangeError()
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

    def rotate(self, angle: int or float):
        return self * complex(cos(angle), sin(angle))

    def rotationFactor(self, angle: int or float):
        return complex(cos(angle), sin(angle))



class Infinity:
    def __init__(self, sign: bool = True):
        self.sign = sign

    def __str__(self):
        return f"Infinity({'positive' if self.sign else 'negative'})"

    def __repr__(self):
        return f"Infinity({'positive' if self.sign else 'negative'})"

    def __add__(self, arg):
        if isinstance(arg, int) or isinstance(arg, float): return Infinity(self.sign)
        if isinstance(arg, Infinity): return Undefined() if (self.sign ^ arg.sign) else Infinity(self.sign)
        if isinstance(arg, complex): return complex(Infinity(self.sign) + arg.real, arg.imaginary)
        if isinstance(arg, Undefined): return Undefined()
        raise MathArgError()

    def __radd__(self, arg):
        if isinstance(arg, int) or isinstance(arg, float): return Infinity(self.sign)
        if isinstance(arg, complex): return complex(Infinity(self.sign) + arg.real, arg.imaginary)
        raise MathArgError()

    def __iadd__(self, arg):
        if isinstance(arg, int) or isinstance(arg, float): return Infinity(self.sign)
        if isinstance(arg, Infinity): return Undefined() if (self.sign ^ arg.sign) else self
        if isinstance(arg, complex): return complex(Infinity(self.sign) + arg.real, arg.imaginary)
        if isinstance(arg, Undefined): return Undefined()
        raise MathArgError()

    def __sub__(self, arg):
        if isinstance(arg, int) or isinstance(arg, float): return Infinity(self.sign)
        if isinstance(arg, Infinity): return Undefined() if not (self.sign ^ arg.sign) else Infinity(self.sign)
        if isinstance(arg, complex): return complex(Infinity(self.sign) - arg.real, -arg.imaginary)
        if isinstance(arg, Undefined): return Undefined()
        raise MathArgError()

    def __rsub__(self, arg):
        if isinstance(arg, int) or isinstance(arg, float): return Infinity(not self.sign)
        if isinstance(arg, complex): return complex(-self + arg.real, arg.imaginary)
        raise MathArgError()

    def __isub__(self, arg):
        if isinstance(arg, int) or isinstance(arg, float): return Infinity(self.sign)
        if isinstance(arg, Infinity): return Undefined() if not (self.sign ^ arg.sign) else Infinity(self.sign)
        if isinstance(arg, complex): return complex(self - arg.real, -arg.imaginary)
        if isinstance(arg, Undefined): return Undefined()
        raise MathArgError()

    def __mul__(self, arg):
        if isinstance(arg, Infinity): return Infinity(not (self.sign ^ arg.sign))
        if isinstance(arg, int) or isinstance(arg, float): return Infinity(self.sign and arg > 0) if arg != 0 else Undefined()
        if isinstance(arg, complex): return complex(arg.real * self, arg.imaginary * self)
        if isinstance(arg, Undefined): return Undefined()
        raise MathArgError()

    def __rmul__(self, arg):
        if isinstance(arg, int) or isinstance(arg, float): return Infinity(self.sign and arg > 0) if arg != 0 else Undefined()
        if isinstance(arg, complex): return complex(arg.real * self, arg.imaginary * self)
        raise MathArgError()

    def __imul__(self, arg):
        if isinstance(arg, Infinity): return Infinity(not (self.sign ^ arg.sign))
        if isinstance(arg, int) or isinstance(arg, float): return Infinity(self.sign and arg > 0) if arg != 0 else Undefined()
        if isinstance(arg, complex): return complex(arg.real * self, arg.imaginary * self)
        if isinstance(arg, Undefined): return Undefined()
        raise MathArgError()

    def __truediv__(self, arg):
        if isinstance(arg, Infinity) or isinstance(arg, Undefined): return Undefined()
        if isinstance(arg, int) or isinstance(arg, float): return Infinity(self.sign ^ arg >= 0)
        if isinstance(arg, complex): return complex(Infinity(self.sign ^ arg.real >= 0), Infinity(self.sign ^ arg.imaginary >= 0))
        raise MathArgError()

    def __floordiv__(self, arg):
        if isinstance(arg, Infinity) or isinstance(arg, Undefined): return Undefined()
        if isinstance(arg, int) or isinstance(arg, float): return Infinity(self.sign ^ arg >= 0)
        if isinstance(arg, complex): return complex(Infinity(self.sign ^ arg.real >= 0), Infinity(self.sign ^ arg.imaginary >= 0))
        raise MathArgError()

    def __rtruediv__(self, arg):
        if isinstance(arg, Infinity): return Undefined()
        if isinstance(arg, int) or isinstance(arg, float): return 0
        raise MathArgError()

    def __rfloordiv__(self, arg):
        if isinstance(arg, Infinity): return Undefined()
        if isinstance(arg, int) or isinstance(arg, float): return 0
        raise MathArgError()

    def __idiv__(self, arg):
        if isinstance(arg, Infinity) or isinstance(arg, Undefined): return Undefined()
        if isinstance(arg, int) or isinstance(arg, float): return self
        if isinstance(arg, complex):
            temp = 1 / arg
            return complex(self * temp.real, self * temp.imaginary)
        raise MathArgError()

    def __ifloordiv__(self, arg):
        if isinstance(arg, Infinity) or isinstance(arg, Undefined): return Undefined()
        if isinstance(arg, int) or isinstance(arg, float): return self
        if isinstance(arg, complex):
            temp = 1 / arg
            return complex(self * temp.real, self * temp.imaginary)
        raise MathArgError()

    def __neg__(self):
        return Infinity(not self.sign)

    def __pow__(self, p: int or float):
        if not p: return Undefined()
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
        return self.sign

    def __le__(self, arg):
        if isinstance(arg, Infinity): return not (self.sign ^ arg.sign) or ~self.sign
        return self.sign

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



