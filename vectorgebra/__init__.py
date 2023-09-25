import math
import random
import threading
import logging
import time

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
    def __init__(self):
        super().__init__("Argument elements are of the wrong type")

class MathRangeError(Exception):
    def __init__(self):
        super().__init__("Argument(s) out of range")

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
        control_list = ["e-0", "e-1", "e-2", "e-3", "e-4", "e-5", "e-6", "e-7", "e-8", "e-9", "e+0", "e+1", "e+2", "e+3", "e+4", "e+5", "e+6", "e+7", "e+8", "e+9"]
        for k in args:
            a = str(k)
            if "e" in a:
                factor = False
                for k in range(0, len(control_list)):
                    factor = factor or (control_list[k] in a)
                if not factor:
                    if not (a.isnumeric()):
                        raise ArgTypeError("f")
            try:
                a = a.replace(".", "")
                a = a.replace("-", "")
                a = a.replace("e", "")
                a = a.replace("+", "")
            except:
                pass
            if not (a.isnumeric() or type(a) == bool):
                raise ArgTypeError("f")
        self.dimension = len(args)
        self.values = [_ for _ in args]

    def __str__(self):
        return str(self.values)

    def __add__(self, arg):
        if type(arg) == int or type(arg) == float:
            return Vector(*[self.values[k] + arg for k in range(0, self.dimension)])
        if not (type(arg) == Vector):
            raise ArgTypeError("vif")
        if not (self.dimension == arg.dimension):
            raise DimensionError(0)
        return Vector(*[self.values[k] + arg.values[k] for k in range(0, self.dimension)])

    def __sub__(self, arg):
        if type(arg) == int or type(arg) == float:
            return Vector(*[self.values[k] - arg for k in range(0, self.dimension)])
        if not (type(arg) == Vector):
            raise ArgTypeError("vif")
        if not (self.dimension == arg.dimension):
            raise DimensionError(0)
        return Vector(*[self.values[k] - arg.values[k] for k in range(0, self.dimension)])

    def dot(self, arg):
        if not (type(arg) == Vector):
            raise ArgTypeError("v")
        if not (self.dimension == arg.dimension):
            raise DimensionError(0)
        mul = [self.values[k] * arg.values[k] for k in range(0, self.dimension)]
        sum = 0
        for k in mul:
            sum += k
        return sum

    def __mul__(self, arg):
        if not (type(arg) == int or type(arg) == float):
            raise ArgTypeError("if")
        return Vector(*[self.values[k] * arg for k in range(0, self.dimension)])

    def __truediv__(self, arg):
        if not (type(arg) == int or type(arg) == float):
            raise ArgTypeError("if")
        return Vector(*[self.values[k] / arg for k in range(0, self.dimension)])

    def __floordiv__(self, arg):
        if not (type(arg) == int or type(arg) == float):
            raise ArgTypeError("if")
        return Vector(*[self.values[k] // arg for k in range(0, self.dimension)])

    def __iadd__(self, arg):
        if type(arg) == int or type(arg) == float:
            return Vector(*[self.values[k] + arg for k in range(0, self.dimension)])
        if not (type(arg) == Vector):
            raise ArgTypeError("vif")
        if not (self.dimension == arg.dimension):
            raise DimensionError(0)
        return Vector(*[self.values[k] + arg.values[k] for k in range(0, self.dimension)])

    def __isub__(self, arg):
        if type(arg) == int or type(arg) == float:
            return Vector(*[self.values[k] - arg for k in range(0, self.dimension)])
        if not (type(arg) == Vector):
            raise ArgTypeError("vif")
        if not (self.dimension == arg.dimension):
            raise DimensionError(0)
        return Vector(*[self.values[k] - arg.values[k] for k in range(0, self.dimension)])

    def __gt__(self, arg):
        if type(arg) == int or type(arg) == float:
            sum = 0
            for k in self.values:
                sum += k * k
            if sum > arg * arg:
                return True
            return False
        if not (type(arg) == Vector):
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
        if type(arg) == int or type(arg) == float:
            sum = 0
            for k in self.values:
                sum += k * k
            if sum >= arg * arg:
                return True
            return False
        if not (type(arg) == Vector):
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
        if type(arg) == int or type(arg) == float:
            sum = 0
            for k in self.values:
                sum += k * k
            if sum > arg * arg:
                return True
            return False
        if not (type(arg) == Vector):
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
        if type(arg) == int or type(arg) == float:
            sum = 0
            for k in self.values:
                sum += k * k
            if sum <= arg * arg:
                return True
            return False
        if not (type(arg) == Vector):
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
        if type(arg) == int or type(arg) == float:
            for k in self.values:
                if not (k == arg):
                    return False
            return True
        if not (type(arg) == Vector):
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
        if not (type(arg) == Vector):
            raise ArgTypeError("v")
        if not (self.dimension == arg.dimension):
            raise DimensionError(0)
        return Vector(*[(self.values[k] and arg.values[k]) for k in range(0, self.dimension)])

    def __iand__(self, arg):
        if not (type(arg) == Vector):
            raise ArgTypeError("v")
        if not (self.dimension == arg.dimension):
            raise DimensionError(0)
        return Vector(*[(self.values[k] and arg.values[k]) for k in range(0, self.dimension)])

    def __or__(self, arg):
        if not (type(arg) == Vector):
            raise ArgTypeError("v")
        if not (self.dimension == arg.dimension):
            raise DimensionError(0)
        return Vector(*[(self.values[k] or arg.values[k]) for k in range(0, self.dimension)])

    def __ior__(self, arg):
        if not (type(arg) == Vector):
            raise ArgTypeError("v")
        if not (self.dimension == arg.dimension):
            raise DimensionError(0)
        return Vector(*[(self.values[k] or arg.values[k]) for k in range(0, self.dimension)])

    def __xor__(self, arg):
        if not (type(arg) == Vector):
            raise ArgTypeError("v")
        if not (self.dimension == arg.dimension):
            raise DimensionError(0)
        return Vector(*[(self.values[k] ^ arg.values[k]) for k in range(0, self.dimension)])

    def __ixor__(self, arg):
        if not (type(arg) == Vector):
            raise ArgTypeError("v")
        if not (self.dimension == arg.dimension):
            raise DimensionError(0)
        return Vector(*[(self.values[k] ^ arg.values[k]) for k in range(0, self.dimension)])

    def __invert__(self):
        return Vector(*[int(not self.values[k]) for k in range(0, self.dimension)])

    def append(self, arg):
        if type(arg) == int or type(arg) == float:
            self.values.append(arg)
            self.dimension += 1
            return
        if not (type(arg) == Vector):
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
            raise RangeError
        popped = self.values.pop(ord)
        self.dimension -= 1
        return popped

    def length(self):
        sum = 0
        for k in self.values:
            sum += k*k
        return math.sqrt(sum)

    def proj(self, arg):
        if not (type(arg) == Vector):
            raise ArgTypeError("v")
        if not (self.dimension == arg.dimension):
            raise DimensionError(0)
        if not self.dimension:
            return 0
        dot = self.dot(arg)
        sum = 0
        for k in arg.values:
            sum += k*k
        dot = (dot/sum)
        res = Vector(*arg.values)
        return res * dot

    def unit(self):
        l = self.length()
        temp = [k/l for k in self.values]
        return Vector(*temp)

    def spanify(*args):
        v_list = list()
        for k in args:
            if not (type(k) == Vector):
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
        if not (type(dim) == int and type(a) == int and type(b) == int):
            raise ArgTypeError("i")
        if not (dim > 0):
            raise RangeError
        return Vector(*[random.randint(a, b) for k in range(0, dim)])

    def randVfloat(dim: int, a: float, b: float):
        if not (type(dim) == int and (type(a) == int or type(a) == float) and (type(b) == int or type(b) == float)):
            raise ArgTypeError("if")
        if not (dim > 0):
            raise RangeError
        return Vector(*[random.uniform(a, b) for k in range(0, dim)])

    def randVbool(dim: int):
        if not (type(dim) == int): raise ArgTypeError("i")
        if not (dim > 0): raise RangeError
        return Vector(*[random.randrange(0, 2) for k in range(0, dim)])

    def determinant(*args):
        for k in args:
            if not (type(k) == Vector): raise ArgTypeError("v")
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
            if not (type(k) == Vector): raise ArgTypeError("v")
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
            if not (type(k) == Vector):
                raise ArgTypeError("v")
            if not (args[0].dimension == k.dimension):
                raise DimensionError(0)
        self.values = [k.values for k in args]
        self.dimension = f"{args[0].dimension}x{len(args)}"
        self.string = [str(k) for k in self.values]
        self.string = "\n".join(self.string)


    def __str__(self):
        return self.string

    def __add__(self, arg):
        v = list()
        if type(arg) == int or type(arg) == float:
            for k in self.values:
                v.append(Vector(*[l + arg for l in k]))
            return Matrix(*v)
        if not (type(arg) == Matrix):
            raise ArgTypeError("ifm")
        if not (self.dimension == arg.dimension):
            raise DimensionError(0)
        for k in range(0, len(self.values)):
            m = list()
            for l in range(0, len(self.values[0])):
                m.append(self.values[k][l] + arg.values[k][l])
            v.append(m)
        return Matrix(*[Vector(*k) for k in v])

    def __iadd__(self, arg):
        v = list()
        if type(arg) == int or type(arg) == float:
            for k in self.values:
                v.append(Vector(*[l + arg for l in k]))
            return Matrix(*v)
        if not (type(arg) == Matrix):
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
        if type(arg) == int or type(arg) == float:
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

    def __isub__(self, arg):
        v = list()
        if type(arg) == int or type(arg) == float:
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
        if type(arg) == int or type(arg) == float:
            for k in self.values:
                v.append(Vector(*[l * arg for l in k]))
            return Matrix(*v)
        if type(arg) == Vector:
            if not (self.dimension.split("x")[0] == str(arg.dimension)):
                raise DimensionError(0)
            for k in range(0, len(self.values)):
                sum = 0
                for l in range(0, len(arg.values)):
                    sum += self.values[k][l] * arg.values[l]
                v.append(sum)
            return Vector(*v)

        if not (type(arg) == Matrix):
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

    def __neg__(self):
        return Matrix(*[Vector(*[-l for l in k]) for k in self.values])

    def __truediv__(self, arg):
        v = list()
        if not (type(arg) == int or type(arg) == float):
            raise ArgTypeError("if")
        for k in self.values:
            v.append(Vector(*[l / arg for l in k]))
        return Matrix(*v)

    def __floordiv__(self, arg):
        v = list()
        if not (type(arg) == int or type(arg) == float):
            raise ArgTypeError("if")
        for k in self.values:
            v.append(Vector(*[l // arg for l in k]))
        return Matrix(*v)

    def determinant(arg):
        if not (type(arg) == Matrix):
            raise ArgTypeError("m")
        if arg.dimension == "1x1":
            return arg.values[0][0]
        return Vector.determinant(*[Vector(*k) for k in arg.values])

    def __or__(self, arg):
        v = list()
        if not (type(arg) == Matrix):
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
        if not (type(arg) == Matrix):
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
        if not (type(arg) == Matrix):
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
        if not (type(arg) == Matrix):
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
        if not (type(arg) == Matrix):
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
        if not (type(arg) == Matrix):
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
        if not (type(arg) == Matrix):
            raise ArgTypeError("m")
        if self.values == arg.values:
            return True
        return False

    def append(self, arg):
        if not type(arg) == Vector:
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
            raise RangeError
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

    def inverse(self):
        if not (self.dimension.split("x")[0] == self.dimension.split("x")[1]):
            raise DimensionError(2)
        if self.dimension == "1x1":
            return (1/self.values[0][0])
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

    def identity(dim: int):
        if dim <= 0:
            raise RangeError
        v = list()
        for k in range(0, dim):
            temp = [0] * dim
            temp[k] = 1
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

        # taken_list = list()
        # taken_list_i = list()
        # end_list = v.copy()
        """for k in range(0, len(self.values)):
            for l in range(0, len(self.values[0])):
                if not v[k][l] == 0 and l not in taken_list:
                    end_list[l] = v[k]
                    end_list_i[l] = i_values[k]
                    counter += 1
                    if not k == l and counter % 2 == 0:
                        end_list[l] = [-z for z in v[k]]
                        end_list_i[l] = [-z for z in i_values[k]]
                    taken_list.append(l)
                    taken_list_i.append(l)
                    break
                elif not v[k][l] == 0 and l in taken_list:
                    for m in range(l, len(self.values)):
                        if m not in taken_list:
                            end_list[m] = v[k]
                            end_list_i[m] = i_values[k]
                            counter += 1
                            if not m == l and counter % 2 == 0:
                                end_list[m] = [-z for z in v[k]]
                                end_list_i[m] = [-z for z in i_values[k]]"""

        # end_list = end_list[::-1]
        # end_list_i = end_list_i[::-1]

        # real = Matrix(*[Vector(*k) for k in end_list])
        # iden = Matrix(*[Vector(*k) for k in end_list_i])

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

        """for k in range(len(iden_values)):
            for l in range(len(iden_values)):
                if abs(iden_values[k][l]) < 0.00000001:
                    iden_values[k][l] = 0"""

        # print(Matrix(*[Vector(*k) for k in v]))
        return Matrix(*[Vector(*k) for k in iden_values])

    def cramer(a, number: int):
        if not type(a) == Matrix:
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

def sqrt(arg: int or float, resolution: int = 10):
    """
    Square root with Newton's method. Speed is very close to
    the math.sqrt(). This may be due to both complex number
    allowance and digit counting while loop, I don't know the
    built-in algorithm.

    :param arg: Number, can be negative
    :param resolution: Number of iterations
    :return: float or complex
    """
    if resolution < 1: raise MathRangeError()
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



def cumsum(arg: list or tuple):
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
        raise MathArgError()


def __cumdiv(x: int or float, power: int):
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


def e(exponent: int or float, resolution: int = 15):
    """
    e^x function.

    :param exponent: x value
    :param resolution: Up to which exponent Taylor series will continue.
    :return: e^x
    """
    if (resolution < 1): raise MathRangeError()
    sum = 1
    for k in range(resolution, 0, -1):
        sum += __cumdiv(exponent, k)
    return sum


def sin(angle: int or float, resolution: int = 15):
    """
    sin(x) using Taylor series. Input is in degrees.

    :param angle: degrees
    :param resolution: Up to which exponent Taylor series will continue.
    :return: sin(angle)
    """
    if not (resolution % 2) or resolution < 1: raise MathRangeError()

    radian: float = (2 * PI * (angle % 360 / 360)) % (2 * PI)
    result: float = 0

    for k in range(resolution, 0, -2):
        result = result + __cumdiv(radian, k) * pow(-1, (k - 1) / 2)
    return result


def cos(angle: int or float, resolution: int = 16):
    """
    cos(x) using Taylor series. Input is in degrees.

    :param angle: degrees
    :param resolution: Up to which exponent Taylor series will continue.
    :return: cos(angle)
    """
    if (resolution % 2) or resolution < 1: raise MathRangeError()
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
    if (resolution % 2) or resolution < 1: raise MathRangeError()
    try:
        return sin(angle, resolution - 1) / cos(angle, resolution)
        # Because of the error amount, probably cos will never be zero.
    except ZeroDivisionError:
        return None


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


def sinh(x: int or float, resolution: int = 15):
    return (e(x, resolution) - e(-x, resolution)) / 2


def cosh(x: int or float, resolution: int = 15):
    return (e(x, resolution) + e(-x, resolution)) / 2


def tanh(x: int or float, resolution: int = 15):
    try:
        return sinh(x, resolution) / cosh(x, resolution)
    except ZeroDivisionError:
        return None


def coth(x: int or float, resolution: int = 15):
    try:
        return cosh(x, resolution) / sinh(x, resolution)
    except ZeroDivisionError:
        return None

def arcsin(x: int or float, resolution: int = 20):
    """
    A Taylor series implementation of arcsin.

    :param x: sin(angle)
    :param resolution: Resolution of operation
    :return: angle
    """
    if not (-1 <= x <= 1): raise MathRangeError()
    if resolution < 1: raise MathRangeError()
    c = 1
    sol = x
    for k in range(1, resolution):
        c *= (2 * k - 1) / (2 * k)
        sol += c * pow(x, 2 * k + 1) / (2 * k + 1)
    return sol * 360 / (2 * PI)

def arccos(x: int or float, resolution: int = 20):
    if not (-1 <= x <= 1): raise MathRangeError()
    if resolution < 1: raise MathRangeError()

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
        type(f)) != "<class 'builtin_function_or_method'>": raise MathArgError()

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
            """get = __find(f, x, x + search_step, search_step / 10)
            if get is not None:
                zeroes.append(get)"""
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
        type(f)) != "<class 'builtin_function_or_method'>": raise MathArgError()

    return (f(x + h) - f(x)) / h


def __mul(row: list, m, id: int, target: dict, amount: int):
    length = len(m[0])  # Number of columns for the second matrix
    result = [0] * length

    """if length > 5:
        pool = [0] * innermax
        count = 0

        for k in range(length):
            if count >= innermax:
                pool[-1].join()
                count = 0
            pool[count] = threading.Thread(target=__row, args=[k, row, [m[l][k] for l in range(amount)], result, amount])
            pool[count].start()
            count += 1

        for k in pool:
            k.join()
        target[id] = result
        return"""

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

def findsol(f, x: int = 0, resolution: int = 15):
    """
    Finds a singular solution at a time with Newton's method.

    :param f: Function
    :param x: Starting value
    :param resolution: Number of iterations
    :return: The found/guessed solution of the function
    """
    if str(type(f)) != "<class 'function'>" and str(
        type(f)) != "<class 'builtin_function_or_method'>": raise MathArgError()
    if resolution < 1: raise MathRangeError()

    for k in range(resolution):
        x = x - (f(x) / derivative(f, x))

    return x



class complex:

    def __init__(self, real: int or float = 0, imaginary: int or float = 0):
        self.real = real
        self.imaginary = imaginary

    def __str__(self):
        if self.imaginary >= 0: return f"{self.real} + {self.imaginary}i"

        return f"{self.real} {self.imaginary}i"

    def __add__(self, arg):
        if type(arg) != int and type(arg) != float and type(arg) != complex: raise MathArgError()

        if type(arg) == complex:
            return complex(self.real + arg.real, self.imaginary + arg.imaginary)
        return complex(self.real + arg, self.imaginary + arg)

    def __iadd__(self, arg):
        if type(arg) != int and type(arg) != float and type(arg) != complex: raise MathArgError()

        if type(arg) == complex:
            self.real += arg.real
            self.imaginary += arg.imaginary
            return self
        self.real += arg
        self.imaginary += arg
        return self

    def __sub__(self, arg):
        if type(arg) != int and type(arg) != float and type(arg) != complex: raise MathArgError()

        if type(arg) == complex:
            return complex(self.real - arg.real, self.imaginary - arg.imaginary)
        return complex(self.real - arg, self.imaginary - arg)

    def __isub__(self, arg):
        if type(arg) != int and type(arg) != float and type(arg) != complex: raise MathArgError()

        if type(arg) == complex:
            self.real -= arg.real
            self.imaginary -= arg.imaginary
            return self
        self.real -= arg
        self.imaginary -= arg
        return self

    def __mul__(self, arg):
        if type(arg) != int and type(arg) != float and type(arg) != complex: raise MathArgError()

        if type(arg) == complex:
            return complex(self.real * arg.real - self.imaginary * arg.imaginary,
                           self.real * arg.imaginary + self.imaginary * arg.real)
        return complex(self.real * arg, self.imaginary * arg)

    def __imul__(self, arg):
        if type(arg) != int and type(arg) != float and type(arg) != complex: raise MathArgError()

        if type(arg) == complex:
            self.real = self.real * arg.real - self.imaginary * arg.imaginary
            self.imaginary = self.real * arg.imaginary + self.imaginary * arg.real
            return self
        self.real *= arg
        self.imaginary *= arg
        return self

    def __truediv__(self, arg):
        if type(arg) != int and type(arg) != float and type(arg) != complex: raise MathArgError()

        if type(arg) == complex:
            return self * arg.inverse()
        return complex(self.real / arg, self.imaginary / arg)

    def __idiv__(self, arg):
        if type(arg) != int and type(arg) != float and type(arg) != complex: raise MathArgError()

        if type(arg) == complex:
            temp = self * arg.inverse()
            self.real, self.imaginary = temp.real, temp.imaginary
            return self
        self.real /= arg
        self.imaginary /= arg
        return self

    def __floordiv__(self, arg):
        if type(arg) != int and type(arg) != float and type(arg) != complex: raise MathArgError()

        if type(arg) == complex:
            temp = self * arg.inverse()
            return complex(temp.real // 1, temp.imaginary // 1)
        return complex(self.real // arg, self.imaginary // arg)

    def __ifloordiv__(self, arg):
        if type(arg) != int and type(arg) != float and type(arg) != complex: raise MathArgError()

        if type(arg) == complex:
            temp = self * arg.inverse()
            self.real, self.imaginary = temp.real // 1, temp.imaginary // 1
            return self
        self.real //= arg
        self.imaginary //= arg
        return self


    def conjugate(self):
        return complex(self.real, -self.imaginary)

    def length(self):
        return (self * self.conjugate()).real

    def unit(self):
        return self / sqrt(self.length())

    def sqrt(arg, resolution: int = 200):
        if not isinstance(arg, complex): raise MathArgError()
        temp = arg.unit()
        angle = arcsin(temp.imaginary, resolution=resolution) / 2
        return complex(cos(angle), sin(angle)) * sqrt(sqrt(arg.length()))

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
        return complex(self.real / divisor, -self.imaginary / divisor)

    def rotate(self, angle: int or float):
        return self * complex(cos(angle), sin(angle))

    def rotationFactor(self, angle: int or float):
        return complex(cos(angle), sin(angle))


