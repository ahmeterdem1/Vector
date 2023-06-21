import math
import random

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
        control_list = ["e-0", "e-1", "e-2", "e-3", "e-4", "e-5", "e-6", "e-7", "e-8", "e-9", "e0", "e1", "e2", "e3", "e4", "e5", "e6", "e7", "e8", "e9"]
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
        if not (type(dim) == int):
            raise ArgTypeError("i")
        if not (dim > 0):
            raise RangeError
        return Vector(*[random.randrange(0, 2) for k in range(0, dim)])

    def determinant(*args):
        for k in args:
            if not (type(k) == Vector):
                raise ArgTypeError("v")
            if not (args[0].dimension == k.dimension):
                raise DimensionError(0)
        if not (len(args) == args[0].dimension):
            raise ArgumentError
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
            if not (type(k) == Vector):
                raise ArgTypeError("v")
            if not (args[0].dimension == k.dimension):
                raise DimensionError(0)
        if len(args) == 2 and args[0].dimension == 2:
            return args[0].values[0] * args[1].values[1] - args[0].values[1] * args[1].values[0]
        if not (len(args) == args[0].dimension - 1):
            raise ArgumentError

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




class Matrix:
    def __init__(self, *args):
        for k in args:
            if not (type(k) == Vector):
                raise ArgTypeError("v")
            if not (args[0].dimension == k.dimension):
                raise DimensionError(0)
        self.values = [k.values for k in args]
        self.dimension = f"{args[0].dimension}x{len(args)}"


    def __str__(self):
        return str(self.values)

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
        if not (self.dimension.split("x")[0] == self.dimension.split("x")[1]):
            raise DimensionError(2)
        if self.dimension == "1x1":
            return (1 / self.values[0][0])
        det = self.det_echelon()
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
                temp.append(pow(-1, k + l) * Matrix(*sub).det_echelon())
            end.append(temp)
        return Matrix(*[Vector(*k) for k in end]).transpose() / det

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


