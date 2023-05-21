import math
import random

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
                    assert (a.isnumeric()), Exception("VectorError: args must be of type float")
            try:
                a = a.replace(".", "")
                a = a.replace("-", "")
                a = a.replace("e", "")
            except:
                pass
            assert (a.isnumeric() or type(a) == bool), Exception("VectorError: args must be of type float")
        self.dimension = len(args)
        self.values = [_ for _ in args]

    def __str__(self):
        return str(self.values)

    def __add__(self, arg):
        if type(arg) == int or type(arg) == float:
            return Vector(*[self.values[k] + arg for k in range(0, self.dimension)])
        assert (type(arg) == Vector), Exception("TypeError: arg must be of type Vector, int, float")
        assert (self.dimension == arg.dimension), Exception("DimensionError: dimensions must match")
        return Vector(*[self.values[k] + arg.values[k] for k in range(0, self.dimension)])

    def __sub__(self, arg):
        if type(arg) == int or type(arg) == float:
            return Vector(*[self.values[k] - arg for k in range(0, self.dimension)])
        assert (type(arg) == Vector), Exception("TypeError: arg must be of type Vector, int, float")
        assert (self.dimension == arg.dimension), Exception("DimensionError: dimensions must match")
        return Vector(*[self.values[k] - arg.values[k] for k in range(0, self.dimension)])

    def dot(self, arg):
        assert type(arg) == Vector, Exception("TypeError: arg must be of type Vector")
        assert (self.dimension == arg.dimension), Exception("DimensionError: dimensions must match")
        mul = [self.values[k] * arg.values[k] for k in range(0, self.dimension)]
        sum = 0
        for k in mul:
            sum += k
        return sum

    def __mul__(self, arg):
        assert type(arg) == int or type(arg) == float, Exception("TypeError: arg must be of type int, float")
        return Vector(*[self.values[k] * arg for k in range(0, self.dimension)])

    def __truediv__(self, arg):
        assert type(arg) == int or type(arg) == float, Exception("TypeError: arg must be of type int, float")
        return Vector(*[self.values[k] / arg for k in range(0, self.dimension)])

    def __floordiv__(self, arg):
        assert type(arg) == int or type(arg) == float, Exception("TypeError: arg must be of type int, float")
        return Vector(*[self.values[k] // arg for k in range(0, self.dimension)])

    def __iadd__(self, arg):
        if type(arg) == int or type(arg) == float:
            return Vector(*[self.values[k] + arg for k in range(0, self.dimension)])
        assert (type(arg) == Vector), Exception("TypeError: arg must be of type Vector, int, float")
        assert (self.dimension == arg.dimension), Exception("DimensionError: dimensions must match")
        return Vector(*[self.values[k] + arg.values[k] for k in range(0, self.dimension)])

    def __isub__(self, arg):
        if type(arg) == int or type(arg) == float:
            return Vector(*[self.values[k] - arg for k in range(0, self.dimension)])
        assert (type(arg) == Vector), Exception("TypeError: arg must be of type Vector, int, float")
        assert (self.dimension == arg.dimension), Exception("DimensionError: dimensions must match")
        return Vector(*[self.values[k] - arg.values[k] for k in range(0, self.dimension)])

    def __gt__(self, arg):
        if type(arg) == int or type(arg) == float:
            sum = 0
            for k in self.values:
                sum += k * k
            if sum > arg * arg:
                return True
            return False
        assert (type(arg) == Vector), Exception("TypeError: arg must be of type Vector, int, float")
        assert (self.dimension == arg.dimension), Exception("DimensionError: dimensions must match")
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
        assert (type(arg) == Vector), Exception("TypeError: arg must be of type Vector, int, float")
        assert (self.dimension == arg.dimension), Exception("DimensionError: dimensions must match")
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
        assert (type(arg) == Vector), Exception("TypeError: arg must be of type Vector, int, float")
        assert (self.dimension == arg.dimension), Exception("DimensionError: dimensions must match")
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
        assert (type(arg) == Vector), Exception("TypeError: arg must be of type Vector, int, float")
        assert (self.dimension == arg.dimension), Exception("DimensionError: dimensions must match")
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
        assert (type(arg) == Vector), Exception("TypeError: arg must be of type Vector, int, float")
        assert (self.dimension == arg.dimension), Exception("DimensionError: dimensions must match")
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
        assert (type(arg) == Vector), Exception("TypeError: arg must be of type Vector")
        assert (self.dimension == arg.dimension), Exception("DimensionError: dimensions must match")
        return Vector(*[(self.values[k] and arg.values[k]) for k in range(0, self.dimension)])

    def __iand__(self, arg):
        assert (type(arg) == Vector), Exception("TypeError: arg must be of type Vector")
        assert (self.dimension == arg.dimension), Exception("DimensionError: dimensions must match")
        return Vector(*[(self.values[k] and arg.values[k]) for k in range(0, self.dimension)])

    def __or__(self, arg):
        assert (type(arg) == Vector), Exception("TypeError: arg must be of type Vector")
        assert (self.dimension == arg.dimension), Exception("DimensionError: dimensions must match")
        return Vector(*[(self.values[k] or arg.values[k]) for k in range(0, self.dimension)])

    def __ior__(self, arg):
        assert (type(arg) == Vector), Exception("TypeError: arg must be of type Vector")
        assert (self.dimension == arg.dimension), Exception("DimensionError: dimensions must match")
        return Vector(*[(self.values[k] or arg.values[k]) for k in range(0, self.dimension)])

    def __xor__(self, arg):
        assert (type(arg) == Vector), Exception("TypeError: arg must be of type Vector")
        assert (self.dimension == arg.dimension), Exception("DimensionError: dimensions must match")
        return Vector(*[(self.values[k] ^ arg.values[k]) for k in range(0, self.dimension)])

    def __ixor__(self, arg):
        assert (type(arg) == Vector), Exception("TypeError: arg must be of type Vector")
        assert (self.dimension == arg.dimension), Exception("DimensionError: dimensions must match")
        return Vector(*[(self.values[k] ^ arg.values[k]) for k in range(0, self.dimension)])

    def __invert__(self):
        return Vector(*[int(not self.values[k]) for k in range(0, self.dimension)])

    def append(self, arg):
        if type(arg) == int or type(arg) == float:
            temp = self.values.copy()
            temp.append(arg)
            return Vector(*temp)
        assert (type(arg) == Vector), Exception("TypeError: arg must be of type Vector, int, float")
        temp = self.values.copy()
        for k in arg.values:
            temp.append(k)
        return Vector(*temp)

    def length(self):
        sum = 0
        for k in self.values:
            sum += k*k
        return math.sqrt(sum)

    def proj(self, arg):
        assert type(arg) == Vector, Exception("TypeError: arg must be of type Vector")
        assert (self.dimension == arg.dimension), Exception("DimensionError: dimensions must match")
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
            assert type(k) == Vector, Exception("TypeError: args must be of type Vector")
            assert k.dimension == (len(args)), Exception("ArgumentError: not the correct amount of args")
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
        assert type(dim) == int and type(a) == int and type(b) == int, Exception("TypeError: args must be of type int")
        assert (dim > 0), Exception("DimensionError: number of dimensions cannot be zero")
        return Vector(*[random.randint(a, b) for k in range(0, dim)])

    def randVfloat(dim: int, a: float, b: float):
        assert type(dim) == int and (type(a) == int or type(a) == float) and (type(b) == int or type(b) == float), Exception("TypeError: args must be of type int")
        assert (dim > 0), Exception("DimensionError: number of dimensions cannot be zero")
        return Vector(*[random.uniform(a, b) for k in range(0, dim)])

    def randVbool(dim: int):
        assert type(dim) == int, Exception("TypeError: arg must be of type int")
        assert (dim > 0), Exception("DimensionError: number of dimensions cannot be zero")
        return Vector(*[random.randrange(0, 2) for k in range(0, dim)])

    def determinant(*args):
        for k in args:
            assert (type(k) == Vector), Exception("TypeError: args must be of type Vector")
            assert (args[0].dimension == k.dimension), Exception("DimensionError: dimensions must match")
        assert (len(args) == args[0].dimension), Exception("ArgumentError: not the correct amount of args")
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
            assert (type(k) == Vector), Exception("TypeError: all inputs must be of type Vector")
            assert (args[0].dimension == k.dimension), Exception("DimensionError: dimensions must match")
        if len(args) == 2 and args[0].dimension == 2:
            return args[0].values[0] * args[1].values[1] - args[0].values[1] * args[1].values[0]
        assert (len(args) == args[0].dimension - 1), Exception("ArgumentError: not the correct amount of args")

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
            assert type(k) == Vector, Exception("TypeError: args must be of type Vector")
            assert k.dimension == args[0].dimension, Exception("DimensionError: dimension of args must be the same")
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
        assert type(arg) == Matrix, Exception("TypeError: type of arg must be of int, float, Matrix")
        assert self.dimension == arg.dimension, Exception("DimensionError: dimension of args must be the same")
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
        assert type(arg) == Matrix, Exception("TypeError: type of arg must be of int, float, Matrix")
        assert self.dimension == arg.dimension, Exception("DimensionError: dimension of args must be the same")
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
        assert type(arg) == Matrix, Exception("TypeError: type of arg must be of int, float, Matrix")
        assert self.dimension == arg.dimension, Exception("DimensionError: dimension of args must be the same")
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
        assert type(arg) == Matrix, Exception("TypeError: type of arg must be of int, float, Matrix")
        assert self.dimension == arg.dimension, Exception("DimensionError: dimension of args must be the same")
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
            assert self.dimension.split("x")[0] == str(arg.dimension), Exception("DimensionError: dimensions of args must match")
            for k in range(0, len(self.values)):
                sum = 0
                for l in range(0, len(arg.values)):
                    sum += self.values[k][l] * arg.values[l]
                v.append(sum)
            return Vector(*v)

        assert type(arg) == Matrix, Exception("TypeError: type of arg must be of int, float, Matrix, Vector")
        assert self.dimension.split("x")[1] == arg.dimension.split("x")[0], Exception("DimensionError: dimensions of args must match")
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
        assert type(arg) == int or type(arg) == float, Exception("TypeError: arg must be of type int, float")
        for k in self.values:
            v.append(Vector(*[l / arg for l in k]))
        return Matrix(*v)

    def __floordiv__(self, arg):
        v = list()
        assert type(arg) == int or type(arg) == float, Exception("TypeError: arg must be of type int, float")
        for k in self.values:
            v.append(Vector(*[l // arg for l in k]))
        return Matrix(*v)

    def determinant(arg):
        assert type(arg) == Matrix, Exception("TypeError: arg must be of type Matrix")
        if arg.dimension == "1x1":
            return arg.values[0][0]
        return Vector.determinant(*[Vector(*k) for k in arg.values])

    def __or__(self, arg):
        v = list()
        assert type(arg) == Matrix, Exception("TypeError: type of arg must be of int, float, Matrix")
        assert self.dimension == arg.dimension, Exception("DimensionError: dimensions must be the same")
        for k in range(0, len(self.values)):
            m = list()
            for l in range(0, len(self.values[k])):
                m.append(self.values[k][l] or arg.values[k][l])
            v.append(Vector(*m))
        return Matrix(*v)

    def __ior__(self, arg):
        v = list()
        assert type(arg) == Matrix, Exception("TypeError: type of arg must be of Matrix")
        assert self.dimension == arg.dimension, Exception("DimensionError: dimensions must be the same")
        for k in range(0, len(self.values)):
            m = list()
            for l in range(0, len(self.values[k])):
                m.append(self.values[k][l] or arg.values[k][l])
            v.append(Vector(*m))
        return Matrix(*v)

    def __and__(self, arg):
        v = list()
        assert type(arg) == Matrix, Exception("TypeError: type of arg must be of Matrix")
        assert self.dimension == arg.dimension, Exception("DimensionError: dimensions must be the same")
        for k in range(0, len(self.values)):
            m = list()
            for l in range(0, len(self.values[k])):
                m.append(self.values[k][l] and arg.values[k][l])
            v.append(Vector(*m))
        return Matrix(*v)

    def __iand__(self, arg):
        v = list()
        assert type(arg) == Matrix, Exception("TypeError: type of arg must be of Matrix")
        assert self.dimension == arg.dimension, Exception("DimensionError: dimensions must be the same")
        for k in range(0, len(self.values)):
            m = list()
            for l in range(0, len(self.values[k])):
                m.append(self.values[k][l] and arg.values[k][l])
            v.append(Vector(*m))
        return Matrix(*v)

    def __xor__(self, arg):
        v = list()
        assert type(arg) == Matrix, Exception("TypeError: type of arg must be of Matrix")
        assert self.dimension == arg.dimension, Exception("DimensionError: dimensions must be the same")
        for k in range(0, len(self.values)):
            m = list()
            for l in range(0, len(self.values[k])):
                m.append(self.values[k][l] ^ arg.values[k][l])
            v.append(Vector(*m))
        return Matrix(*v)

    def __ixor__(self, arg):
        v = list()
        assert type(arg) == Matrix, Exception("TypeError: type of arg must be of Matrix")
        assert self.dimension == arg.dimension, Exception("DimensionError: dimensions must be the same")
        for k in range(0, len(self.values)):
            m = list()
            for l in range(0, len(self.values[k])):
                m.append(self.values[k][l] ^ arg.values[k][l])
            v.append(Vector(*m))
        return Matrix(*v)

    def __invert__(self):
        return Matrix(*[Vector(*[int(not l) for l in k]) for k in self.values])

    def __eq__(self, arg):
        assert type(arg) == Matrix, Exception("TypeError: type of arg must be of Matrix")
        if self.values == arg.values:
            return True
        return False

    def transpose(self):
        v = list()
        for k in range(0, len(self.values[0])):
            m = list()
            for l in range(0, len(self.values)):
                m.append(self.values[l][k])
            v.append(Vector(*m))
        return Matrix(*v)

    def inverse(self):
        assert self.dimension.split("x")[0] == self.dimension.split("x")[1], Exception("DimensionError: Matrix must be a square")
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
        v = list()
        for k in range(0, dim):
            temp = [0] * dim
            temp[k] = 1
            v.append(Vector(*temp))
        return Matrix(*v)

    def randMint(m: int, n: int, a: int, b: int):
        v = list()
        for k in range(0, m):
            temp = list()
            for l in range(0, n):
                temp.append(random.randint(a, b))
            v.append(Vector(*temp))
        return Matrix(*v)

    def randMfloat(m: int, n: int, a: float, b: float):
        v = list()
        for k in range(0, m):
            temp = list()
            for l in range(0, n):
                temp.append(random.uniform(a, b))
            v.append(Vector(*temp))
        return Matrix(*v)

    def randMbool(m: int, n: int):
        v = list()
        for k in range(0, m):
            temp = list()
            for l in range(0, n):
                temp.append(random.randint(0, 1))
            v.append(Vector(*temp))
        return Matrix(*v)

