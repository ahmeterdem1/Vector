import math
import time
# These imports are just for speed comparisons...

PI = 3.14159265359
E = 2.718281828459
ln2 = 0.6931471805599569
class MathArgError(Exception):
    def __init__(self):
        super().__init__("Argument elements are of the wrong type")

class MathRangeError(Exception):
    def __init__(self):
        super().__init__("Argument(s) out of range")

def Range(low: int or float, high: int or float):
    if (high < low): raise MathRangeError()

    """
    A lazy implementation for creating ranges.
    
    This works almost at the exact speed as built 
    in range() when given a function inside the 
    loop such as print().

    :param low: low limit
    :param high: high limit
    :return: yields
    """
    while low < high:
        yield low
        low += 1
def abs(arg: int or float):
    """
    Basic absolute value function

    :param arg: float or int
    :return: Absolute value of the arg
    """
    return arg if (arg >= 0) else -arg

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
        result = result + __cumdiv(radian, k) * pow(-1, (k - 1)/2)
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
        result = result + __cumdiv(radian, k) * pow(-1, k/2)
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

"""def ln(x: int or float, resolution: int = 40):
    if x <= 0: raise MathRangeError()

    result: float = 1
    if x > 1:
        while x >= 2:
            result += ln2
            x /= 2
    else:
        while x < 1:
            result -= ln2
            x *= 2
    if not x == 1:
        for k in range(resolution, 1, -1):
            result += pow(-1, k - 1) * pow(1 / x, k)

    return result"""

class complex:

    def __init__(self, real: int or float, imaginary: int or float):
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
            return complex(self.real * arg.real - self.imaginary * arg.imaginary, self.real * arg.imaginary + self.imaginary * arg.real)
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

    def conjugate(self):
        return complex(self.real, -self.imaginary)

    def length(self):
        return (self * self.conjugate()).real

    # noinspection PyMethodFirstArgAssignment
    def range(lowreal: int or float, highreal: int or float, lowimg: int or float, highimg: int or float):
        if highreal < lowreal or highimg < lowimg: raise MathRangeError()
        reset = lowimg
        while lowreal < highreal:
            lowimg = reset
            while lowimg < highimg:
                yield complex(lowreal, lowimg)
                lowimg += 1
            lowreal += 1

    def inverse(self):
        divisor = self.length()
        return complex(self.real / divisor, -self.imaginary / divisor)

    def rotate(self, angle: int or float):
        return self * complex(cos(angle), sin(angle))

    def rotationFactor(self, angle: int or float):
        return complex(cos(angle), sin(angle))



if __name__ == "__main__":

    for k in complex.range(0, 5, 0, 5):
        print(k)


