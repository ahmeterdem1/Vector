import math
import time
# These imports are just for speed comparisons...

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
    def __init__(self):
        super().__init__("Argument elements are of the wrong type")

class MathRangeError(Exception):
    def __init__(self):
        super().__init__("Argument(s) out of range")

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

def sinh(x: int or float, resolution: int = 15):
    return (e(x, resolution) - e(-x, resolution))/2

def cosh(x: int or float, resolution: int = 15):
    return(e(x, resolution) + e(-x, resolution))/2

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


def solve(f, low: int or float = -50, high: int or float = 50, search_step: int or float = 0.1, res: float = 0.0001) -> list:
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
    if str(type(f)) != "<class 'function'>" and str(type(f)) != "<class 'builtin_function_or_method'>": raise MathArgError()

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

    zeroes = list(map(lambda x: x if(abs(x) > 0.00001) else 0, zeroes))

    return zeroes

def derivative(f, x: int or float, h: float = 0.0000000001) -> float:
    if str(type(f)) != "<class 'function'>" and str(type(f)) != "<class 'builtin_function_or_method'>": raise MathArgError()

    return (f(x + h) - f(x)) / h



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
    def range(lowreal: int or float, highreal: int or float, lowimg: int or float, highimg: int or float, step1: float = 1, step2: float = 1):
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



if __name__ == "__main__":

    #for k in complex.range(0, 5, 5, 0, 1, -1):
    #    print(k)
    begin = time.time()
    print(solve(lambda x: e(x) - 18, low=2, high=4))
    end = time.time()
    print(end - begin)

    print(derivative(lambda x: (x + 3)**2, 0))


