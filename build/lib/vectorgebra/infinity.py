from .undefined import *
from decimal import *
from typing import Callable, Union, List

PI = 3.14159265359

def sqrt(arg, resolution: int = 10):
    """
        Computes the square root of a numerical argument with a specified resolution.

        Args:
            arg (int, float, Decimal, Complex, Infinity, Undefined): The numerical value or type for which the square root is to be computed.
            resolution (int, optional): The number of iterations for the approximation. Defaults to 10.

        Returns:
            int, float, Decimal, Complex, Infinity, Undefined: The square root of the argument.

        Raises:
            ArgTypeError: If the argument is not of a valid type.
            RangeError: If the resolution is not a positive integer.
    """
    if isinstance(arg, Union[int, float, Decimal]):
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
        return Complex(0, estimate)
    if isinstance(arg, Complex):
        return arg.sqrt()
    if isinstance(arg, Infinity):
        if arg.sign: return Infinity()
        return sqrt(Complex(0, 1)) * Infinity()
    if isinstance(arg, Undefined):
        return Undefined()
    raise ArgTypeError()

def __cumdiv(x, power: int):
    """
        Computes the cumulative division of a numerical value 'x' by a specified power, cumulative
        divison being a related term at Taylor Series.

        Args:
            x (int, float, Decimal, Infinity, Undefined): The numerical value to be divided cumulatively.
            power (int): The power by which 'x' is to be divided cumulatively.

        Returns:
            float: The result of the cumulative division.

        Raises:
            ArgTypeError: If 'x' is not a numerical value.
    """
    if not isinstance(x, Union[int, float, Decimal, Infinity, Undefined]):
        raise ArgTypeError("Must be a numerical value.")

    result: float = 1
    for k in range(power, 0, -1):
        result *= x / k
    return result

def sin(angle, resolution=15):
    """
        Computes the sine of the given angle using Taylor Series approximation.

        Args:
            angle (int, float, Decimal, Infinity, Undefined): The angle in degrees.
            resolution (int, optional): The resolution for the approximation. Defaults to 15.

        Returns:
            float: The sine value of the given angle.

        Raises:
            ArgTypeError: If 'angle' is not a numerical value.
            RangeError: If 'resolution' is not a positive integer.
    """
    if not (isinstance(resolution, int) and resolution >= 1): raise RangeError("Resolution must be a positive integer")
    if not isinstance(angle, Union[int, float, Decimal, Infinity, Undefined]):
        raise ArgTypeError("Must be a numerical value.")

    radian: float = (2 * PI * (angle % 360 / 360)) % (2 * PI)
    result: float = 0
    if not resolution % 2:
        resolution += 1
    for k in range(resolution, 0, -2):
        result = result + __cumdiv(radian, k) * pow(-1, (k - 1) / 2)

    return result

def cos(angle, resolution=16):
    """
        Computes the cosine of the given angle using Taylor Series approximation.

        Args:
            angle (int, float, Decimal, Infinity, Undefined): The angle in degrees.
            resolution (int, optional): The resolution for the approximation. Defaults to 16.

        Returns:
            float: The cosine value of the given angle.

        Raises:
            ArgTypeError: If 'angle' is not a numerical value.
            RangeError: If 'resolution' is not a positive integer.
        """
    if not (isinstance(resolution, int) and resolution >= 1): raise RangeError("Resolution must be a positive integer")
    if not isinstance(angle, Union[int, float, Decimal, Infinity, Undefined]):
        raise ArgTypeError("Must be a numerical value.")

    radian: float = (2 * PI * (angle % 360 / 360)) % (2 * PI)
    result: float = 1

    if resolution % 2:
        resolution += 1

    for k in range(resolution, 0, -2):
        result = result + __cumdiv(radian, k) * pow(-1, k / 2)
    return result

def arcsin(x, resolution: int = 20):
    """
        Computes the arc sine of the given number in degrees.

        Args:
            x (int, float, Decimal, Infinity, Undefined): The number.
            resolution (int, optional): The resolution for the approximation. Defaults to 20.

        Returns:
            float: The arc sine value of the given number in degrees.

        Raises:
            ArgTypeError: If 'x' is not a numerical value.
            RangeError: If 'x' is not in the range [-1, 1], or if 'resolution' is not a positive integer.
    """
    if not isinstance(x, Union[int, float, Decimal, Infinity, Undefined]):
        raise ArgTypeError("Must be a numerical value.")
    if not (-1 <= x <= 1): raise RangeError()
    if resolution < 1: raise RangeError("Resolution must be a positive integer")
    c = 1
    sol = x
    for k in range(1, resolution):
        c *= (2 * k - 1) / (2 * k)
        sol += c * pow(x, 2 * k + 1) / (2 * k + 1)
    return sol * 360 / (2 * PI)

class Complex:
    """
        Represents a complex number with real and imaginary parts.

        Attributes:
            real (Union[int, float, Decimal, Infinity, Undefined]): The real part of the complex number.
            imaginary (Union[int, float, Decimal, Infinity, Undefined]): The imaginary part of the complex number.
    """

    def __init__(self, real=0, imaginary=0):
        """
            Initializes a Complex object with the given real and imaginary parts.

            Args:
                real (Union[int, float, Decimal, Infinity, Undefined]): The real part of the complex number.
                imaginary (Union[int, float, Decimal, Infinity, Undefined]): The imaginary part of the complex number.
        """

        # Initialization is erroneous when expressions with infinities result in NoneTypes
        if not (isinstance(real, Union[int, float, Decimal, Infinity, Undefined])
                and isinstance(imaginary, Union[int, float, Decimal, Infinity, Undefined])): raise ArgTypeError()
        self.real = real
        self.imaginary = imaginary

    def __str__(self):
        if self.imaginary >= 0: return f"{self.real} + {self.imaginary}i"

        return f"{self.real} - {-self.imaginary}i"

    def __repr__(self):
        if self.imaginary >= 0: return f"{self.real} + {self.imaginary}i"

        return f"{self.real} - {-self.imaginary}i"

    def __add__(self, arg):
        if isinstance(arg, Complex):
            return Complex(self.real + arg.real, self.imaginary + arg.imaginary)
        if isinstance(arg, Union[int, float, Decimal, Infinity, Undefined]):
            return Complex(self.real + arg, self.imaginary)
        raise ArgTypeError()

    def __radd__(self, arg):
        if not isinstance(arg, Union[int, float, Decimal, Infinity, Undefined]): raise ArgTypeError()
        return Complex(self.real + arg, self.imaginary)

    def __iadd__(self, arg):
        if isinstance(arg, Complex):
            self.real += arg.real
            self.imaginary += arg.imaginary
            return self
        if isinstance(arg, Union[int, float, Decimal, Infinity, Undefined]):
            self.real += arg
            return self
        raise ArgTypeError()

    def __sub__(self, arg):
        if isinstance(arg, Complex):
            return Complex(self.real - arg.real, self.imaginary - arg.imaginary)
        if isinstance(arg, Union[int, float, Decimal, Infinity, Undefined]):
            return Complex(self.real - arg, self.imaginary)
        raise ArgTypeError()

    def __rsub__(self, arg):
        if not isinstance(arg, Union[int, float, Decimal, Infinity, Undefined]): raise ArgTypeError()
        return -Complex(self.real - arg, self.imaginary)

    def __isub__(self, arg):
        if isinstance(arg, Complex):
            self.real -= arg.real
            self.imaginary -= arg.imaginary
            return self
        if isinstance(arg, Union[int, float, Decimal, Infinity, Undefined]):
            self.real -= arg
            return self
        raise ArgTypeError()

    def __mul__(self, arg):
        if isinstance(arg, Complex):
            return Complex(self.real * arg.real - self.imaginary * arg.imaginary,
                           self.real * arg.imaginary + self.imaginary * arg.real)
        if isinstance(arg, Union[int, float, Decimal, Infinity, Undefined]):
            return Complex(self.real * arg, self.imaginary * arg)
        raise ArgTypeError()

    def __rmul__(self, arg):
        if not isinstance(arg, Union[int, float, Decimal, Infinity, Undefined]): raise ArgTypeError()
        return Complex(self.real * arg, self.imaginary * arg)

    def __imul__(self, arg):
        if isinstance(arg, Complex):
            self.real = self.real * arg.real - self.imaginary * arg.imaginary
            self.imaginary = self.real * arg.imaginary + self.imaginary * arg.real
            return self
        if isinstance(arg, Union[int, float, Decimal, Infinity, Undefined]):
            self.real *= arg
            self.imaginary *= arg
            return self
        raise ArgTypeError()

    def __truediv__(self, arg):
        if isinstance(arg, Complex):
            return self * arg.inverse()
        if isinstance(arg, Union[int, float, Decimal, Infinity, Undefined]):
            if arg: return Complex(self.real / arg, self.imaginary / arg)
            return Complex(Infinity(self.real >= 0), Infinity(self.imaginary >= 0))
        raise ArgTypeError()

    def __rtruediv__(self, arg):
        if not isinstance(arg, Union[int, float, Decimal, Infinity, Undefined]): raise ArgTypeError()
        return arg * self.inverse()

    def __idiv__(self, arg):
        if isinstance(arg, Complex):
            temp = self * arg.inverse()
            self.real, self.imaginary = temp.real, temp.imaginary
            return self
        if isinstance(arg, Union[int, float, Decimal, Infinity, Undefined]):
            self.real /= arg
            self.imaginary /= arg
            return self
        raise ArgTypeError()

    def __floordiv__(self, arg):
        if isinstance(arg, Complex):
            temp = self * arg.inverse()
            return Complex(temp.real // 1, temp.imaginary // 1)
        if isinstance(arg, Union[int, float, Decimal, Infinity, Undefined]):
            return Complex(self.real // arg, self.imaginary // arg)
        raise ArgTypeError()

    def __ifloordiv__(self, arg):
        if isinstance(arg, Complex):
            temp = self * arg.inverse()
            self.real, self.imaginary = temp.real // 1, temp.imaginary // 1
            return self
        if isinstance(arg, Union[int, float, Decimal, Infinity, Undefined]):
            self.real //= arg
            self.imaginary //= arg
            return self
        raise ArgTypeError()

    def __neg__(self):
        return Complex(-self.real, -self.imaginary)

    def __eq__(self, arg):
        if isinstance(arg, Complex):
            if self.real == arg.real and self.imaginary == arg.imaginary:
                return True
            else:
                return False
        return False

    def __ne__(self, arg):
        if isinstance(arg, Complex):
            if self.real != arg.real or self.imaginary != arg.imaginary:
                return True
            else:
                return False
        return True

    def __gt__(self, arg):
        if isinstance(arg, Complex):
            return self.length() > arg.length()
        if isinstance(arg, Union[int, float, Decimal, Infinity, Undefined]): return self.length() > arg
        raise ArgTypeError()

    def __ge__(self, arg):
        if isinstance(arg, Complex):
            return self.length() >= arg.length()
        if isinstance(arg, Union[int, float, Decimal, Infinity, Undefined]): return self.length() >= arg
        raise ArgTypeError()

    def __lt__(self, arg):
        if isinstance(arg, Complex):
            return self.length() < arg.length()
        if isinstance(arg, Union[int, float, Decimal, Infinity, Undefined]): return self.length() < arg
        raise ArgTypeError()

    def __le__(self, arg):
        if isinstance(arg, Complex):
            return self.length() <= arg.length()
        if isinstance(arg, Union[int, float, Decimal, Infinity, Undefined]): return self.length() <= arg
        raise ArgTypeError()

    def __pow__(self, p):
        temp = 1
        for k in range(p):
            temp = temp * self
        return temp

    def conjugate(self):
        """
            Returns the complex conjugate of the current complex number.

            Returns:
                Complex: A new Complex object representing the complex conjugate of the current complex number.
        """
        return Complex(self.real, -self.imaginary)

    def length(self):
        """
            Calculates the length (magnitude) of the complex number.

            Returns:
                float: The length (magnitude) of the complex number.
        """
        return (self * self.conjugate()).real

    def unit(self):
        """
            Computes the unit vector of the complex number.

            Returns:
                Complex: A complex number representing the unit vector of the original complex number.
        """
        return self / sqrt(self.length())

    def sqrt(arg, resolution: int = 200):
        """
            Computes the square root of a complex number.

            Args:
                arg (Complex): The complex number to compute the square root of.
                resolution (int, optional): The resolution for calculating the square root. Defaults to 200.

            Returns:
                Complex: The square root of the input complex number.

            Raises:
                ArgTypeError: If the input argument is not a complex number.
        """
        if isinstance(arg, Complex):
            temp = arg.unit()
            angle = arcsin(temp.imaginary, resolution=resolution) / 2
            return Complex(cos(angle), sin(angle)) * sqrt(sqrt(arg.length()))
        raise ArgTypeError()

    # noinspection PyMethodFirstArgAssignment
    def range(lowreal, highreal, lowimg, highimg, step1=1, step2=1):
        """
            Generates a range of complex numbers within specified boundaries.

            Args:
                lowreal (Union[int, float, Decimal, Infinity]): The lower bound of the real part.
                highreal (Union[int, float, Decimal, Infinity]): The upper bound of the real part.
                lowimg (Union[int, float, Decimal, Infinity]): The lower bound of the imaginary part.
                highimg (Union[int, float, Decimal, Infinity]): The upper bound of the imaginary part.
                step1 (Union[int, float, Decimal], optional): The step size for the real part. Defaults to 1.
                step2 (Union[int, float, Decimal, Infinity], optional): The step size for the imaginary part. Defaults to 1.

            Yields:
                Complex: A complex number within the specified boundaries.

            Raises:
                ArgTypeError: If any of the input arguments are not numerical values.
                RangeError: If the boundaries and step sizes result in an invalid range.
        """
        if not ((isinstance(lowreal, Union[int, float, Decimal, Infinity]))
                and (isinstance(highreal, Union[int, float, Decimal, Infinity]))
                and (isinstance(lowimg, Union[int, float, Decimal, Infinity]))
                and (isinstance(highimg, Union[int, float, Decimal, Infinity]))
                and (isinstance(step1, Union[int, float, Decimal]))
                and (isinstance(step2, Union[int, float, Decimal, Infinity]))):
            raise ArgTypeError("Must be a numerical value.")
        if (highreal < lowreal ^ step1 > 0) or (highimg < lowimg ^ step2 > 0): raise RangeError()
        reset = lowimg
        while (not highreal < lowreal ^ step1 > 0) and not highreal == lowreal:
            lowimg = reset
            while (not highimg < lowimg ^ step2 > 0) and not highimg == lowimg:
                yield Complex(lowreal, lowimg)
                lowimg += step2
            lowreal += step1

    def inverse(self):
        """
            Calculates the multiplicative inverse of the complex number.

            Returns:
                Complex: The multiplicative inverse of the complex number.

            Notes:
                If the length of the complex number is zero, returns a complex number
                with real part Infinity and imaginary part Infinity if the imaginary
                part is negative.

            Example:
                If self = a + bi, the inverse is 1 / (a + bi) = a / (a^2 + b^2) - (b / (a^2 + b^2))i

        """
        divisor = self.length()
        if divisor: return Complex(self.real / divisor, -self.imaginary / divisor)
        else: return Complex(Infinity(self.real >= 0), Infinity(-self.imaginary >= 0))

    def rotate(self, angle):
        """
            Rotates the complex number by a given angle.

            Args:
                angle (float): The angle in radians by which to rotate the complex number.

            Returns:
                Complex: The rotated complex number.

            Example:
                If self = a + bi, and angle is the rotation angle in radians,
                the rotated complex number is obtained by multiplying self by the
                rotation factor, which is a complex number with real part equal to
                cos(angle) and imaginary part equal to sin(angle).

        """
        return self * Complex(cos(angle), sin(angle))

    def rotationFactor(angle):
        if not isinstance(angle, Union[int, float, Decimal]): raise ArgTypeError("Must be a numerical value.")
        """
            Computes the rotation factor for a given angle.

            Args:
                angle (Union[int, float, Decimal]): The angle in degrees for which to compute the rotation factor.

            Returns:
                Complex: The rotation factor, a complex number with real part equal to
                cos(angle) and imaginary part equal to sin(angle).

            Example:
                If angle is the rotation angle in degrees, the rotation factor is a
                complex number with real part equal to cos(angle) and imaginary part
                equal to sin(angle), which represents the amount of rotation in the
                complex plane.
        """
        return Complex(cos(angle), sin(angle))

class Infinity:
    """
        Represents positive or negative infinity.

        Attributes:
            sign (bool): True if positive infinity, False if negative infinity.
    """
    def __init__(self, sign: bool = True):
        """
            Initializes an Infinity object.

            Args:
                sign (bool, optional): True for positive infinity, False for negative infinity. Defaults to True.
        """
        self.sign = sign

    def __str__(self):
        return f"Infinity({'positive' if self.sign else 'negative'})"

    def __repr__(self):
        return f"Infinity({'positive' if self.sign else 'negative'})"

    def __add__(self, arg):
        if isinstance(arg, Union[int, float, Decimal]): return Infinity(self.sign)
        if isinstance(arg, Infinity): return Undefined() if (self.sign ^ arg.sign) else Infinity(self.sign)
        if isinstance(arg, Complex): return Complex(Infinity(self.sign) + arg.real, arg.imaginary)
        if isinstance(arg, Undefined): return Undefined()
        raise ArgTypeError()

    def __radd__(self, arg):
        if isinstance(arg, Union[int, float, Decimal]): return Infinity(self.sign)
        if isinstance(arg, Complex): return Complex(Infinity(self.sign) + arg.real, arg.imaginary)
        raise ArgTypeError()

    def __iadd__(self, arg):
        if isinstance(arg, Union[int, float, Decimal]): return Infinity(self.sign)
        if isinstance(arg, Infinity): return Undefined() if (self.sign ^ arg.sign) else self
        if isinstance(arg, Complex): return Complex(Infinity(self.sign) + arg.real, arg.imaginary)
        if isinstance(arg, Undefined): return Undefined()
        raise ArgTypeError()

    def __sub__(self, arg):
        if isinstance(arg, Union[int, float, Decimal]): return Infinity(self.sign)
        if isinstance(arg, Infinity): return Undefined() if not (self.sign ^ arg.sign) else Infinity(self.sign)
        if isinstance(arg, Complex): return Complex(Infinity(self.sign) - arg.real, -arg.imaginary)
        if isinstance(arg, Undefined): return Undefined()
        raise ArgTypeError()

    def __rsub__(self, arg):
        if isinstance(arg, Union[int, float, Decimal]): return Infinity(not self.sign)
        if isinstance(arg, Complex): return Complex(-self + arg.real, arg.imaginary)
        raise ArgTypeError()

    def __isub__(self, arg):
        if isinstance(arg, Union[int, float, Decimal]): return Infinity(self.sign)
        if isinstance(arg, Infinity): return Undefined() if not (self.sign ^ arg.sign) else Infinity(self.sign)
        if isinstance(arg, Complex): return Complex(self - arg.real, -arg.imaginary)
        if isinstance(arg, Undefined): return Undefined()
        raise ArgTypeError()

    def __mul__(self, arg):
        if isinstance(arg, Infinity): return Infinity(not (self.sign ^ arg.sign))
        if isinstance(arg, Union[int, float, Decimal]): return Infinity(self.sign and arg > 0) if arg != 0 else Undefined()
        if isinstance(arg, Complex): return Complex(arg.real * self, arg.imaginary * self)
        if isinstance(arg, Undefined): return Undefined()
        raise ArgTypeError()

    def __rmul__(self, arg):
        if isinstance(arg, Union[int, float, Decimal]): return Infinity(self.sign and arg > 0) if arg != 0 else Undefined()
        if isinstance(arg, Complex): return Complex(arg.real * self, arg.imaginary * self)
        raise ArgTypeError()

    def __imul__(self, arg):
        if isinstance(arg, Infinity): return Infinity(not (self.sign ^ arg.sign))
        if isinstance(arg, Union[int, float, Decimal]): return Infinity(self.sign and arg > 0) if arg != 0 else Undefined()
        if isinstance(arg, Complex): return Complex(arg.real * self, arg.imaginary * self)
        if isinstance(arg, Undefined): return Undefined()
        raise ArgTypeError()

    def __truediv__(self, arg):
        if isinstance(arg, Infinity) or isinstance(arg, Undefined): return Undefined()
        if isinstance(arg, Union[int, float, Decimal]): return Infinity(self.sign ^ arg >= 0)
        if isinstance(arg, Complex): return Complex(Infinity(self.sign ^ arg.real >= 0), Infinity(self.sign ^ arg.imaginary >= 0))
        raise ArgTypeError()

    def __floordiv__(self, arg):
        if isinstance(arg, Infinity) or isinstance(arg, Undefined): return Undefined()
        if isinstance(arg, Union[int, float, Decimal]): return Infinity(self.sign ^ arg >= 0)
        if isinstance(arg, Complex): return Complex(Infinity(self.sign ^ arg.real >= 0), Infinity(self.sign ^ arg.imaginary >= 0))
        raise ArgTypeError()

    def __rtruediv__(self, arg):
        if isinstance(arg, Infinity): return Undefined()
        if isinstance(arg, Union[int, float, Decimal]): return 0
        raise ArgTypeError()

    def __rfloordiv__(self, arg):
        if isinstance(arg, Infinity): return Undefined()
        if isinstance(arg, Union[int, float, Decimal]): return 0
        raise ArgTypeError()

    def __idiv__(self, arg):
        if isinstance(arg, Infinity) or isinstance(arg, Undefined): return Undefined()
        if isinstance(arg, Union[int, float, Decimal]): return self
        if isinstance(arg, Complex):
            temp = 1 / arg
            return Complex(self * temp.real, self * temp.imaginary)
        raise ArgTypeError()

    def __ifloordiv__(self, arg):
        if isinstance(arg, Infinity) or isinstance(arg, Undefined): return Undefined()
        if isinstance(arg, Union[int, float, Decimal]): return self
        if isinstance(arg, Complex):
            temp = 1 / arg
            return Complex(self * temp.real, self * temp.imaginary)
        raise ArgTypeError()

    def __neg__(self):
        return Infinity(not self.sign)

    def __pow__(self, p):
        if not p: return Undefined()
        if not isinstance(p, Union[int, float, Decimal]): raise ArgTypeError("Must be a numerical value.")
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