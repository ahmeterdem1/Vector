from ..variable import Variable, SQRT, EXP, SIN, COS, ARCSIN, LOG2, LN, SIG
from .infinity import Infinity, Complex
from ..utils import *
import threading
import logging
from typing import Union, List, Callable
from decimal import Decimal

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

def sqrt(arg, resolution: int = 10):
    """
        Computes the square root of a numerical argument with a specified resolution.

        Args:
            arg: The numerical value or type for which the square root is to be computed.
            resolution (int, optional): The number of iterations for the approximation. Defaults to 10.

        Returns:
            int, float, Decimal, Complex, Infinity, Undefined: The square root of the argument.

        Raises:
            ArgTypeError: If the argument is not of a valid type.
            RangeError: If the resolution is not a positive integer.
    """
    if isinstance(arg, Union[int, float, Decimal]):
        if resolution < 1:
            raise RangeError("Resolution must be a positive integer")
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
    if isinstance(arg, Variable):
        if resolution < 1:
            raise RangeError("Resolution must be a positive integer")
        c = True if arg.value >= 0 else False
        arg = abs(arg.value)
        digits = 0
        temp = arg.value
        first_digit = 0
        while temp != 0:
            first_digit = temp
            temp //= 10
            digits += 1

        estimate = (first_digit // 2 + 1) * pow(10, digits // 2)

        for k in range(resolution):
            estimate = (estimate + arg.value / estimate) / 2

        # Yes we can return the negatives too.
        if c:
            estimate = Variable(estimate)
        else:
            estimate = Variable(Complex(0, estimate))

        estimate.operation = SQRT
        estimate.backward = (arg, Variable(1))
        return estimate
    if isinstance(arg, Complex):
        return arg.sqrt()
    if isinstance(arg, Infinity):
        if arg.sign: return Infinity()
        return sqrt(Complex(0, 1)) * Infinity()
    if isinstance(arg, Undefined):
        return Undefined()
    raise ArgTypeError()

def Range(low: Union[int, float, Decimal, Infinity, Undefined],
          high: Union[int, float, Decimal, Infinity, Undefined],
          step: Union[int, float, Decimal, Infinity, Undefined] = 1):
    """
        Generator function that yields a range of numerical values from 'low' to 'high' (inclusive) with a specified step.

        Args:
            low: The starting value of the range.
            high: The ending value of the range.
            step: The step size between consecutive values. Default is 1.

        Yields:
            The next value in the range sequence.

        Raises:
            ArgTypeError: If 'low' or 'high' is not a numerical value.
            RangeError: If the direction of the range is ambiguous or if 'high' and 'low' are equal.

        Note:
            The direction of the range is determined by the relative values of 'low', 'high', and 'step'.
            If 'step' is positive, the range moves from 'low' to 'high'. If 'step' is negative, the range moves from 'high' to 'low'.
    """
    if not ((isinstance(low, Union[int, float, Decimal, Infinity, Undefined]))
            and (isinstance(high, Union[int, float, Decimal, Infinity, Undefined]))):
        raise ArgTypeError("Must be a numerical value.")
    if not ((high < low) ^ (step > 0)): raise RangeError()

    while (high < low) ^ (step > 0) and high != low:
        yield low
        low += step

def abs(arg: Union[int, float, Decimal, Infinity, Undefined, Variable]):
    """
        Returns the absolute value of a numerical argument.

        Args:
            arg: The numerical value whose absolute value is to be computed.

        Returns:
            The absolute value of the argument.

        Raises:
            ArgTypeError: If the argument is not a numerical value.
    """
    if not isinstance(arg, Union[int, float, Decimal, Infinity, Undefined, Variable]):
        raise ArgTypeError("Must be a numerical value.")

    return arg if (arg >= 0) else -arg

def __cumdiv(x: Union[int, float, Decimal, Infinity, Undefined, Variable], power: int):
    """
        Computes the cumulative division of a numerical value 'x' by a specified power, cumulative
        division being a related term at Taylor Series.

        Args:
            x: The numerical value to be divided cumulatively.
            power (int): The power by which 'x' is to be divided cumulatively.

        Returns:
            float: The result of the cumulative division.

        Raises:
            ArgTypeError: If 'x' is not a numerical value.

        Notes:
            This function optimizes for accuracy, not speed.
    """
    if not isinstance(x, Union[int, float, Decimal, Infinity, Undefined, Variable]):
        raise ArgTypeError("Must be a numerical value.")

    result: float = 1
    for k in range(power, 0, -1):
        result *= x / k
    return result

def e(exponent: Union[int, float, Decimal, Infinity, Undefined, Variable], resolution: int = 15):
    """
        Computes the approximation of the mathematical constant 'e' raised to the power of the given exponent.

        Args:
            exponent: The numerical exponent for 'e'.
            resolution (int, optional): The resolution for the approximation. Defaults to 15.

        Returns:
            float: The approximation of 'e' raised to the power of the given exponent.

        Raises:
            ArgTypeError: If 'exponent' is not a numerical value.
            RangeError: If 'resolution' is not a positive integer.
    """
    if not (isinstance(resolution, int) and resolution >= 1): raise RangeError("Resolution must be a positive integer")
    if not isinstance(exponent, Union[int, float, Decimal, Infinity, Undefined, Variable]):
        raise ArgTypeError("Must be a numerical value.")

    if isinstance(exponent, Variable):
        sum = 1
        for k in range(resolution, 0, -1):
            sum += __cumdiv(exponent.value, k)
        sum = Variable(sum)
        sum.operation = EXP
        sum.backward = (exponent, Variable(1),)
        return sum

    sum = 1
    for k in range(resolution, 0, -1):
        sum += __cumdiv(exponent, k)
    return sum

def sin(angle: Union[int, float, Decimal, Infinity, Undefined, Variable], resolution: int = 15):
    """
        Computes the sine of the given angle using Taylor Series approximation.

        Args:
            angle: The angle in degrees.
            resolution (int, optional): The resolution for the approximation. Defaults to 15.

        Returns:
            float: The sine value of the given angle.

        Raises:
            ArgTypeError: If 'angle' is not a numerical value.
            RangeError: If 'resolution' is not a positive integer.
    """
    if not (isinstance(resolution, int) and resolution >= 1): raise RangeError("Resolution must be a positive integer")
    if not isinstance(angle, Union[int, float, Decimal, Infinity, Undefined, Variable]):
        raise ArgTypeError("Must be a numerical value.")

    if isinstance(angle, Variable):
        radian = (2 * PI * (angle.value % 360 / 360)) % (2 * PI)
        result = 0
        if not resolution % 2:
            resolution += 1
        for k in range(resolution, 0, -2):
            result = result + __cumdiv(radian, k) * (-1) ** ((k - 1) / 2)

        result = Variable(result)
        result.operation = SIN
        result.backward = (angle, Variable(1))
        return result

    # Below calculation can be optimized away. It is done so in Vectorgebra/C++.
    radian = (2 * PI * (angle % 360 / 360)) % (2 * PI)
    result = 0
    if not resolution % 2:
        resolution += 1
    for k in range(resolution, 0, -2):
        result = result + __cumdiv(radian, k) * (-1)**((k - 1) / 2)

    return result

def cos(angle: Union[int, float, Decimal, Infinity, Undefined, Variable], resolution: int = 16):
    """
        Computes the cosine of the given angle using Taylor Series approximation.

        Args:
            angle: The angle in degrees.
            resolution (int, optional): The resolution for the approximation. Defaults to 16.

        Returns:
            float: The cosine value of the given angle.

        Raises:
            ArgTypeError: If 'angle' is not a numerical value.
            RangeError: If 'resolution' is not a positive integer.
        """
    if not (isinstance(resolution, int) and resolution >= 1): raise RangeError("Resolution must be a positive integer")
    if not isinstance(angle, Union[int, float, Decimal, Infinity, Undefined, Variable]):
        raise ArgTypeError("Must be a numerical value.")

    if isinstance(angle, Variable):
        radian = (2 * PI * (angle.value % 360 / 360)) % (2 * PI)
        result = 1

        if resolution % 2:
            resolution += 1

        for k in range(resolution, 0, -2):
            result = result + __cumdiv(radian, k) * (-1) ** (k / 2)

        result = Variable(result)
        result.operation = COS
        result.backward = (angle, Variable(1),)
        return result

    radian = (2 * PI * (angle % 360 / 360)) % (2 * PI)
    result = 1

    if resolution % 2:
        resolution += 1

    for k in range(resolution, 0, -2):
        result = result + __cumdiv(radian, k) * (-1)**(k / 2)
    return result

def tan(angle: Union[int, float, Decimal, Infinity, Undefined, Variable], resolution: int = 16):
    """
        Computes the tangent of the given angle.

        Args:
            angle: The angle in degrees.
            resolution (int, optional): The resolution for the approximation. Defaults to 16.

        Returns:
            The tangent value of the given angle.

        Raises:
            ArgTypeError: If 'angle' is not a numerical value.
            RangeError: If 'resolution' is not a positive integer.
    """
    if not (isinstance(resolution, int) and resolution >= 1):
        raise RangeError("Resolution must be a positive integer")
    try:
        return sin(angle, resolution - 1) / cos(angle, resolution)
        # Because of the error amount, probably cos will never be zero.
    except ZeroDivisionError:
        # sin * cos is positive in this area
        if 90 >= (angle % 360) >= 0 or 270 >= (angle % 360) >= 180:
            return Infinity()
        return Infinity(False)

def cot(angle: Union[int, float, Decimal, Infinity, Undefined, Variable], resolution: int = 16):
    """
        Computes the cotangent of the given angle.

        Args:
            angle: The angle in degrees.
            resolution (int, optional): The resolution for the approximation. Defaults to 16.

        Returns:
            The tangent value of the given angle.

        Raises:
            ArgTypeError: If 'angle' is not a numerical value.
            RangeError: If 'resolution' is not a positive integer.
    """
    try:
        return 1 / tan(angle, resolution)
    except ZeroDivisionError:  # Probably will never happen
        return None

def sinh(x: Union[int, float, Decimal, Infinity, Undefined, Variable], resolution: int = 15):
    """
        Computes the hyperbolic sine of the given number.

        Args:
            x: The number.
            resolution (int, optional): The resolution for the approximation. Defaults to 15.

        Returns:
            float: The hyperbolic sine value of the given number.

        Raises:
            RangeError: If 'resolution' is not a positive integer.
            ArgTypeError: If 'x' is not a numerical value.
        """
    return (e(x, resolution) - e(-x, resolution)) / 2

def cosh(x: Union[int, float, Decimal, Infinity, Undefined, Variable], resolution: int = 15):
    """
        Computes the hyperbolic cosine of the given number.

        Args:
            x: The number.
            resolution (int, optional): The resolution for the approximation. Defaults to 15.

        Returns:
            float: The hyperbolic cosine value of the given number.

        Raises:
            RangeError: If 'resolution' is not a positive integer.
            ArgTypeError: If 'x' is not a numerical value.
        """
    return (e(x, resolution) + e(-x, resolution)) / 2

def tanh(x: Union[int, float, Decimal, Infinity, Undefined, Variable], resolution: int = 15):
    """
        Computes the hyperbolic tangent of the given number.

        Args:
            x: The number.
            resolution (int, optional): The resolution for the approximation. Defaults to 15.

        Returns:
            float: The hyperbolic tangent value of the given number.

        Raises:
            RangeError: If 'resolution' is not a positive integer.
            ArgTypeError: If 'x' is not a numerical value.
        """
    try:
        return sinh(x, resolution) / cosh(x, resolution)
        # Indeed cosh is non-negative
    except ZeroDivisionError:
        return None

def coth(x: Union[int, float, Decimal, Infinity, Undefined, Variable], resolution: int = 15):
    """
        Computes the hyperbolic cotangent of the given number.

        Args:
            x: The number.
            resolution (int, optional): The resolution for the approximation. Defaults to 15.

        Returns:
            float: The hyperbolic cotangent value of the given number.

        Raises:
            RangeError: If 'resolution' is not a positive integer.
            ArgTypeError: If 'x' is not a numerical value.
        """
    try:
        return cosh(x, resolution) / sinh(x, resolution)
    except ZeroDivisionError:
        if x >= 0:
            return Infinity()
        return Infinity(False)

def arcsin(x: Union[int, float, Decimal, Undefined, Variable], resolution: int = 20):
    """
        Computes the arc sine of the given number in degrees.

        Args:
            x: The number.
            resolution (int, optional): The resolution for the approximation. Defaults to 20.

        Returns:
            float: The arc sine value of the given number in degrees.

        Raises:
            ArgTypeError: If 'x' is not a numerical value.
            RangeError: If 'x' is not in the range [-1, 1], or if 'resolution' is not a positive integer.
    """
    if not isinstance(x, Union[int, float, Decimal, Infinity, Undefined, Variable]):
        raise ArgTypeError("Must be a numerical value.")
    if not (-1 <= x <= 1):
        raise RangeError()
    if resolution < 1:
        raise RangeError("Resolution must be a positive integer")

    if isinstance(x, Variable):
        c = 1
        sol = x.value
        for k in range(1, resolution):
            c *= (2 * k - 1) / (2 * k)
            sol += c * x.value ** (2 * k + 1) / (2 * k + 1)
        sol = Variable(sol * 360 / (2 * PI))
        sol.operation = ARCSIN
        sol.backward = (x, Variable(1),)
        return sol

    c = 1
    sol = x
    for k in range(1, resolution):
        c *= (2 * k - 1) / (2 * k)
        sol += c * x**(2 * k + 1) / (2 * k + 1)
    return sol * 360 / (2 * PI)

def arccos(x: Union[int, float, Decimal, Undefined, Variable], resolution: int = 20):
    """
        Computes the arc cosine of the given number in degrees.

        Args:
            x: The number.
            resolution (int, optional): The resolution for the approximation. Defaults to 20.

        Returns:
            float: The arc cosine value of the given number in degrees.

        Raises:
            ArgTypeError: If 'x' is not a numerical value.
            RangeError: If 'x' is not in the range [-1, 1], or if 'resolution' is not a positive integer.
    """
    if not (-1 <= x <= 1):
        raise RangeError()
    if resolution < 1:
        raise RangeError("Resolution must be a positive integer")

    return 90 - arcsin(x, resolution)

def log2(x: Union[int, float, Decimal, Infinity, Undefined, Variable], resolution: int = 15):
    """
        Computes the base-2 logarithm of the given number.

        Args:
            x: The number.
            resolution (int, optional): The resolution for the approximation. Defaults to 15.

        Returns:
            float: The base-2 logarithm of the given number.

        Raises:
            ArgTypeError: If 'x' is not a numerical value.
            RangeError: If 'x' is less than or equal to 0, or if 'resolution' is not a positive integer.
    """
    if not isinstance(x, Union[int, float, Decimal, Infinity, Undefined, Variable]):
        raise ArgTypeError("Must be a numerical value.")
    if x <= 0:
        raise RangeError()
    if resolution < 1:
        raise RangeError()

    if isinstance(x, Variable):
        count = 0
        factor = 1
        res = x.value
        if res < 1:
            factor = -1
            res = 1 / res

        while res > 2:
            res = res / 2
            count += 1

        for i in range(1, resolution + 1):
            res = res ** 2
            if res >= 2:
                count += 1 / (2 ** i)
                res /= 2

        res = Variable(factor * count)
        res.operation = LOG2
        res.backward = (x, Variable(1),)
        return res

    count = 0
    factor = 1
    if x < 1:
        factor = -1
        x = 1 / x

    while x > 2:
        x = x / 2
        count += 1

    for i in range(1, resolution + 1):
        x = x * x
        if x >= 2:
            count += 1 / (2**i)
            x = x / 2

    return factor * count

def ln(x: Union[int, float, Decimal, Infinity, Undefined, Variable], resolution: int = 15):
    """
        Computes the base-e logarithm of the given number.

        Args:
            x: The number.
            resolution (int, optional): The resolution for the approximation. Defaults to 15.

        Returns:
            float: The base-e logarithm of the given number.

        Raises:
            ArgTypeError: If 'x' is not a numerical value.
            RangeError: If 'x' is less than or equal to 0, or if 'resolution' is not a positive integer.
        """

    if isinstance(x, Variable):
        res = Variable(log2(x.value, resolution) / log2E)
        res.operation = LN
        res.backward = (x, Variable(1),)
        return res

    return log2(x, resolution) / log2E

def log10(x: Union[int, float, Decimal, Infinity, Undefined, Variable], resolution: int = 15):
    """
        Computes the base-10 logarithm of the given number.

        Args:
            x: The number.
            resolution (int, optional): The resolution for the approximation. Defaults to 15.

        Returns:
            float: The base-10 logarithm of the given number.

        Raises:
            ArgTypeError: If 'x' is not a numerical value.
            RangeError: If 'x' is less than or equal to 0, or if 'resolution' is not a positive integer.
        """
    return log2(x, resolution) / log2_10

def log(x: Union[int, float, Decimal, Infinity, Undefined, Variable],
        base: Union[int, float, Decimal, Variable] = 2, resolution: int = 15):
    """
        Computes the logarithm of a number 'x' with respect to the specified base.

        Args:
            x: The number.
            base: The base of the logarithm. Defaults to 2.
            resolution (int, optional): The resolution for the approximation. Defaults to 15.

        Returns:
            float: The logarithm of 'x' with respect to the specified base.

        Raises:
            ArgTypeError: If 'base' is not a numerical value.
            RangeError: If 'base' is less than or equal to 0 or equal to 1.
    """
    if not (isinstance(base, Union[int, float, Decimal, Variable])):
        raise ArgTypeError("Must be a numerical value.")
    if base <= 0 or base == 1:
        raise RangeError()

    return log2(x, resolution) / log2(base, resolution)

def __find(f: Callable,
           low: Union[int, float, Decimal, Variable],
           high: Union[int, float, Decimal, Variable],
           search_step: Union[int, float, Decimal, Variable],
           res: Union[int, float, Decimal, Variable] = 0.0001):
    """
        Find a zero of a function within a specified range using recursive search.

        Args:
        - f (Callable): The function for which a zero are to be found.
        - low: The lower bound of the search range.
        - high: The upper bound of the search range.
        - search_step: The step size for searching.
        - res: The resolution for finding zeros. Defaults to Decimal(0.0001).

        Returns:
        - The approximate zero found within the specified range, or None if not found.

        Raises:
        - ArgTypeError: If the arguments are not numerical values.

        Notes:
            This function performs a recursive search within the given range to find a zero of the provided function.
    """
    if not ((isinstance(low, Union[int, float, Decimal, Variable]))
            and (isinstance(high, Union[int, float, Decimal, Variable]))
            and (isinstance(search_step, Union[int, float, Decimal, Variable]))
            and (isinstance(res, Union[int, float, Decimal, Variable]))):
        raise ArgTypeError("Must be a numerical value.")
    if not isinstance(f, Callable):
        raise ArgTypeError(f"f must be a callable.")

    global __results
    last_sign = (f(low) >= 0)
    for x in Range(low, high, search_step):
        value = f(x)
        temp_sign = (value >= 0)
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

def solve(f: Callable,
          low: Union[int, float, Decimal, Variable] = -50,
          high: Union[int, float, Decimal, Variable] = 50,
          search_step: Union[int, float, Decimal, Variable] = 0.1,
          res: Union[int, float, Decimal, Variable] = 0.0001) -> List[Union[int, float, Decimal, Variable]]:
    """
        Find approximate zeros of a given function within a specified range.

        Args:
        - f (Callable): The function for which zeros are to be found.
        - low: The lower bound of the range. Defaults to -50.
        - high: The upper bound of the range. Defaults to 50.
        - search_step: The step size for searching. Defaults to 0.1.
        - res: The resolution for finding zeros. Defaults to Decimal(0.0001).

        Returns:
        - A list of approximate zeros found within the specified range.

        Raises:
        - ArgTypeError: If the arguments are not numerical values or if the function `f` is not callable.
        - RangeError: If the range limits are invalid or if the search step or resolution is non-positive.

        The function uses threading to improve efficiency in finding zeros within the specified range.
    """
    if not ((isinstance(low, Union[int, float, Decimal]))
            and (isinstance(high, Union[int, float, Decimal]))
            and (isinstance(search_step, Union[int, float, Decimal]))
            and (isinstance(res, Union[int, float, Decimal]))):
        raise ArgTypeError("Must be a numerical value.")

    # I couldn't find any other way to check it
    if not isinstance(f, Callable): raise ArgTypeError("f must be a callable")

    if high <= low:
        raise RangeError()
    if search_step <= 0:
        raise RangeError()
    if res <= 0 or res >= 1:
        raise RangeError()

    zeroes = []
    thread_list = []
    last_sign = (f(low) >= 0)
    for x in Range(low, high, search_step):
        value = f(x)
        temp_sign = (value >= 0)
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

def derivative(f: Callable, x: Union[int, float, Decimal, Variable], h: Union[int, float, Decimal, Variable] = 0.0000000001):
    """
        Computes the derivative of a function 'f' at a point 'x' using the finite difference method.

        Args:
            f (Callable): The function for which the derivative is to be computed.
            x: The point at which the derivative is evaluated.
            h: The step size for the finite difference method.
                Defaults to 0.0000000001.

        Returns:
            float: The derivative of 'f' at point 'x' using the finite difference method.

        Raises:
            ArgTypeError: If 'x' or 'h' is not a numerical value, or if 'f' is not callable.
    """
    if not ((isinstance(x, Union[int, float, Decimal, Variable]))
            and (isinstance(h, Union[int, float, Decimal, Variable]))):
        raise ArgTypeError("Must be a numerical value.")
    if not isinstance(f, Callable):
        raise ArgTypeError("f must be a callable.")

    return (f(x + h) - f(x)) / h

def integrate(f: Callable,
              a: Union[int, float, Decimal, Variable],
              b: Union[int, float, Decimal, Variable],
              delta: Union[int, float, Decimal, Variable] = 0.01):
    """
        Numerically integrates a function 'f' over the interval [a, b] using the rectangle method.

        Args:
            f (Callable): The function to be integrated.
            a: The lower limit of integration.
            b: The upper limit of integration.
            delta: The width of each rectangle used in the integration.
                Defaults to 0.01.

        Returns:
            float: The approximate value of the integral of 'f' over the interval [a, b].

        Raises:
            ArgTypeError: If 'a', 'b', or 'delta' is not a numerical value, or if 'f' is not callable.
    """
    if not ((isinstance(a, Union[int, float, Decimal, Variable]))
            and (isinstance(b, Union[int, float, Decimal, Variable]))
            and (isinstance(delta, Union[int, float, Decimal, Variable]))):
        raise ArgTypeError("Must be a numerical value.")

    if not isinstance(f, Callable): raise ArgTypeError("f must be a callable.")

    if a == b:
        return 0
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

def findsol(f: Callable, x: Union[int, float, Decimal, Variable] = 0, resolution: int = 15):
    """
        Find a numerical solution to the equation f(x) = 0 using Newton's method.

        Args:
        - f (Callable): The function for which the root is sought.
        - x: The initial guess for the root. Default is 0.
        - resolution (int, optional): The number of iterations to refine the solution. Default is 15.

        Returns:
        - x: The numerical approximation of the root.

        Notes:
            This function applies Newton's method to iteratively find a solution to the equation f(x) = 0.
            It uses the derivative of the function f(x) to approximate the root.
            The initial guess for the root is provided by the x parameter, and the resolution parameter determines
            the number of iterations used to refine the solution.
    """
    if not isinstance(f, Callable):
        raise ArgTypeError("f must be a callable")
    if resolution < 1 or not isinstance(resolution, int):
        raise RangeError("Resolution must be a positive integer")

    for k in range(resolution):
        try:
            x = x - (f(x) / derivative(f, x))
        except ZeroDivisionError:
            x = Infinity(f(x) < 0)
    return x

def sigmoid(x: Union[int, float, Decimal, Infinity, Undefined, Variable], a: Union[int, float, Decimal, Infinity, Undefined, Variable] = 1):
    """
        Compute the sigmoid function for the given input value(s).

        Args:
        - x: The input value for which the sigmoid function is computed.
        - a: The scaling factor. Defaults to 1.

        Returns:
        - float: The value of the sigmoid function evaluated at x.

        Raises:
        - ArgTypeError: If the input arguments are not numerical values.
    """
    if not((isinstance(a, Union[int, float, Decimal, Infinity, Undefined, Variable]))
           and (isinstance(x, Union[int, float, Decimal, Infinity, Undefined, Variable]))):
        raise ArgTypeError("Must be a numerical value.")

    if isinstance(x, Variable):
        res = Variable(1 / (1 + e(-a*x.value)))
        res.operation = SIG
        res.backward = (x, Variable(1),)
        return res

    return 1 / (1 + e(-a*x))

def ReLU(x: Union[int, float, Decimal, Infinity, Undefined, Variable],
         leak: Union[int, float, Decimal, Infinity, Undefined, Variable] = 0,
         cutoff: Union[int, float, Decimal, Infinity, Undefined, Variable] = 0):
    """
        Implements the Rectified Linear Unit (ReLU) activation function.

        Args:
            x: The input value to apply the ReLU function to.
            leak: The slope for negative inputs (default is 0).
            cutoff: The threshold value where ReLU starts to have an effect (default is 0).

        Returns:
            The result of applying the ReLU function to the input value.

        Raises:
            ArgTypeError: If any of the input arguments is not a numerical value.
    """
    if not ((isinstance(x, Union[int, float, Decimal, Infinity, Undefined, Variable]))
            and (isinstance(leak, Union[int, float, Decimal, Infinity, Undefined, Variable]))
            and (isinstance(cutoff, Union[int, float, Decimal, Infinity, Undefined, Variable]))):
        raise ArgTypeError("Must be a numerical value.")

    if x >= cutoff:
        return x
    elif x < 0:
        return leak * x
    else:
        return Variable(cutoff) if isinstance(x, Variable) else cutoff

def deriv_relu(x: Union[int, float, Decimal, Infinity, Undefined, Variable],
               leak: Union[int, float, Decimal, Infinity, Undefined, Variable] = 0,
               cutoff: Union[int, float, Decimal, Infinity, Undefined, Variable] = 0):
    """
        Computes the derivative of the Rectified Linear Unit (ReLU) activation function.

        Args:
            x: The input value to compute the derivative for.
            leak: The slope for negative inputs (default is 0).
            cutoff: The threshold value where ReLU starts to have an effect (default is 0).

        Returns:
            The derivative of the ReLU function with respect to the input value.

        Raises:
            ArgTypeError: If any of the input arguments is not a numerical value.
    """
    if not ((isinstance(x, Union[int, float, Decimal, Infinity, Undefined, Variable]))
            and (isinstance(leak, Union[int, float, Decimal, Infinity, Undefined, Variable]))
            and (isinstance(cutoff, Union[int, float, Decimal, Infinity, Undefined, Variable]))):
        raise ArgTypeError("Must be a numerical value.")
    return 1 if x >= cutoff else 0 if x >= 0 else leak

def Sum(f: Callable,
        a: Union[int, float, Decimal, Infinity, Undefined, Variable],
        b: Union[int, float, Decimal, Infinity, Undefined, Variable],
        step: Union[int, float, Decimal, Infinity, Undefined, Variable] = 0.01,
        control: bool = False,
        limit: Union[int, float, Decimal, Infinity, Undefined, Variable] = 0.000001):
    """
        Computes the sum of a function over a given interval.

        Args:
            f (Callable): The function to be summed.
            a: The lower bound of the interval.
            b: The upper bound of the interval.
            step: The step size for sampling points within the interval (default is 0.01).
            control (bool, optional): A flag indicating whether to use control over derivative values to stop the summing process (default is False).
            limit: The limit value to control the derivative when 'control' is True (default is 0.000001).

        Returns:
            Union[int, float, Decimal, Infinity, Undefined]: The sum of the function over the interval [a, b).

        Raises:
            ArgTypeError: If any of the input arguments is not a numerical value.
            RangeError: If the upper bound 'b' is less than the lower bound 'a'.
    """

    # Yes, with infinities this can blow up. It is the users problem to put infinities in.
    if not ((isinstance(a, Union[int, float, Decimal, Infinity, Undefined, Variable]))
            and (isinstance(b, Union[int, float, Decimal, Infinity, Undefined, Variable]))
            and (isinstance(step, Union[int, float, Decimal, Infinity, Undefined, Variable]))
            and (isinstance(limit, Union[int, float, Decimal, Infinity, Undefined, Variable]))):
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

def factorial(x: int = 0):
    """
        Computes the factorial of a non-negative integer.

        Args:
            x (int, optional): The integer whose factorial is to be computed (default is 0).

        Returns:
            int: The factorial of the input integer 'x'.

        Raises:
            ArgTypeError: If the input 'x' is not an integer.
            RangeError: If the input 'x' is negative.
        """
    if not isinstance(x, int):
        raise ArgTypeError("Must be an integer.")
    if x < 0:
        raise RangeError()
    if x <= 1:
        return 1
    mul = 1
    for k in range(2, x):
        mul *= k
    return mul * x

def permutation(x: int = 1, y: int = 1):
    """
        Computes the permutation of 'x' items taken 'y' at a time.

        Args:
            x (int, optional): The total number of items (default is 1).
            y (int, optional): The number of items to be taken at a time (default is 1).

        Returns:
            int: The number of permutations of 'x' items taken 'y' at a time.

        Raises:
            ArgTypeError: If 'x' or 'y' is not an integer.
            RangeError: If 'x' or 'y' is less than 1, or 'y' is greater than 'x'.
    """
    if not (isinstance(x, int) and isinstance(y, int)): raise ArgTypeError("Must be an integer.")
    if x < 1 or y < 1 or y > x: raise RangeError()
    result = 1
    for v in range(y + 1, x + 1):
        result *= v
    return result

def combination(x: int = 0, y: int = 0):
    """
        Computes the combination of 'x' items choose 'y' at a time.

        Args:
            x (int, optional): The total number of items (default is 0).
            y (int, optional): The number of items to choose at a time (default is 0).

        Returns:
            float: The number of combinations of 'x' items choose 'y' at a time.

        Raises:
            ArgTypeError: If 'x' or 'y' is not an integer.
            RangeError: If 'x' or 'y' is less than 0, or 'y' is greater than 'x'.
    """
    if not (isinstance(x, int) and isinstance(y, int)): raise ArgTypeError("Must be an integer.")
    if x < 0 or y < 0 or y > x: raise RangeError()
    result = 1
    count = 1
    for v in range(y + 1, x + 1):
        result *= v / count
        count += 1
    return result

def multinomial(n: int = 0, *args):
    """
        Computes the multinomial coefficient for the given parameters.

        Args:
            n (int, optional): The total number of items in the multinomial coefficient (default is 0).
            *args (int): The number of items for each partition in the multinomial coefficient.

        Returns:
            float: The multinomial coefficient value.

        Raises:
            ArgTypeError: If 'n' or any of the partition arguments is not an integer.
            RangeError: If 'n' is negative or if any of the partition arguments is negative, or if the sum of partition arguments is not equal to 'n'.
    """
    if not isinstance(n, int): raise ArgTypeError("Must be an integer.")
    sum = 0
    for k in args:
        if not isinstance(k, int): raise ArgTypeError()
        if k < 0: raise RangeError()
        sum += k
    c = [k for k in args]
    if sum != n: raise RangeError("Sum of partitions must be equal to n")
    result = 1
    N = len(c)
    while n != 1:
        result *= n
        for k in range(N):
            if not c[k]:
                continue
            result /= c[k]
            c[k] -= 1
        n -= 1
    return result

def binomial(n: int, k: int, p: Union[int, float, Decimal, Variable]):
    """
        Computes the probability of obtaining 'k' successes in 'n' independent Bernoulli trials,
        each with a success probability of 'p'.

        Args:
            n (int): The total number of trials.
            k (int): The number of successes.
            p: The probability of success in each trial. Must be in the range [0, 1].

        Returns:
            float: The probability of obtaining 'k' successes in 'n' trials with success probability 'p'.

        Raises:
            ArgTypeError: If 'n' or 'k' is not an integer, or if 'p' is not a numerical value.
            RangeError: If 'p' is outside the range [0, 1].
    """
    if not (isinstance(n, int) and isinstance(k, int)):
        raise ArgTypeError("Must be an integer.")
    if not (isinstance(p, Union[int, float, Decimal, Variable])):
        raise ArgTypeError("Must be a numerical value.")
    if p < 0 or p > 1:
        raise RangeError("Probability cannot be negative or bigger than 1")
    return combination(n, k) * p**k * (1-p)**k

def geometrical(n: int, p: Union[int, float, Decimal, Variable]):
    """
        Computes the probability of the nth trial being the first success in a sequence of independent
        Bernoulli trials, each with a success probability of 'p'.

        Args:
            n (int): The trial number for which the probability is calculated. Must be a non-negative integer.
            p: The probability of success in each trial. Must be in the range [0, 1].

        Returns:
            Union[float, Undefined]: The probability of the nth trial being the first success, or Undefined if 'n' is 0.

        Raises:
            ArgTypeError: If 'n' is not an integer, or if 'p' is not a numerical value.
            RangeError: If 'p' is outside the range [0, 1], or if 'n' is negative.
    """
    if not isinstance(n, int):
        raise ArgTypeError("Must be an integer.")
    if not (isinstance(p, Union[int, float, Decimal, Variable])):
        raise ArgTypeError("Must be a numerical value.")
    if p < 0 or p > 1:
        raise RangeError("Probability cannot be negative or bigger than 1")
    if n < 0:
        raise RangeError("Trial number cannot be negative")
    if n == 0:
        return Undefined()
    return p * (1-p)**(n-1)

def poisson(k: int, l: Union[int, float, Decimal, Variable]):
    """
        Computes the probability of observing 'k' events in a fixed interval of time or space,
        given that the average rate of occurrence of the event is 'l' events per unit interval.

        Args:
            k (int): The number of events to be observed. Must be a non-negative numerical value.
            l: The average rate of occurrence of the event per unit interval.
                Must be a non-negative numerical value.

        Returns:
            Union[float, Decimal]: The probability of observing 'k' events in the given interval, based on the Poisson distribution.

        Raises:
            ArgTypeError: If 'k' or 'l' is not a numerical value.
            RangeError: If 'k' or 'l' is negative.

        Notes:
            The Poisson distribution models the number of events that occur in a fixed interval
            of time or space, given a known average rate of occurrence ('l'). The probability of
            observing 'k' events in the interval is given by the formula: '(l^k * e^(-l)) / k!',
            where 'e' is the base of the natural logarithm and 'k!' is the factorial of 'k'.
    """
    if not ((isinstance(k, int))
            and (isinstance(l, Union[int, float, Decimal, Variable]))):
        raise ArgTypeError("Must be a numerical value.")
    if l < 0 or k < 0:
        raise RangeError()
    return l**k * e(-l) / factorial(k)

def normal(x: Union[int, float, Decimal, Infinity, Undefined, Variable], resolution: int = 15):
    """
        Computes the value of the standard normal probability density function (PDF) at a given point 'x'.

        Args:
            x: The point at which to evaluate the standard normal PDF.
                Must be a numerical value.
            resolution (int, optional): The resolution of calculations for intermediate steps. Defaults to 15.
                Must be a positive integer.

        Returns:
            Union[float, Decimal]: The value of the standard normal PDF at the given point 'x'.

        Raises:
            ArgTypeError: If 'x' is not a numerical value, or if 'resolution' is not an integer.
    """
    if not (isinstance(x, Union[int, float, Decimal, Infinity, Undefined, Variable])):
        raise ArgTypeError("Must be a numerical value.")
    if not isinstance(resolution, int):
        raise ArgTypeError("Must be an integer.")
    return e(-(x**2) / 2, resolution=resolution) / sqrt2pi

def gaussian(x: Union[int, float, Decimal, Infinity, Undefined, Variable],
             mean: Union[int, float, Decimal, Infinity, Undefined, Variable],
             sigma: Union[int, float, Decimal, Infinity, Undefined, Variable],
             resolution: int = 15):
    """
        Computes the value of the Gaussian (normal) distribution function at a given point 'x'.

        Args:
            x: The point at which to evaluate the Gaussian distribution.
                Must be a numerical value.
            mean: The mean of the Gaussian distribution.
                Must be a numerical value.
            sigma: The standard deviation of the Gaussian distribution.
                Must be a numerical value.
            resolution (int, optional): The resolution of calculations for intermediate steps. Defaults to 15.
                Must be a positive integer.

        Returns:
            Union[float, Decimal]: The value of the Gaussian distribution at the given point 'x'.

        Raises:
            ArgTypeError: If 'x', 'mean', or 'sigma' are not numerical values, or if 'resolution' is not an integer.
    """
    if not ((isinstance(x, Union[int, float, Decimal, Infinity, Undefined, Variable]))
            and (isinstance(mean, Union[int, float, Decimal, Infinity, Undefined, Variable]))
            and (isinstance(sigma, Union[int, float, Decimal, Infinity, Undefined, Variable]))):
        raise ArgTypeError("Must be a numerical value.")
    if not isinstance(resolution, int):
        raise ArgTypeError("Must be an integer.")
    coef = 1 / (sqrt2pi * sigma)
    power = - (x - mean)**2 / (2 * sigma**2)
    return coef * e(power, resolution=resolution)

def laplace(x: Union[int, float, Decimal, Infinity, Undefined, Variable],
            sigma: Union[int, float, Decimal, Infinity, Undefined, Variable], resolution: int = 15):
    """
        Computes the value of the Laplace distribution function at a given point 'x'.

        Args:
            x: The point at which to evaluate the Laplace distribution.
                Must be a numerical value.
            sigma: The scale parameter (standard deviation) of the Laplace distribution.
                Must be a numerical value.
            resolution (int, optional): The resolution of calculations for intermediate steps. Defaults to 15.
                Must be a positive integer.

        Returns:
            Union[float, Decimal]: The value of the Laplace distribution at the given point 'x'.

        Raises:
            ArgTypeError: If 'x' or 'sigma' are not numerical values, or if 'resolution' is not an integer.
    """
    if not ((isinstance(x, Union[int, float, Decimal, Infinity, Undefined, Variable]))
            and (isinstance(sigma, Union[int, float, Decimal, Infinity, Undefined, Variable]))):
        raise ArgTypeError("Must be a numerical value.")
    if not isinstance(resolution, int):
        raise ArgTypeError("Must be an integer.")
    coef = 1 / (sqrt2 * sigma)
    power = - (sqrt2 / sigma) * abs(x)
    return coef * e(power, resolution=resolution)
