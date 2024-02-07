from .infinity import *
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
log2E = 1.4426950408889392
log2_10 = 3.3219280948873626
sqrtpi = 1.77245385091
sqrt2 = 1.41421356237
sqrt2pi = 2.50662827463

__results = {}

def Range(low, high, step=1):
    """
        Generator function that yields a range of numerical values from 'low' to 'high' (inclusive) with a specified step.

        Args:
            low (int, float, Decimal, Infinity, Undefined): The starting value of the range.
            high (int, float, Decimal, Infinity, Undefined): The ending value of the range.
            step (int, float, Decimal, Infinity, Undefined, optional): The step size between consecutive values. Default is 1.

        Yields:
            int, float, Decimal, Infinity, Undefined: The next value in the range sequence.

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

    while (high < low) ^ (step > 0) and not high == low:
        yield low
        low += step

def abs(arg):
    """
        Returns the absolute value of a numerical argument.

        Args:
            arg (int, float, Decimal, Infinity, Undefined): The numerical value whose absolute value is to be computed.

        Returns:
            int, float, Decimal, Infinity, Undefined: The absolute value of the argument.

        Raises:
            ArgTypeError: If the argument is not a numerical value.
    """
    if not isinstance(arg, Union[int, float, Decimal, Infinity, Undefined]):
        raise ArgTypeError("Must be a numerical value.")

    return arg if (arg >= 0) else -arg

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

def e(exponent, resolution=15):
    """
        Computes the approximation of the mathematical constant 'e' raised to the power of the given exponent.

        Args:
            exponent (int, float, Decimal, Infinity, Undefined): The numerical exponent for 'e'.
            resolution (int, optional): The resolution for the approximation. Defaults to 15.

        Returns:
            float: The approximation of 'e' raised to the power of the given exponent.

        Raises:
            ArgTypeError: If 'exponent' is not a numerical value.
            RangeError: If 'resolution' is not a positive integer.
    """
    if not (isinstance(resolution, int) and resolution >= 1): raise RangeError("Resolution must be a positive integer")
    if not isinstance(exponent, Union[int, float, Decimal, Infinity, Undefined]):
        raise ArgTypeError("Must be a numerical value.")

    sum = 1
    for k in range(resolution, 0, -1):
        sum += __cumdiv(exponent, k)
    return sum

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

    radian = (2 * PI * (angle % 360 / 360)) % (2 * PI)
    result = 0
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

    radian = (2 * PI * (angle % 360 / 360)) % (2 * PI)
    result = 1

    if resolution % 2:
        resolution += 1

    for k in range(resolution, 0, -2):
        result = result + __cumdiv(radian, k) * pow(-1, k / 2)
    return result

def tan(angle, resolution=16):
    """
        Computes the tangent of the given angle.

        Args:
            angle (int, float, Decimal, Infinity, Undefined): The angle in degrees.
            resolution (int, optional): The resolution for the approximation. Defaults to 16.

        Returns:
            float or Infinity: The tangent value of the given angle.

        Raises:
            ArgTypeError: If 'angle' is not a numerical value.
            RangeError: If 'resolution' is not a positive integer.
    """
    if not (isinstance(resolution, int) and resolution >= 1): raise RangeError("Resolution must be a positive integer")
    try:
        return sin(angle, resolution - 1) / cos(angle, resolution)
        # Because of the error amount, probably cos will never be zero.
    except ZeroDivisionError:
        # sin * cos is positive in this area
        if 90 >= (angle % 360) >= 0 or 270 >= (angle % 360) >= 180: return Infinity()
        return Infinity(False)

def cot(angle, resolution: int = 16):
    """
        Computes the cotangent of the given angle.

        Args:
            angle (int, float, Decimal, Infinity, Undefined): The angle in degrees.
            resolution (int, optional): The resolution for the approximation. Defaults to 16.

        Returns:
            float or Infinity: The tangent value of the given angle.

        Raises:
            ArgTypeError: If 'angle' is not a numerical value.
            RangeError: If 'resolution' is not a positive integer.
    """
    try:
        return 1 / tan(angle, resolution)
    except ZeroDivisionError:  # Probably will never happen
        return None

def sinh(x, resolution: int = 15):
    """
        Computes the hyperbolic sine of the given number.

        Args:
            x (int, float, Decimal, Infinity, Undefined): The number.
            resolution (int, optional): The resolution for the approximation. Defaults to 15.

        Returns:
            float: The hyperbolic sine value of the given number.

        Raises:
            RangeError: If 'resolution' is not a positive integer.
            ArgTypeError: If 'x' is not a numerical value.
        """
    return (e(x, resolution) - e(-x, resolution)) / 2

def cosh(x, resolution: int = 15):
    """
        Computes the hyperbolic cosine of the given number.

        Args:
            x (int, float, Decimal, Infinity, Undefined): The number.
            resolution (int, optional): The resolution for the approximation. Defaults to 15.

        Returns:
            float: The hyperbolic sine value of the given number.

        Raises:
            RangeError: If 'resolution' is not a positive integer.
            ArgTypeError: If 'x' is not a numerical value.
        """
    return (e(x, resolution) + e(-x, resolution)) / 2

def tanh(x, resolution: int = 15):
    """
        Computes the hyperbolic tangent of the given number.

        Args:
            x (int, float, Decimal, Infinity, Undefined): The number.
            resolution (int, optional): The resolution for the approximation. Defaults to 15.

        Returns:
            float: The hyperbolic sine value of the given number.

        Raises:
            RangeError: If 'resolution' is not a positive integer.
            ArgTypeError: If 'x' is not a numerical value.
        """
    try:
        return sinh(x, resolution) / cosh(x, resolution)
        # Indeed cosh is non-negative
    except ZeroDivisionError:
        return None

def coth(x, resolution: int = 15):
    """
        Computes the hyperbolic cotangent of the given number.

        Args:
            x (int, float, Decimal, Infinity, Undefined): The number.
            resolution (int, optional): The resolution for the approximation. Defaults to 15.

        Returns:
            float: The hyperbolic sine value of the given number.

        Raises:
            RangeError: If 'resolution' is not a positive integer.
            ArgTypeError: If 'x' is not a numerical value.
        """
    try:
        return cosh(x, resolution) / sinh(x, resolution)
    except ZeroDivisionError:
        if x >= 0: return Infinity()
        return Infinity(False)

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

def arccos(x: int or float, resolution: int = 20):
    """
        Computes the arc cosine of the given number in degrees.

        Args:
            x (int, float, Decimal, Infinity, Undefined): The number.
            resolution (int, optional): The resolution for the approximation. Defaults to 20.

        Returns:
            float: The arc cosine value of the given number in degrees.

        Raises:
            ArgTypeError: If 'x' is not a numerical value.
            RangeError: If 'x' is not in the range [-1, 1], or if 'resolution' is not a positive integer.
    """
    if not (-1 <= x <= 1): raise RangeError()
    if resolution < 1: raise RangeError("Resolution must be a positive integer")

    return 90 - arcsin(x, resolution)

def log2(x, resolution: int = 15):
    """
        Computes the base-2 logarithm of the given number.

        Args:
            x (int, float, Decimal, Infinity, Undefined): The number.
            resolution (int, optional): The resolution for the approximation. Defaults to 15.

        Returns:
            float: The base-2 logarithm of the given number.

        Raises:
            ArgTypeError: If 'x' is not a numerical value.
            RangeError: If 'x' is less than or equal to 0, or if 'resolution' is not a positive integer.
    """
    if not isinstance(x, Union[int, float, Decimal, Infinity, Undefined]):
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
    """
        Computes the base-e logarithm of the given number.

        Args:
            x (int, float, Decimal, Infinity, Undefined): The number.
            resolution (int, optional): The resolution for the approximation. Defaults to 15.

        Returns:
            float: The base-e logarithm of the given number.

        Raises:
            ArgTypeError: If 'x' is not a numerical value.
            RangeError: If 'x' is less than or equal to 0, or if 'resolution' is not a positive integer.
        """
    return log2(x, resolution) / log2E

def log10(x, resolution: int = 15):
    """
        Computes the base-10 logarithm of the given number.

        Args:
            x (int, float, Decimal, Infinity, Undefined): The number.
            resolution (int, optional): The resolution for the approximation. Defaults to 15.

        Returns:
            float: The base-10 logarithm of the given number.

        Raises:
            ArgTypeError: If 'x' is not a numerical value.
            RangeError: If 'x' is less than or equal to 0, or if 'resolution' is not a positive integer.
        """
    return log2(x, resolution) / log2_10

def log(x, base=2, resolution: int = 15):
    """
        Computes the logarithm of a number 'x' with respect to the specified base.

        Args:
            x (int, float, Decimal, Infinity, Undefined): The number.
            base (int, float, Decimal): The base of the logarithm. Defaults to 2.
            resolution (int, optional): The resolution for the approximation. Defaults to 15.

        Returns:
            float: The logarithm of 'x' with respect to the specified base.

        Raises:
            ArgTypeError: If 'base' is not a numerical value.
            RangeError: If 'base' is less than or equal to 0 or equal to 1.
    """
    if not (isinstance(base, Union[int, float, Decimal])):
        raise ArgTypeError("Must be a numerical value.")
    if base <= 0 or base == 1: raise RangeError()
    return log2(x, resolution) / log2(base, resolution)

def __find(f, low, high, search_step, res=0.0001):
    """
        Find a zero of a function within a specified range using recursive search.

        Args:
        - f (Callable): The function for which a zero are to be found.
        - low (Union[int, float, Decimal]): The lower bound of the search range.
        - high (Union[int, float, Decimal]): The upper bound of the search range.
        - search_step (Union[int, float, Decimal]): The step size for searching.
        - res (Union[int, float, Decimal], optional): The resolution for finding zeros. Defaults to Decimal(0.0001).

        Returns:
        - Union[int, float, Decimal]: The approximate zero found within the specified range, or None if not found.

        Raises:
        - ArgTypeError: If the arguments are not numerical values.

        Notes:
            This function performs a recursive search within the given range to find a zero of the provided function.
    """
    if not ((isinstance(low, Union[int, float, Decimal]))
            and (isinstance(high, Union[int, float, Decimal]))
            and (isinstance(search_step, Union[int, float, Decimal]))
            and (isinstance(res, Union[int, float, Decimal]))):
        raise ArgTypeError("Must be a numerical value.")
    if not isinstance(f, Callable):
        raise ArgTypeError(f"f must be a callable.")

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

def solve(f: Callable, low=-50, high=50, search_step=0.1, res=0.0001) -> List[Union[int, float, Decimal]]:
    """
        Find approximate zeros of a given function within a specified range.

        Args:
        - f (Callable): The function for which zeros are to be found.
        - low (Union[int, float, Decimal], optional): The lower bound of the range. Defaults to -50.
        - high (Union[int, float, Decimal], optional): The upper bound of the range. Defaults to 50.
        - search_step (Union[int, float, Decimal], optional): The step size for searching. Defaults to 0.1.
        - res (Union[int, float, Decimal], optional): The resolution for finding zeros. Defaults to Decimal(0.0001).

        Returns:
        - List[Union[int, float, Decimal]]: A list of approximate zeros found within the specified range.

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
    """
        Computes the derivative of a function 'f' at a point 'x' using the finite difference method.

        Args:
            f (callable): The function for which the derivative is to be computed.
            x (int, float, Decimal): The point at which the derivative is evaluated.
            h (int, float, Decimal, optional): The step size for the finite difference method.
                Defaults to 0.0000000001.

        Returns:
            float: The derivative of 'f' at point 'x' using the finite difference method.

        Raises:
            ArgTypeError: If 'x' or 'h' is not a numerical value, or if 'f' is not callable.
    """
    if not ((isinstance(x, Union[int, float, Decimal]))
            and (isinstance(h, Union[int, float, Decimal]))):
        raise ArgTypeError("Must be a numerical value.")
    if not isinstance(f, Callable): raise ArgTypeError("f must be a callable.")

    return (f(x + h) - f(x)) / h

def integrate(f, a, b, delta=0.01):
    """
        Numerically integrates a function 'f' over the interval [a, b] using the rectangle method.

        Args:
            f (callable): The function to be integrated.
            a (int, float, Decimal): The lower limit of integration.
            b (int, float, Decimal): The upper limit of integration.
            delta (int, float, Decimal, optional): The width of each rectangle used in the integration.
                Defaults to 0.01.

        Returns:
            float: The approximate value of the integral of 'f' over the interval [a, b].

        Raises:
            ArgTypeError: If 'a', 'b', or 'delta' is not a numerical value, or if 'f' is not callable.
    """
    if not ((isinstance(a, Union[int, float, Decimal]))
            and (isinstance(b, Union[int, float, Decimal]))
            and (isinstance(delta, Union[int, float, Decimal]))):
        raise ArgTypeError("Must be a numerical value.")

    if not isinstance(f, Callable): raise ArgTypeError("f must be a callable.")

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

def findsol(f, x=0, resolution = 15):
    """
        Find a numerical solution to the equation f(x) = 0 using Newton's method.

        Args:
        - f: The function for which the root is sought. It must be a callable function.
        - x (int, float, Decimal, optional): The initial guess for the root. Default is 0.
        - resolution (int, optional): The number of iterations to refine the solution. Default is 15.

        Returns:
        - x: The numerical approximation of the root.

        Notes:
            This function applies Newton's method to iteratively find a solution to the equation f(x) = 0.
            It uses the derivative of the function f(x) to approximate the root.
            The initial guess for the root is provided by the x parameter, and the resolution parameter determines
            the number of iterations used to refine the solution.
    """
    if not isinstance(f, Callable): raise ArgTypeError("f must be a callable")
    if resolution < 1 or not isinstance(resolution, int): raise RangeError("Resolution must be a positive integer")

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
    """
        Compute the sigmoid function for the given input value(s).

        Args:
        - x (int, float, Decimal, Infinity, Undefined): The input value(s) for which the sigmoid function is computed.
        - a (int, float, Decimal, Infinity, Undefined, optional): The scaling factor. Defaults to 1.

        Returns:
        - float: The value of the sigmoid function evaluated at x.

        Raises:
        - ArgTypeError: If the input arguments are not numerical values.
    """
    if not((isinstance(a, Union[int, float, Decimal, Infinity, Undefined]))
           and (isinstance(x, Union[int, float, Decimal, Infinity, Undefined]))):
        raise ArgTypeError("Must be a numerical value.")
    return 1 / (1 + e(-a*x))

def ReLU(x, leak=0, cutoff=0):
    """
        Implements the Rectified Linear Unit (ReLU) activation function.

        Args:
            x (Union[int, float, Decimal, Infinity, Undefined]): The input value to apply the ReLU function to.
            leak (Union[int, float, Decimal, Infinity, Undefined], optional): The slope for negative inputs (default is 0).
            cutoff (Union[int, float, Decimal, Infinity, Undefined], optional): The threshold value where ReLU starts to have an effect (default is 0).

        Returns:
            Union[int, float, Decimal, Infinity, Undefined]: The result of applying the ReLU function to the input value.

        Raises:
            ArgTypeError: If any of the input arguments is not a numerical value.
    """
    if not ((isinstance(x, Union[int, float, Decimal, Infinity, Undefined]))
            and (isinstance(leak, Union[int, float, Decimal, Infinity, Undefined]))
            and (isinstance(cutoff, Union[int, float, Decimal, Infinity, Undefined]))):
        raise ArgTypeError("Must be a numerical value.")

    if x >= cutoff:
        return x
    elif x < 0:
        return leak * x
    else:
        return cutoff

def deriv_relu(x, leak=0, cutoff=0):
    """
        Computes the derivative of the Rectified Linear Unit (ReLU) activation function.

        Args:
            x (Union[int, float, Decimal, Infinity, Undefined]): The input value to compute the derivative for.
            leak (Union[int, float, Decimal, Infinity, Undefined], optional): The slope for negative inputs (default is 0).
            cutoff (Union[int, float, Decimal, Infinity, Undefined], optional): The threshold value where ReLU starts to have an effect (default is 0).

        Returns:
            Union[int, float, Decimal, Infinity, Undefined]: The derivative of the ReLU function with respect to the input value.

        Raises:
            ArgTypeError: If any of the input arguments is not a numerical value.
    """
    if not ((isinstance(x, Union[int, float, Decimal, Infinity, Undefined]))
            and (isinstance(leak, Union[int, float, Decimal, Infinity, Undefined]))
            and (isinstance(cutoff, Union[int, float, Decimal, Infinity, Undefined]))):
        raise ArgTypeError("Must be a numerical value.")
    return 1 if x >= cutoff else 0 if x >= 0 else leak

def Sum(f, a, b, step=0.01, control: bool = False, limit=0.000001):
    """
        Computes the sum of a function over a given interval.

        Args:
            f (Callable): The function to be summed.
            a (Union[int, float, Decimal, Infinity, Undefined]): The lower bound of the interval.
            b (Union[int, float, Decimal, Infinity, Undefined]): The upper bound of the interval.
            step (Union[int, float, Decimal, Infinity, Undefined], optional): The step size for sampling points within the interval (default is 0.01).
            control (bool, optional): A flag indicating whether to use control over derivative values to stop the summing process (default is False).
            limit (Union[int, float, Decimal, Infinity, Undefined], optional): The limit value to control the derivative when 'control' is True (default is 0.000001).

        Returns:
            Union[int, float, Decimal, Infinity, Undefined]: The sum of the function over the interval [a, b).

        Raises:
            ArgTypeError: If any of the input arguments is not a numerical value.
            RangeError: If the upper bound 'b' is less than the lower bound 'a'.
    """

    # Yes, with infinities this can blow up. It is the users problem to put infinities in.
    if not ((isinstance(a, Union[int, float, Decimal, Infinity, Undefined]))
            and (isinstance(b, Union[int, float, Decimal, Infinity, Undefined]))
            and (isinstance(step, Union[int, float, Decimal, Infinity, Undefined]))
            and (isinstance(limit, Union[int, float, Decimal, Infinity, Undefined]))):
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

def factorial(x=0):
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
    if not isinstance(x, int): raise ArgTypeError("Must be an integer.")
    if x < 0: raise RangeError()
    if x <= 1: return 1
    mul = 1
    for k in range(2, x):
        mul *= k
    return mul * x

def permutation(x=1, y=1):
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

def combination(x=0, y=0):
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

def multinomial(n=0, *args):
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
    while n != 1:
        result *= n
        for k in range(len(c)):
            if not c[k]: continue
            result /= c[k]
            c[k] -= 1
        n -= 1
    return result

def binomial(n, k, p):
    """
        Computes the probability of obtaining 'k' successes in 'n' independent Bernoulli trials,
        each with a success probability of 'p'.

        Args:
            n (int): The total number of trials.
            k (int): The number of successes.
            p (Union[int, float, Decimal]): The probability of success in each trial. Must be in the range [0, 1].

        Returns:
            float: The probability of obtaining 'k' successes in 'n' trials with success probability 'p'.

        Raises:
            ArgTypeError: If 'n' or 'k' is not an integer, or if 'p' is not a numerical value.
            RangeError: If 'p' is outside the range [0, 1].
    """
    if not (isinstance(n, int) and isinstance(k, int)): raise ArgTypeError("Must be an integer.")
    if not (isinstance(p, Union[int, float, Decimal])):
        raise ArgTypeError("Must be a numerical value.")
    if p < 0 or p > 1: raise RangeError("Probability cannot be negative or bigger than 1")
    return combination(n, k) * pow(p, k) * pow(1 - p, k)

def geometrical(n, p):
    """
        Computes the probability of the nth trial being the first success in a sequence of independent
        Bernoulli trials, each with a success probability of 'p'.

        Args:
            n (int): The trial number for which the probability is calculated. Must be a non-negative integer.
            p (Union[int, float, Decimal]): The probability of success in each trial. Must be in the range [0, 1].

        Returns:
            Union[float, Undefined]: The probability of the nth trial being the first success, or Undefined if 'n' is 0.

        Raises:
            ArgTypeError: If 'n' is not an integer, or if 'p' is not a numerical value.
            RangeError: If 'p' is outside the range [0, 1], or if 'n' is negative.
    """
    if not isinstance(n, int): raise ArgTypeError("Must be an integer.")
    if not (isinstance(p, Union[int, float, Decimal])):
        raise ArgTypeError("Must be a numerical value.")
    if p < 0 or p > 1: raise RangeError("Probability cannot be negative or bigger than 1")
    if n < 0: raise RangeError("Trial number cannot be negative")
    if n == 0: return Undefined()
    return p * pow(1 - p, n - 1)

def poisson(k, l):
    """
        Computes the probability of observing 'k' events in a fixed interval of time or space,
        given that the average rate of occurrence of the event is 'l' events per unit interval.

        Args:
            k (Union[int, float, Decimal]): The number of events to be observed. Must be a non-negative numerical value.
            l (Union[int, float, Decimal]): The average rate of occurrence of the event per unit interval.
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
    if not ((isinstance(k, Union[int, float, Decimal]))
            and (isinstance(l, Union[int, float, Decimal]))):
        raise ArgTypeError("Must be a numerical value.")
    if l < 0 or k < 0: raise RangeError()
    return pow(l, k) * e(-l) / factorial(k)

def normal(x, resolution: int = 15):
    """
        Computes the value of the standard normal probability density function (PDF) at a given point 'x'.

        Args:
            x (Union[int, float, Decimal, Infinity]): The point at which to evaluate the standard normal PDF.
                Must be a numerical value.
            resolution (int, optional): The resolution of calculations for intermediate steps. Defaults to 15.
                Must be a positive integer.

        Returns:
            Union[float, Decimal]: The value of the standard normal PDF at the given point 'x'.

        Raises:
            ArgTypeError: If 'x' is not a numerical value, or if 'resolution' is not an integer.
    """
    if not (isinstance(x, Union[int, float, Decimal, Infinity])): raise ArgTypeError("Must be a numerical value.")
    if not isinstance(resolution, int): raise ArgTypeError("Must be an integer.")
    return e(-(x**2) / 2, resolution=resolution) / sqrt2pi

def gaussian(x, mean, sigma, resolution: int = 15):
    """
        Computes the value of the Gaussian (normal) distribution function at a given point 'x'.

        Args:
            x (Union[int, float, Decimal, Infinity]): The point at which to evaluate the Gaussian distribution.
                Must be a numerical value.
            mean (Union[int, float, Decimal, Infinity]): The mean of the Gaussian distribution.
                Must be a numerical value.
            sigma (Union[int, float, Decimal, Infinity]): The standard deviation of the Gaussian distribution.
                Must be a numerical value.
            resolution (int, optional): The resolution of calculations for intermediate steps. Defaults to 15.
                Must be a positive integer.

        Returns:
            Union[float, Decimal]: The value of the Gaussian distribution at the given point 'x'.

        Raises:
            ArgTypeError: If 'x', 'mean', or 'sigma' are not numerical values, or if 'resolution' is not an integer.
    """
    if not ((isinstance(x, Union[int, float, Decimal, Infinity]))
            and (isinstance(mean, Union[int, float, Decimal, Infinity]))
            and (isinstance(sigma, Union[int, float, Decimal, Infinity]))):
        raise ArgTypeError("Must be a numerical value.")
    if not isinstance(resolution, int): raise ArgTypeError("Must be an integer.")
    coef = 1 / (sqrt2pi * sigma)
    power = - pow(x - mean, 2) / (2 * pow(sigma, 2))
    return coef * e(power, resolution=resolution)

def laplace(x, sigma, resolution: int = 15):
    """
        Computes the value of the Laplace distribution function at a given point 'x'.

        Args:
            x (Union[int, float, Decimal, Infinity]): The point at which to evaluate the Laplace distribution.
                Must be a numerical value.
            sigma (Union[int, float, Decimal, Infinity]): The scale parameter (standard deviation) of the Laplace distribution.
                Must be a numerical value.
            resolution (int, optional): The resolution of calculations for intermediate steps. Defaults to 15.
                Must be a positive integer.

        Returns:
            Union[float, Decimal]: The value of the Laplace distribution at the given point 'x'.

        Raises:
            ArgTypeError: If 'x' or 'sigma' are not numerical values, or if 'resolution' is not an integer.
    """
    if not ((isinstance(x, Union[int, float, Decimal, Infinity]))
            and (isinstance(sigma, Union[int, float, Decimal, Infinity]))):
        raise ArgTypeError("Must be a numerical value.")
    if not isinstance(resolution, int): raise ArgTypeError("Must be an integer.")
    coef = 1 / (sqrt2 * sigma)
    power = - (sqrt2 / sigma) * abs(x)
    return coef * e(power, resolution=resolution)
