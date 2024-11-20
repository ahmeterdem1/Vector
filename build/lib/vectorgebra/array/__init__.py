"""
    General module of all array types, containing Vector and Matrix, and some related
    functions and methods.
"""

from ..math import *
from ..utils import *
from .vmarray import Vector, Matrix, minimum, maximum
from typing import Union
from decimal import Decimal
import random
import threading

def __mul(row: list, m, id: int, target: dict, amount: int):
    """
        Perform multiplication of a row with a matrix.

        Args:
        - row (list): The row of the first matrix to be multiplied.
        - m: The second matrix.
        - id (int): Identifier for the row being processed.
        - target (dict): Dictionary to store the resulting row.
        - amount (int): Number of elements in the row and the number of rows in the second matrix.

        This function multiplies a given row from the first matrix with the second matrix.
        It calculates the dot product of the row with each column of the second matrix and stores the result in the target dictionary.
        The id parameter is used as an identifier for the row being processed.
    """
    length = len(m[0])  # Number of columns for the second matrix
    result = [0 for k in range(length)]

    for k in range(length):
        sum = 0
        for l in range(amount):
            sum += row[l] * m[l][k]
        result[k] = sum

    target[id] = result

def matmul(m1, m2, max: int = 10):
    """
        Perform matrix multiplication between two matrices.

        Args:
        - m1 (Matrix): The first matrix.
        - m2 (Matrix): The second matrix.
        - max (int, optional): The maximum number of threads to use for parallel computation. Defaults to 10.

        Returns:
        - Matrix: The resulting matrix after multiplication.

        Raises:
        - ArgTypeError: If either m1 or m2 is not an instance of the Matrix class.
        - DimensionError: If the dimensions of the matrices are incompatible for matrix multiplication.

        This function performs matrix multiplication between two matrices, m1 and m2.
        It utilizes threading for parallel computation when the number of rows in m1 exceeds a threshold.
        The max parameter determines the maximum number of threads to use concurrently.
        """
    if not (isinstance(m1, Matrix) and isinstance(m2, Matrix)): raise ArgTypeError()
    a, b = [int(k) for k in m1.dimension.split("x")]
    data = {}
    m1values = m1.values

    c, d = [int(k) for k in m2.dimension.split("x")]
    if b != c:
        raise DimensionError(0)
    m2values = m2.values
    if a < 5:
        return m1 * m2

    count = 0
    pool = [0 for k in range(max)]
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

def cumsum(arg: Union[list, tuple, Vector, Matrix]):
    """
        Computes the cumulative sum of numerical elements in the input iterable.

        Args:
            arg: The iterable containing numerical elements for which the cumulative sum is to be computed.

        Returns:
            The cumulative sum of numerical elements in the iterable.

        Raises:
            ArgTypeError: If the argument is not of a valid type or if elements of 'arg' are not numerical.
    """
    if isinstance(arg, Union[list, tuple, Vector]):
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

def mode(arg: Union[tuple, list, Vector, Matrix]):
    """
        Finds the mode in the given dataset.

        Args:
            arg: The dataset to find the mode.

        Returns:
            Any: The mode value found in the dataset.

        Raises:
            ArgTypeError: If the dataset type is not supported.
    """
    if isinstance(arg, Union[tuple, list]):
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

def mean(arg: Union[tuple, list, dict, Vector, Matrix]):
    """
        Calculates the mean (average) of the given dataset.

        Args:
            arg: The dataset for which to calculate the mean.

        Returns:
            float: The mean value of the dataset.

        Raises:
            ArgTypeError: If the dataset type is not supported or invalid.
    """
    if isinstance(arg, Union[list, tuple, Vector]):
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

def median(data: Union[list, tuple, Vector]):
    """
        Calculates the median of the given dataset.

        Args:
            data: The dataset for which to calculate the median.

        Returns:
            float: The median value of the dataset.

        Raises:
            ArgTypeError: If the dataset type is not supported or invalid.
    """
    if isinstance(data, list):
        arg = data.copy()
    elif isinstance(data, tuple):
        arg = list(data)
    elif isinstance(data, Vector):
        arg = data.values.copy()
    else:
        raise ArgTypeError()

    arg.sort()
    n = len(arg)
    point = n // 2
    if point == n / 2:
        return arg[point]
    return (arg[point] + arg[point + 1]) / 2

def expectation(values: Union[list, tuple, Vector], probabilities: Union[list, tuple, Vector], moment: int = 1):
    """
        Calculates the expectation of a random variable.

        Args:
            values: The values of the random variable.
            probabilities: The corresponding probabilities for each value.
            moment (int, optional): The order of the moment. Defaults to 1.

        Returns:
            float: The expectation value of the random variable.

        Raises:
            RangeError: If the moment is negative.
            DimensionError: If the lengths of values and probabilities are different.
            ArgTypeError: If arguments are not one-dimensional iterables.
    """
    if moment < 0:
        raise RangeError()
    if (isinstance(values, Union[list, tuple, Vector])) \
    and (isinstance(probabilities, Union[list, tuple, Vector])):
        if len(values) != len(probabilities):
            raise DimensionError(0)

        sum = 0
        for k in range(len(values)):
            sum += (values[k]**moment) * probabilities[k]
        return sum
    raise ArgTypeError("Arguments must be one dimensional iterables")

def variance(values: Union[list, tuple, Vector], probabilities: Union[list, tuple, Vector]):
    """
        Calculates the variance of a random variable.

        Args:
            values: The values of the random variable.
            probabilities: The corresponding probabilities for each value.

        Returns:
            float: The variance of the random variable.

        Raises:
            DimensionError: If the lengths of values and probabilities are different.
            ArgTypeError: If arguments are not one-dimensional iterables.
    """
    if isinstance(values, Union[list, tuple, Vector]) and isinstance(probabilities, Union[list, tuple, Vector]):
        if len(values) != len(probabilities): raise DimensionError(0)

        sum = 0
        sum2 = 0
        for k in range(len(values)):
            sum += (values[k]**2) * probabilities[k]
            sum2 += values[k] * probabilities[k]
        return sum - sum2**2
    raise ArgTypeError("Arguments must be one dimensional iterables")

def sd(values: Union[list, tuple, Vector], probabilities: Union[list, tuple, Vector]):
    """
        Calculates the standard deviation of a random variable.

        Args:
            values: The values of the random variable.
            probabilities: The corresponding probabilities for each value.

        Returns:
            float: The standard deviation of the random variable.

        Raises:
            DimensionError: If the lengths of values and probabilities are different.
            ArgTypeError: If arguments are not one-dimensional iterables.
    """
    if isinstance(values, Union[list, tuple, Vector]) and isinstance(probabilities, Union[list, tuple, Vector]):
        if len(values) != len(probabilities): raise DimensionError(0)

        sum = 0
        for k in range(len(values)):
            #sum += (values[k]**2) * probabilities[k] - values[k] * probabilities[k]
            sum += values[k] * probabilities[k] * (values[k] - 1)
        return sqrt(sum)
    raise ArgTypeError("Arguments must be one dimensional iterables")

def linear_fit(x: Union[list, tuple, Vector],
               y: Union[list, tuple, Vector],
               rate: Union[int, float, Decimal] = 0.01,
               iterations: int = 15) -> tuple:
    """
        Performs linear regression to fit a line to the given data points. Uses Mean Squared Error.

        Args:
            x: The independent variable data points.
            y: The dependent variable data points corresponding to x.
            rate: The learning rate for gradient descent. Defaults to 0.01.
            iterations (int, optional): The number of iterations for gradient descent. Defaults to 15.

        Returns:
            tuple: A tuple containing the coefficients of the linear model (b0, b1).

        Raises:
            ArgTypeError: If arguments are not one-dimensional iterables or if rate is not a numerical value.
            RangeError: If iterations is less than 1.
            DimensionError: If the lengths of x and y are different.

        Notes:
            - Linear regression is performed using gradient descent to minimize the mean squared error.
            - The algorithm iteratively adjusts the coefficients (b0, b1) to minimize the error.
            - The learning rate controls the step size of each iteration.
            - The number of iterations determines the convergence of the algorithm.
    """
    if not (isinstance(x, Union[list, tuple, Vector]))\
            and (isinstance(y, Union[list, tuple, Vector])):
        raise ArgTypeError("Arguments must be one dimensional iterables")
    if not (isinstance(rate, Union[int, float, Decimal])):
        raise ArgTypeError("Must be a numerical value.")
    if not isinstance(iterations, int): raise ArgTypeError("Must be an integer.")
    if iterations < 1: raise RangeError()
    if len(x) != len(y): raise DimensionError(0)

    N = len(x)
    b0, b1 = 0, 0
    sum1, sum2 = 0, 0
    factor = -2 * rate / N

    for k in range(iterations):
        for i in range(N):
            sum1 += (y[i] - b0 - b1 * x[i])
            sum2 += (y[i] - b0 - b1 * x[i]) * x[i]
        b0 = b0 - sum1 * factor
        b1 = b1 - sum2 * factor
        sum1 = 0
        sum2 = 0
    return b0, b1

def general_fit(x: Union[list, tuple, Vector],
                y: Union[list, tuple, Vector],
                rate: Union[int, float, Decimal] = 0.0000002,
                iterations: int = 15,
                degree: int = 1) -> Vector:
    """
        Performs polynomial regression to fit a polynomial of specified degree to the given data points.
        Uses Mean Squared Error.

        Args:
            x: The independent variable data points.
            y: The dependent variable data points corresponding to x.
            rate: The learning rate for gradient descent. Defaults to 0.0000002.
            iterations (int, optional): The number of iterations for gradient descent. Defaults to 15.
            degree (int, optional): The degree of the polynomial model. Defaults to 1.

        Returns:
            Vector: A vector containing the coefficients of the polynomial model.

        Raises:
            ArgTypeError: If arguments are not one-dimensional iterables or if rate is not a numerical value.
            RangeError: If iterations or degree is less than 1.
            DimensionError: If the lengths of x and y are different.

        Notes:
            - Polynomial regression is performed using gradient descent to minimize the mean squared error.
            - The polynomial model is of the form: b0 + b1*x + b2*x^2 + ... + bn*x^n (where n = degree)
            - The algorithm iteratively adjusts the coefficients to minimize the error.
            - The learning rate controls the step size of each iteration.
            - The number of iterations determines the convergence of the algorithm.
    """
    if (not (isinstance(x, Union[list, tuple, Vector]))
            and (isinstance(y, Union[list, tuple, Vector]))): raise ArgTypeError("Arguments must be one dimensional iterables")
    if not (isinstance(rate, Union[int, float, Decimal])):
        raise ArgTypeError("Must be a numerical value.")
    if not isinstance(iterations, int): raise ArgTypeError("Must be an integer.")
    if iterations < 1 or degree < 1: raise RangeError()
    if len(x) != len(y): raise DimensionError(0)

    # Preprocess
    if not isinstance(x, Vector):
        x = Vector(*[k for k in x])
    if not isinstance(y, Vector):
        y = Vector(*[k for k in y])
    N = len(x)
    factor = -2 * rate / N
    b = Vector(*[1 for k in range(degree + 1)])

    # Work
    for k in range(iterations):
        c = Vector(*[0 for p in range(degree + 1)])
        for i in range(N):
            v = Vector(*[x[i]**p for p in range(degree + 1)])
            c += (y[i] - b.dot(v)) * v
        b = b - c * factor

    return b

def kmeans(dataset: Union[list, tuple, Vector],
           k: int = 2,
           iterations: int = 15,
           a: Union[int, float, Decimal] = 0,
           b: Union[int, float, Decimal] = 10):
    """
        Performs k-means clustering on the given dataset.

        Args:
            dataset: The dataset to be clustered.
            k (int, optional): The number of clusters. Defaults to 2.
            iterations (int, optional): The number of iterations for the k-means algorithm. Defaults to 15.
            a: The lower bound for generating initial cluster centers. Defaults to 0.
            b: The upper bound for generating initial cluster centers. Defaults to 10.

        Returns:
            tuple: A tuple containing the cluster centers and the data assigned to each cluster.

        Raises:
            ArgTypeError: If the dataset is not a one-dimensional iterable or if a or b are not numerical values.
            DimensionError: If the dataset is empty.
            RangeError: If k or iterations are less than 1.
            ArgTypeError: If the elements in the dataset are not of the same type or not of type Vector.

        Notes:
            - The k-means algorithm partitions the dataset into k clusters.
            - It iteratively assigns data points to the nearest cluster center and updates the center.
            - The algorithm terminates when either the maximum number of iterations is reached or the cluster centers converge.
            - The initial cluster centers are randomly generated within the range [a, b].
            - The output is a tuple where the first element is a list of cluster centers (Vectors) and the second element
              is a list of lists containing the data assigned to each cluster.
    """
    if not (isinstance(dataset, Union[list, tuple, Vector])): raise ArgTypeError()
    if len(dataset) == 0: raise DimensionError(1)
    if k < 1: raise RangeError()
    if iterations < 1: raise RangeError()
    if not ((isinstance(a, Union[int, float, Decimal]))
            and (isinstance(b, Union[int, float, Decimal]))): raise ArgTypeError("Must be a numerical value.")

    # This is a strange logical operation.
    # I don't want to spend time here optimizing this.
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

def unique(data: Union[list, tuple, Vector, Matrix]):
    """
        Counts the occurrences of unique elements in the given data.

        Args:
            data: The data to find unique elements and their counts.

        Returns:
            dict: A dictionary containing unique elements as keys and their counts as values.

        Raises:
            ArgTypeError: If the data is not an iterable (list, tuple, Vector, or Matrix).

        Notes:
            - For one-dimensional data (list, tuple, Vector), this function counts the occurrences of each unique element.
            - For a Matrix, the function reshapes it into a one-dimensional structure before counting unique elements.
            - Returns a dictionary where keys represent unique elements and values represent their counts.
        """
    if isinstance(data, Union[list, tuple, Vector]):
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

def isAllUnique(data: Union[list, tuple, Vector, Matrix]):
    """
        Checks if all elements in the given data are unique.

        Args:
            data: The data to check for uniqueness.

        Returns:
            bool: True if all elements are unique, False otherwise.

        Raises:
            ArgTypeError: If the data is not an iterable (list, tuple, Vector, or Matrix).
    """
    if isinstance(data, Union[list, tuple]):
        return len(data) == len(set(data))
    if isinstance(data, Vector):
        return len(data) == len(set(data.values))
    if isinstance(data, Matrix):
        val = len(data) * len(data[0])
        v = data.reshape(val)
        return val == len(set(v.values))
    raise ArgTypeError("Must be an iterable.")

def __permutate(sample: list, count: int, length: int, target: list):
    """
        Recursive helper function to generate permutations of a given sample.

        Args:
            sample (list): The sample elements to permute.
            count (int): The current count of elements permuted.
            length (int): The total length of the sample.
            target (list): The list to store generated permutations.

        Returns:
            None: The permutations are stored in the target list.

        Notes:
            This function is not intended for direct use. It's called by the permutate function to generate permutations recursively.

    """
    if count == length:
        target.append(sample.copy())
    for k in range(count, length):
        sample[k], sample[count] = sample[count], sample[k]
        __permutate(sample, count + 1, length, target)
        sample[count], sample[k] = sample[k], sample[count]

def permutate(sample: Union[list, tuple, Vector, Matrix]):
    """
        Generates all possible permutations of the elements in the given sample.

        Args:
            sample: The sample elements to permute.

        Returns:
            list: A list containing all possible permutations of the sample elements.

        Raises:
            ArgTypeError: If the sample is not an iterable (list, tuple, Vector, or Matrix).
    """
    if isinstance(sample, Union[list, tuple]):
        arg = list(set(sample))
    elif isinstance(sample, Vector):
        arg = list(set(sample.values))
    elif isinstance(sample, Matrix):
        arg = sample.values
    else: raise ArgTypeError("Must be an iterable.")
    target = []
    __permutate(arg, 0, len(arg), target)
    return target

