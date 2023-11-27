# Vectorgebra

A numerical methods tool for python, in python.

There are 5 main subclasses; Vector, Matrix, complex, Infinity, Undefined.
And also there are functions, constants and exception classes. 
Each section is explained below.

## Project details

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

_pip install vectorgebra_

https://pypi.org/project/vectorgebra/

### Update notes on 2.3.0

New constants added to prevent recalculation of the same values.

Outer product added to Vectors.

"decimal" parameter added to pow() method on Matrix.

Neumann inverse method added to Matrix. This method will only
work for right conditioned matrices.

Normalization, Hermitian conjugation, norm, least squares methods added to Matrix class.

Undefined support increased. Also, new statistical functions added on distributions 
(gaussian, etc.).

## Vectorgebra.Vector

Includes basic and some sophisticated operations on vectors.

Addition, multiplication subtraction, division and operations alike are 
implemented as overloads. Comparison operators compare the length of the
vectors. Only exception is _==_ which returns True if and only if all the
terms of the vectors are equal.

Methods are listed below.

### _Vectorgebra.Vector_.dot(v)

Returns the dot product of self with v.

### _Vectorgebra.Vector_.append(v)

Appends the argument to the vector. Returns the new vector. Argument
can be int, float or Vector.

### _Vectorgebra.Vector_.copy()

Returns the copy of the vector.

### _Vectorgebra.Vector_.pop(ord)

Functions exactly the same as .pop() for the list class. If left blank,
pops the last element and returns it. If specified, pops the intended
element and returns it.

### _Vectorgebra.Vector_.length()

Returns the length of self.

### _Vectorgebra.Vector_.proj(v)

Projects self onto v and returns the resulting vector. Due to the 
division included, may result in inaccurate values that should have
been zero. However, these values are very close to zero and are of
magnitude 10<sup>-17</sup>. This situation is important for the method
Vector.spanify().

### _Vectorgebra.Vector_.unit()

Returns the unit vector.

### Vectorgebra.Vector.spanify(*args)

Applies Gram-Schmidt process to the given list of vectors. Returns the
list of resulting vectors. May have inaccuracies explained for Vector.dot()
method.

### Vectorgebra.Vector.does_span(*args)

Returns True if the given list of vectors span the R<sup>n</sup> space
where n is the number of vectors. Returns False otherwise. Eliminates
the possible error from Vector.spanify() method. Therefore, this method
will work just fine regardless the errors from divisions.

### Vectorgebra.Vector.randVint(dim , a, b, decimal=True)

Returns _dim_ dimensional vector which has its elements randomly selected
as integers within the interval (a, b). If decimal is true, generated contents
are decimal objects.

### Vectorgebra.Vector.randVfloat(dim, a, b, decimal=True)

Returns _dim_ dimensional vector which has its elements randomly selected
as floats within the interval (a, b). If decimal is true, generated contents
are decimal objects.

### Vectorgebra.Vector.randVbool(dim, decimal=True)

Returns _dim_ dimensional vector which has its elements randomly selected
as booleans. If decimal is true, generated contents are decimal objects.

### Vectorgebra.Vector.determinant(*args)

Returns the determinant of the matrix which has rows given as vectors in
*args. This method is not intended for casual use. It is a background
tool for cross product and the determinant method for Matrix class.

### Vectorgebra.Vector.cross(*args)

Returns the cross product of the vectors given in *args.

### Vectorgebra.Vector.outer(v, w)

Returns the outer product of Vectors v and w. Return type is therefore
a Matrix.

### _Vectorgebra.Vector_.cumsum()

Returns the cumulative sum.

### Vectorgebra.Vector.zero(dim)

Returns _dim_ dimensional zero vector.

## Vectorgebra.Vector.one(dim)

Returns _dim_ dimensional all ones vector.

### _Vectorgebra.Vector_.reshape(a, b)

Returns the reshaped matrix.


#### Footnotes

Rotation is not implemented, because only reasonable way to implement
it would be for just 2D and 3D vectors, which is not satisfying. 


<hr>

## Vectorgebra.Matrix

Includes basic operations for matrices.

Basic operations like addition, multiplication subtraction, division
are implemented as overloads. Only comparison operator implemented is
_==_ which returns true if and only if all the elements of the matrices
are equal.

"pow" method accepts the parameter "decimal".

Methods are listed below.

### _Vectorgebra.Matrix_.determinant(choice="echelon")

Returns the determinant of the matrix m. Two choices are available currently;
echelon, analytic. Default is echelon.

### _Vectorgebra.Matrix_.append(arg)

Appends arg as a new row to self. Only accepts vectors as arguments.

### _Vectorgebra.Matrix_.copy()

Returns the copy of the matrix.

### _Vectorgebra.Matrix_.pop(ord)

Functions exactly the same as .pop() in list class. If left blank, pops
the last row and returns it. If specified, pops the row in the given 
order and returns it.

### _Vectorgebra.Matrix_.transpose()

Returns the transpose matrix of self

### _Vectorgebra.Matrix_.conjugate()

Returns the complex conjugate of the self.

### _Vectorgebra.Matrix_.normalize()

Divides self with its determinant.

### _Vectorgebra.Matrix_.hconj()

Returns the Hermitian conjugate of self.

### _Vectorgebra.Matrix_.norm(resolution: int = 15, decimal: bool = True)

Returns the Frobenius norm of self. Utilizes the eigenvalue function.
Parameters are directly passed to eigenvalue function.

### _Vectorgebra.Matrix_.inverse(method="iteraitve", resolution=10, lowlimit=0.0000000001, highlimit=100000)

Returns the inverse matrix of self. Returns None if not invertible.
method can ben "analytic", "gauss", "neumann" or "iterative". Default is iterative
which uses Newton's method for matrix inversion. Resolution is the number
of iterations. lowlimit and highlimit are only for gauss method. They control
the "resolution" of multiplication and divisions. See the source code for a
better inside look.

Neumann will only work for the right conditioned matrices (see [here](https://en.wikipedia.org/wiki/Matrix_norm)). Neumann only uses
resolution parameter.

### Vectorgebra.Matrix.identity(dim, decimal=True)

Returns _dimxdim_ dimensional identity matrix. If decimal is true, generated 
contents are decimal objects.

### Vectorgebra.Matrix.zero(dim, decimal=True)

Returns _dimxdim_ dimensional all 0 matrix. If decimal is true, generated 
contents are decimal objects.

### Vectorgebra.Matrix.one(dim, decimal=True)

Returns _dimxdim_ dimensional all 1 matrix. If decimal is true, generated 
contents are decimal objects.

### Vectorgebra.Matrix.randMint(m, n, a, b, decimal=True)

Returns _mxn_ matrix of random integers selected from the interval (a, b).
If decimal is true, generated contents are decimal objects.

### Vectorgebra.Matrix.randMfloat(m, n, a, b, decimal=True)

Returns _mxn_ matrix of random floats selected from the interval (a, b).
If decimal is true, generated contents are decimal objects.

### Vectorgebra.Matrix.randMbool(m, n, decimal=True)

Returns _mxn_ matrix of random booleans. If decimal is true, generated 
contents are decimal objects.

### _Vectorgebra.Matrix_.echelon()

Returns reduced row echelon form of self. Also does reorganization on
rows and multiplies one of them by -1 every 2 reorganization. This is
for the determinant to remain unchanged.

### Vectorgebra.Matrix.cramer(a, number)

Applies Cramers rule to the equation system represented by the matrix _a_.
_number_ indicates which variable to calculate.

### _Vectorgebra.Matrix_.cumsum()

Returns the cumulative sum.

### _Vectorgebra.Matrix_.reshape(*args)

Returns the reshaped matrix/vector. If the return is a matrix,
makes a call to the vectors reshape.

### _Vectorgebra.Matrix_.eigenvalue(resolution: int)

Calculates the eigenvalues of self and returns a list of them.
This function cannot calculate complex valued eigenvalues. So
if there are any, there will be incorrect numbers in the returned
list instead of the complex ones.

The underlying algorithm is QR decomposition and iteration. Resolution
is the number of iterations. Default is 10.

### _Vectorgebra.Matrix_.qr()

Applies QR decomposition to self and returns the tuple (Q, R). The algorithm
just uses .spanify() from Vector class. If the columns of self do not consist
of independent vectors, returns matrices of zeroes for both Q and R. This is to
prevent type errors that may have otherwise risen from written code.

### _Vectorgebra.Matrix_.cholesky()

Applies Cholesky decomposition to self, and returns the L matrix. Applies algorithm
is textbook Cholesky–Banachiewicz algorithm.

### _Vectorgebra.Matrix_.trace()

Returns the trace of self.

### _Vectorgebra.Matrix_.diagonals()

Returns the list of diagonals.

### _Vectorgebra.Matrix_.diagonal_mul()

Returns the multiplication of diagonals.

### _Vectorgebra.Matrix_.gauss_seidel(b: Vector, initial=None, resolution=15, decimal=True)

Applies Gauss-Seidel method to the equation self * x = b. Returns the resultant x Vector.
If "initial" left unchanged, code creates an initial guess by default. This may not converge
though. "resolution" is the number of iterations. "decimal" argument is also passed to called
functions inside this method. Same for resolution.

### _Vectorgebra.Matrix_.least_squares(b, *args)

Accepts every argument that .inverse() accepts. Solves the equation _self * x = b_ for x.
Every argument except b is passed to .inverse(). 

<hr>

## Constants

Pi, e, log2(e), log2(10), ln(2), sqrt(2), sqrt(pi), sqrt(2 * pi).

## Functions

### Vectorgebra.Range(low, high, step)

A lazy implementation of range. There is indeed no range. Just a loop with
yield statement. Almost as fast as the built-in range.

### Vectorgebra.abs(arg)

Returns the absolute value of the argument.

### Vectorgebra.sqrt(arg, resolution: int = 10)

A square root implementation that uses Newton's method. You may choose the 
resolution, but any change is not needed there. Pretty much at the same
accuracy as the built-in math.sqrt(). Accepts negative numbers too.

### Vectorgebra.cumsum(arg: list or float)

Returns the cumulative sum of the iterable.

### Vectorgebra.__cumdiv(x, power: int)

Calculates x<sup>n</sup> / power!. This is used to calculate Taylor series
without data loss (at least minimal data loss).

### Vectorgebra.e(exponent, resolution: int)

Calculates e<sup>exponent</sup>. resolution is passed as _power_ to the
__cumdiv(). It will then define the maximal power of the power series
implementation, therefore is a resolution.

### Logarithms

There are 4 distinct logarithm functions: log2, ln, log10, log. Each have
the arguments x and resolution. "resolution" is the number of iterations
and by default is set to 15. "log" function also takes a "base" parameter.

All logarithms are calculated based on "log2" function. "ln" and "log10"
use the related constants instead of recalculating the same value over
and over again.

### Vectorgebra.sigmoid(x, a=1)

Returns the sigmoid functions value at x, where a is the coefficient of x.

### Vectorgebra.Sum(f, a, b, step=Decimal(0.01), control: bool=False, limit=Decimal(0.000001))

Returns the sum of f(x) from a to b. step is the step for Range. If control is true,
stops the sum when the absolute value of the derivative drops under "limit".

### Vectorgebra.mode(data)

Returns the mode of the data. Tuples, lists, vectors and matrices are
accepted.

### Vectorgebra.mean(data)

Calculates the mean of data. "data" must be a one dimensional iterable.

### Vectorgebra.expectation(values, probabilities, moment: int = 1)

"values" and "probabilities" are one dimensional iterables and their lengths must be
equal. There is no value checking for the probabilities. If they sum up
to more than 1 or have negative values, it is up to the user to check 
that. "moment" is the power of "values". Returns the expectation value of
given data.

### Vectorgebra.variance(values, probabilities)

Same constraints as "expectation" apply here. Returns the variance of the given data.

### Vectorgebra.sd(values, probabilities)

Same constraints as "variance" apply here. Returns the standard deviation of the
given data.

### Vectorgebra.maximum(dataset)

Returns the maximum value of dataset. Dataset can be anywhere from tuples to Matrices.

### Vectorgebra.minimum(dataset)

Returns the minimum value of dataset. Dataset can be anywhere from tuples to Matrices.

### Vectorgebra.factorial(x: int)

Calculates the factorial with recursion. Default argument is 1.

### Vectorgebra.permutation(x: int, y: int)

Calculates the _y_ permutations of _x_ elements. Does not utilize
the factorial function. Indeed, uses loops to calculate the aimed
value more efficiently.

### Vectorgebra.combination(x: int, y: int)

Calculates _y_ combinations of _x_ elements. Again, this does not
utilize the factorial function.

### Vectorgebra.multinomial(n: int, *args)

Calculates the multinomial coefficient with n elements, with partitions
described in "args". Does not utilize the factorial.

### Vectorgebra.binomial(n: int, k: int, p: float)

Calculates the probability according to the binomial distribution. n is
the maximum number of events, k is the events that p describes, p is the
probability of the "event" happening.

### Vectorgebra.geometrical(n: int, p: float)

Calculates the probability according to the geometric distribution. n is
the number of total events. p is the probability that the event happens.

### Vectorgebra.poisson(k, l)

Calculates the probability according to the Poisson formula. l is the
lambda factor. k is the "variable" on the whatever system this function
is used to describe.

### Vectorgebra.normal(x, resolution: int = 15)

Calculates the _normal_ gaussian formula given x. "resolution" is directly
passed to the e() function.

### Vectorgebra.gaussian(x, mean, sigma, resolution: int = 15)

Calculates the gaussian given the parameters. "resolution" is directly 
passed to the e() function.

### Vectorgebra.laplace(x, sigma, resolution: int = 15)

Calculates the Laplace distribution given the parameters. "resolution" is directly 
passed to the e() function.

### Vectorgebra.linear_fit(x, y, rate = Decimal(0.01), iterations: int = 15)

Returns the b0 and b1 constants for the linear regression of the given data.
x and y must be one dimensional iterables and their lengths must be equal.
"rate" is the learning rate. "iterations" is the total number of iterations
that this functions going to update the coefficients.

### Vectorgebra.general_fit(x, y, rate = Decimal(0.0000002), iterations: int = 15, degree: int = 1)

Calculates the coefficients for at _degree_ polynomial regression. 
Default _rate_ argument is much much lower because otherwise result 
easily blows up. Returns the coefficients starting from the zeroth 
degree as a Vector object. 

Internally, x and y sets are converted to Vectors if they were not,
so it is faster to initialize them as Vectors.

### Vectorgebra.kmeans(dataset, k=2, iterations=15, a = 0, b = 10)

Applies the K-means algorithm on the dataset. "k" is the number of 
points to assign data clusters. "iterations" is the number of iterations
that the algorithm applies. Dataset must be non-empty. Each row of 
dataset is converted to Vectors internally. Predefining them as
such would make the function faster. 

Every element of dataset must be of the same type whether they are
Vectors or not. Type-checking is based on the first row of the dataset.

Returns a 2-element tuple. First element is a list of Vectors which
point to the cluster centers. Second element is a list of Vectors
which consist of the initial data. This list has the same length as
number of generated cluster centers. Each internal list corresponds
to the same indexed center point. So this data is grouped by cluster
centers.

Initial guesses are random points whose components are random floats
between a and b.

This function does not have decimal support yet.

### Trigonometrics

All are of the format Vectorgebra.name(x, resolution: int).
Calculates the value of the named trigonometric function via Taylor
series. Again, resolution is passed as _power_ to __cumdiv().

Inverse trigonometrics(arcsin, arccos) do not use the helper function 
__cumdiv().

### Vectorgebra.__find(...)

This is a helper function for Math.solve(). Arguments are the same.
Returns the first zero that it finds and saves it to memory.

### Vectorgebra.solve(f, low, high, search_step, res)

Finds zeroes of function f. It may not be able to find all zeroes,
but is pretty precise when it finds some. If the functions derivative
large around its zero, then you should increase resolution to do a
better search.

Retrieves found zeroes from the memory, then clears it. Calling 
multiple instances of this function at the same time will result
in errors because of this global memory usage.

This function is optimized for polynomials. It doesn't matter how many
zeroes they have since this function utilizes a thread pool. This solver
is slow when utilized with Taylor series based functions.

There is an obvious solution to this speed problem though. Just put 
expanded form as the argument. Not the implicit function form.

Exits the main loop if the maximum thread count is reached. This can be used as
a limiter when given b=Infinity(). However, the thread count limit is
most likely 4096.

### Vectorgebra.derivative(f, x, h)

Takes the derivative of f around x with h = _h_. There is no algorithm 
here. It just calculates the derivative.

### Vectorgebra.integrate(f, a, b, delta)

Calculates the integral of f(x) in the interval (a, b) with the specified
delta. Default for delta is 0.01. Uses the midpoint rule.

### Vectorgebra.__mul(...)

A helper function to Vectorgebra.matmul(). Threads inside matmul call this
function.

### Vectorgebra.matmul(m1, m2, max=10)

Threaded matrix multiplication. Its speed is depended on dimensions of 
matrices. Let it be axb and bxc, (a - b) is proportional to this functions
speed. Worst case scenario is square matrices. 44x44 (On CPython) is limit 
for this function to be faster than the overload version of matrix multiplication.

If b > a, normally this function gets even more slower. But there is a
way around. Let it be b > a;

A * B = C

B<sup>T</sup> * A<sup>T</sup> = C<sup>T</sup>

After taking the transposes, we get a > b again. All we have to do is 
to calculate the matrix C<sup>T</sup> instead of C directly then to
calculate the transpose of it.

(I didn't add this function to Matrix class because I have more plans on it.)

### Vectorgebra.findsol(f, x, resolution)

Calculates a single solution of f with Newton's method. x is the starting
guess. resolution is the number of iterations.

### Vectorgebra.complex 

This is the complex number class. It has + - / * overloaded.

#### _Vectorgebra.complex_.conjugate()

Returns the complex conjugate of self.

#### _Vectorgebra.complex_.length()

Returns the length of self, with treating it as a vector.

#### _Vectorgebra.complex_.unit()

Treats the complex number as a vector and returns the unit vector.

#### Vectorgebra.complex.sqrt(arg, resolution: int = 200)

Calculates the square root of the complex number _arg_ and returns it
again, as a complex number. Resolution argument is only passed to arcsin
since it is the only limiting factor for this functions accuracy. Has an
average of 1 degree of error as angle. You may still increase the resolution.
But reaching less than half a degree of error requires for it to be at 
least 600.

The used algorithm calculates the unit vector as e<sup>i*x</sup>. Then halves
the degree, x. Returns the resultant vector at the proper length.

#### _Vectorgebra.complex_.range(lowreal, highreal, lowimg, highimg, step1, step2)

Creates a complex number range, ranging from complex(lowreal, lowimg)
to complex(highreal, highimg). Steps are 1 by default. Again this is 
a lazy implementation.

#### _Vectorgebra.complex_.inverse()

Returns 1 / self. This is used in division. If divisor is complex,
then this function is applied with multiplication to get the result.

#### _Vectorgebra.complex_.rotate(angle: int or float)

Rotates self by angle.

#### _Vectorgebra.complex_.rotationFactor(angle: int or float)

Returns e<sup>i*angle</sup> as a complex number.

<hr>


### Vectorgebra.Infinity

The class of infinities. When initialized, takes one argument describing its sign.
If "True", the infinity is positive (which is the default). If "False", the infinity
is negative.

Logical and mathematical operations are all overloaded. Operations may return "Undefined".
This is a special class that has every mathematical and logical operation overloaded.

### Vectorgebra.Undefined

A special class that corresponds to "undefined" in mathematics.

## Exceptions

### Vectorgebra.DimensionError

Anything related to dimensions of vectors and matrices. Raised in 
Vector class when dimensions of operands don't match or 0 is given
as a dimension to random vector generating functions.

This error is raised in Matrix class when non-square matrices are
passed into inverse calculating functions.

**DimensionError(1) has been changed to RangeError, but is still
in the code.**

### Vectorgebra.ArgTypeError

Anything related to types of arguments. Can take in a str argument
flagged as "hint".

### Vectorgebra.RangeError

Raised when given arguments are out of required range.

### Vectorgebra.AmountError

Raised when the amount of arguments in a function is wrong.

