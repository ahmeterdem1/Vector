# Vectorgebra

An algebra tool for python

There are 3 main subclasses; Vector, Matrix, complex.
And also there are functions, constants and exception classes. 
Each section is explained below.

## Project details

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

_pip install vectorgebra_

https://pypi.org/project/vectorgebra/

### Update notes on 1.2.0

Algorithm of _Matrix_.fest_inverse() has been completely changed.
It is faster by 200+ times compared to the previous algorithm.
Accuracy is also much much more higher. Old _Matrix_.inverse()
has been left untouched because floating point errors are almost
non existent there in spite of the slowness.

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

### Vectorgebra.Vector.randVint(dim , a, b)

Returns _dim_ dimensional vector which has its elements randomly selected
as integers within the interval (a, b).

### Vectorgebra.Vector.randVfloat(dim, a, b)

Returns _dim_ dimensional vector which has its elements randomly selected
as floats within the interval (a, b).

### Vectorgebra.Vector.randVbool(dim)

Returns _dim_ dimensional vector which has its elements randomly selected
as booleans.

### Vectorgebra.Vector.determinant(*args)

Returns the determinant of the matrix which has rows given as vectors in
*args. This method is not intended for casual use. It is a background
tool for cross product and the determinant method for Matrix class.

### Vectorgebra.Vector.cross(*args)

Returns the cross product of the vectors given in *args.

### _Vectorgebra.Vector_.cumsum()

Returns the cumulative sum.


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

Methods are listed below.

### Vectorgebra.Matrix.determinant(_m_)

Returns the determinant of the matrix m.

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

### _Vectorgebra.Matrix_.inverse()

Returns the inverse matrix of self. Returns None if not invertible.

### Vectorgebra.Matrix.identity(dim)

Returns _dim_ dimensional identity matrix.

### Vectorgebra.Matrix.randMint(m, n, a, b)

Returns _mxn_ matrix of random integers selected from the interval (a, b).

### Vectorgebra.Matrix.randMfloat(m, n, a, b)

Returns _mxn_ matrix of random floats selected from the interval (a, b).

### Vectorgebra.Matrix.randMbool(m, n)

Returns _mxn_ matrix of random booleans.

### _Vectorgebra.Matrix_.echelon()

Returns reduced row echelon form of self. Also does reorganization on
rows and multiplies one of them by -1 every 2 reorganization. This is
for the determinant to remain unchanged.

### _Vectorgebra.Matrix_.det_echelon()

Returns determinant of self via _.echelon()_ method. This is faster
than the other determinant method, which also loses time during type
conversions. But this method does not return the exact value and has a
very small error compared to the original determinant. This is due to
floating point numbers and can be disregarded for most of the uses.

### _Vectorgebra.Matrix_.fast_inverse()

Underlying algorithm of this inverse is completely based on echelon forms.
Sometimes may get some rows wrong, this is due to floating point numbers.
Error amount increases as the dimensions of the matrix increases.

### Vectorgebra.Matrix.cramer(a, number)

Applies Cramers rule to the equation system represented by the matrix _a_.
_number_ indicates which variable to calculate.

### _Vectorgebra.Matrix_.cumsum()

Returns the cumulative sum.

<hr>

## Constants

Pi, e and ln(2). Nothing more.

## Functions

### Vectorgebra.Range(low: int or float, high: int or float, step)

A lazy implementation of range. There is indeed no range. Just a loop with
yield statement. Almost as fast as the built-in range.

### Vectorgebra.abs(arg: int or float)

Returns the absolute value of the argument.

### Vectorgebra.sqrt(arg: int or float, resolution: int = 10)

A square root implementation that uses Newton's method. You may choose the 
resolution, but any change is not needed there. Pretty much at the same
accuracy as the built-in math.sqrt(). Accepts negative numbers too.

### Vectorgebra.cumsum(arg: list or float)

Returns the cumulative sum of the iterable.

### Vectorgebra.__cumdiv(x: int or float, power: int)

Calculates x<sup>n</sup> / power!. This is used to calculate Taylor series
without data loss (at least minimal data loss).

### Vectorgebra.e(exponent: int or float, resolution: int)

Calculates e<sup>exponent</sup>. resolution is passed as _power_ to the
__cumdiv(). It will then define the maximal power of the power series
implementation, therefore is a resolution.

### Trigonometrics

All are of the format Vectorgebra.name(x: int or float, resolution: int).
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

### Vectorgebra.derivative(f, x, h)

Takes the derivative of f around x with h = _h_. There is no algorithm 
here. It just calculates the derivative.

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

Anything related to types of arguments. There are 8 modes of this
exception depending on the conditions. These modes are defined by
different combinations of types. For example type "i" is used for
errors about arguments that should have been _only_ integers.

### Vectorgebra.ArgumentError

Raised when an incorrect amount of arguments is passed into functions.

### Vectorgebra.RangeError

Raised when given arguments are out of required range.

### Vectorgebra.MathArgError

Raised when argument(s) are of wrong type.

### Vectorgebra.MathRangeError

Raised when argument(s) are off range.

