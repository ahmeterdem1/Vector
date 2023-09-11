# Vector and more

An algebra tool for python

General layout consists of 2 files. Vector.py is for
the linear algebra tools with classes of vectors and
matrices. Math.py is for general mathematical operations
and a class for complex numbers.

## Vector

Includes basic and some sophisticated operations on vectors.

Addition, multiplication subtraction, division and operations alike are 
implemented as overloads. Comparison operators compare the length of the
vectors. Only exception is _==_ which returns True if and only if all the
terms of the vectors are equal.

Methods are listed below.

### _Vector_.dot(v)

Returns the dot product of self with v.

### _Vector_.append(v)

Appends the argument to the vector. Returns the new vector. Argument
can be int, float or Vector.

### _Vector_.copy()

Returns the copy of the vector.

### _Vector_.pop(ord)

Functions exactly the same as .pop() for the list class. If left blank,
pops the last element and returns it. If specified, pops the intended
element and returns it.

### _Vector_.length()

Returns the length of self.

### _Vector_.proj(v)

Projects self onto v and returns the resulting vector. Due to the 
division included, may result in inaccurate values that should have
been zero. However, these values are very close to zero and are of
magnitude 10<sup>-17</sup>. This situation is important for the method
Vector.spanify().

### _Vector_.unit()

Returns the unit vector.

### Vector.spanify(*args)

Applies Gram-Schmidt process to the given list of vectors. Returns the
list of resulting vectors. May have inaccuracies explained for Vector.dot()
method.

### Vector.does_span(*args)

Returns True if the given list of vectors span the R<sup>n</sup> space
where n is the number of vectors. Returns False otherwise. Eliminates
the possible error from Vector.spanify() method. Therefore, this method
will work just fine regardless the errors from divisions.

### Vector.randVint(dim , a, b)

Returns _dim_ dimensional vector which has its elements randomly selected
as integers within the interval (a, b).

### Vector.randVfloat(dim, a, b)

Returns _dim_ dimensional vector which has its elements randomly selected
as floats within the interval (a, b).

### Vector.randVbool(dim)

Returns _dim_ dimensional vector which has its elements randomly selected
as booleans.

### Vector.determinant(*args)

Returns the determinant of the matrix which has rows given as vectors in
*args. This method is not intended for casual use. It is a background
tool for cross product and the determinant method for Matrix class.

### Vector.cross(*args)

Returns the cross product of the vectors given in *args.

### _Vector_.cumsum()

Returns the cumulative sum.


#### Footnotes

Rotation is not implemented, because only reasonable way to implement
it would be for just 2D and 3D vectors, which is not satisfying. 


<hr>

## Matrix

Includes basic operations for matrices.

Basic operations like addition, multiplication subtraction, division
are implemented as overloads. Only comparison operator implemented is
_==_ which returns true if and only if all the elements of the matrices
are equal.

Methods are listed below.

### Matrix.determinant(_m_)

Returns the determinant of the matrix m.

### _Matrix_.append(arg)

Appends arg as a new row to self. Only accepts vectors as arguments.

### _Matrix_.copy()

Returns the copy of the matrix.

### _Matrix_.pop(ord)

Functions exactly the same as .pop() in list class. If left blank, pops
the last row and returns it. If specified, pops the row in the given 
order and returns it.

### _Matrix_.transpose()

Returns the transpose matrix of self

### _Matrix_.inverse()

Returns the inverse matrix of self. Returns None if not invertible.

### Matrix.identity(dim)

Returns _dim_ dimensional identity matrix.

### Matrix.randMint(m, n, a, b)

Returns _mxn_ matrix of random integers selected from the interval (a, b).

### Matrix.randMfloat(m, n, a, b)

Returns _mxn_ matrix of random floats selected from the interval (a, b).

### Matrix.randMbool(m, n)

Returns _mxn_ matrix of random booleans.

### _Matrix_.echelon()

Returns reduced row echelon form of self. Also does reorganization on
rows and multiplies one of them by -1 every 2 reorganization. This is
for the determinant to remain unchanged.

### _Matrix_.det_echelon()

Returns determinant of self via _.echelon()_ method. This is faster
than the other determinant method, which also loses time during type
conversions. But this method does not return the exact value and has a
very small error compared to the original determinant. This is due to
floating point numbers and can be disregarded for most of the uses.

### _Matrix_.fast_inverse()

Returns the inverse matrix, faster. Implements echelon algorithm to
calculate determinants. Otherwise, it's the same. The other inverse
method is faster when working with smaller dimensional matrices. But
as the dimension grows, speed of this algorithm gets faster many orders
of magnitude rapidly.

### Matrix.cramer(a, number)

Applies Cramers rule to the equation system represented by the matrix _a_.
_number_ indicates which variable to calculate.

### _Matrix_.cumsum()

Returns the cumulative sum.


<hr>

## Math.py

This files has definitions for general constants and functions. And it also
has a class for complex numbers. 

### Constants

Pi, e and ln(2). Nothing more.

### Math.Range(low: int or float, high: int or float, step)

A lazy implementation of range. There is indeed no range. Just a loop with
yield statement. Almost as fast as the built-in range.

### Math.abs(arg: int or float)

Returns the absolute value of the argument.

### Math.cumsum(arg: list or float)

Returns the cumulative sum of the iterable.

### Math.__cumdiv(x: int or float, power: int)

Calculates x<sup>n</sup> / power!. This is used to calculate Taylor series
without data loss (at least minimal data loss).

### Math.e(exponent: int or float, resolution: int)

Calculates e<sup>exponent</sup>. resolution is passed as _power_ to the
__cumdiv(). It will then define the maximal power of the power series
implementation, therefore is a resolution.

### Trigonometrics

All are of the format Math.name(x: int or float, resolution: int).
Calculates the value of the named trigonometric function via Taylor
series. Again, resolution is passed as _power_ to __cumdiv().

### Math.__find(...)

This is a helper function for Math.solve(). Arguments are the same.
Returns the first zero that it finds and saves it to memory.

### Math.solve(f, low, high, search_step, res)

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

### _class_ Math.complex 

This is the complex number class. It has + - / * overloaded.

#### _Math.complex_.conjugate()

Returns the complex conjugate of self.

#### _Math.complex_.length()

Returns the length of self, with treating it as a vector.

#### _Math.complex_.range(lowreal, highreal, lowimg, highimg, step1, step2)

Creates a complex number range, ranging from complex(lowreal, lowimg)
to complex(highreal, highimg). Steps are 1 by default. Again this is 
a lazy implementation.

#### _Math.complex_.inverse()

Returns 1 / self. This is used in division. If divisor is complex,
then this function is applied with multiplication to get the result.

#### _Math.complex_.rotate(angle: int or float)

Rotates self by angle.

#### _Math.complex_.rotationFactor(angle: int or float)

Returns e<sup>i*angle</sup> as a complex number.

<hr>

## Exceptions

### DimensionError

Anything related to dimensions of vectors and matrices. Raised in 
Vector class when dimensions of operands don't match or 0 is given
as a dimension to random vector generating functions.

This error is raised in Matrix class when non-square matrices are
passed into inverse calculating functions.

**DimensionError(1) has been changed to RangeError, but is still
in the code.**

### ArgTypeError

Anything related to types of arguments. There are 8 modes of this
exception depending on the conditions. These modes are defined by
different combinations of types. For example type "i" is used for
errors about arguments that should have been _only_ integers.

### ArgumentError

Raised when an incorrect amount of arguments is passed into functions.

### RangeError

Raised when given arguments are out of required range.

### MathArgError

Raised when argument(s) are of wrong type.

### MathRangeError

Raised when argument(s) are off range.

