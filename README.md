# Vector.py 

A linear algebra tool for python

General layout is seperated to two different classes, Vector and Matrix. 
Both have their constructors seperated, but the Matrix class only accepts
vectors as arguments. 

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


#### Footnotes

Rotation is not implemented, because only reasonable way to implement
it would be for just 2D and 3D vectors, which is not satisfying. 


## Matrix

Includes basic operations for matrices.

Basic operations like addition, multiplication subtraction, division
are implemented as overloads. Only comparison operator implemented is
_==_ which returns true if and only if all the elements of the matrices
are equal.

Methods are listed below.

### Matrix.determinant(_m_)

Returns the determinant of the matrix m.

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

## Exceptions

### DimensionError

Anything related to dimensions of vectors and matrices. Raised in 
Vector class when dimensions of operands don't match or 0 is given
as a dimension to random vector generating functions.

This error is raised in Matrix class when non-square matrices are
passed into inverse calculating functions.

### ArgTypeError

Anything related to types of arguments. There are 8 modes of this
exception depending on the conditions. These modes are defined by
different combinations of types. For example type "i" is used for
errors about arguments that should have been _only_ integers.

### ArgumentError

Raised when an incorrect amount of arguments is passed into functions.