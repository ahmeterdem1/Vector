# Vectorgebra

A numerical methods tool for python, in python.

There are 8 main subclasses; Variable, Vector, Matrix, Tensor, Graph, Complex, Infinity, Undefined.
And also there are functions, constants and exception classes. 
Each section is explained below.

## Project details

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

[![Documentation Status](https://readthedocs.org/projects/vectorgebra/badge/?version=latest)](https://vectorgebra.readthedocs.io/en/latest/?badge=latest)

_pip install vectorgebra_

https://pypi.org/project/vectorgebra/

[Github](https://github.com/ahmeterdem1/Vector)

A C++ remake of this library is currently being developed at [here](https://github.com/ahmeterdem1/Vector_cpp).

Tutorials for this library can be found at [here](https://github.com/ahmeterdem1/examples).

### What can be done with Vectorgebra?

[Here](https://github.com/ahmeterdem1/MLgebra) is a cool little project
I created with Vectorgebra. I want this library to be applicable to
more and more bigger projects as scale. And this was a good test to it.
Creating and training an ML model requires both floating point precision
and good handling of high dimensional tensors.

### Vectorgebra 4.0 beta ready!

The beta version of Vectorgebra 4.0 is ready for testing. It is available
under the "dist" directory of the repository. You can install it with
simply calling pip install on the file path.

Version 4.0.0-beta1(b1) covers all core Python Array API features completely.
Extensive test cases have been introduced to ensure functionality. 

The things that will be coming until the release:

- Linalg extension of the Array API. Implementing this extension is one of the
main goals of the 4.0 release. Project scope covers all functions defined under
this extension, without exceptions.
- Extensive test cases for the computational graph system. Currently implemented
array class supports holding Variable objects, but this is not yet tested to a
decent extent.

The things that are in Array API but will not be implemented in Vectorgebra:

- Dtype functions, we leave this to Python interpreter. The usage of Python
lists as the inner data structure already removes the possibility of any
consistent dtype handling guarantee. Therefore it is mostly pointless to
implement such functions.
- Device related operations, there is no obvious way to implement those
with pure Python.
- Sorting functions, currently this is not in the project scope of Vectorgebra.
Focus is on more calculation heavy operations instead of sorting.
- "take" function under "api.indexing" submodule.
- (Not sure) FFT extension, this is not in the project scope of Vectorgebra,
but may be considered in the future.

These lists are not finalized, as this is the only usable beta release currently.
Things still can change quite a lot.

### New Vectorgebra, 4.0

Yet another major release of Vectorgebra. This release will be Array API
compatibility update.

This version, will be _mostly_ if not fully compatible with Array API.
Device related operations will not be implemented but rest of the API
will be implemented, and is being implemented.

There was a "linalg" module, which was invisible to the end user. Now,
a separate linalg is being created. This module will host most of the
methods from Matrix class. In the final release of 4.0, there will only
be a single Array parent class, from which special but not always necessary
Vector and Matrix are created. For that purpose, methods like inverse, Cholesky
decomposition and etc. will all be carried to "linalg" module.

Autograd is now more performant, with addition of special unary operations.
Common operations like sigmoid, natural exponential, logarithm, etc. each
have their corresponding operators in the computational graph. Each of these
unary operations integrate to the graph with a dummy second Variable object,
that has no children, so that they imitate being binary. This was easier than
updating the graph traversal algorithm.

For now, eager calculation of the computational graph is enforced, but in the
final 4.0 release, I aim to make this optional with a context manager.

And a final notice, a new MLgebra will be developed after the 4.0 release.
Currently, it is not more than just an experiment with Vectorgebra.
