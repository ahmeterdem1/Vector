"""
    The numerical analysis library, fully implemented with Python.
"""

from . import linalg  # Hide complex methods under linalg name
from .array import *
from .math import *
from .utils import *
from .variable import Variable, grad, autograd


__version__ = "4.0.0b2"
