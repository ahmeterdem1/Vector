"""
    The Array API v2023.12 implementation of Vectorgebra.

    Provides Array API standard functionalities, with linear algebra extension.
"""

from .constants import *
from .creation import *
from .dtypefunc import *
from .dtype import *
from .ewise import *
from .indexing import *
from .statistical import *
from .utility import *
from .searching import *
from .set import *
from . import random, linalg


# DO NOT blindly import any above. This module is hidden under namespace "api",
# so it probably won't mess with rest of vectorgebra.
