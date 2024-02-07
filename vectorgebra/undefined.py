from .exceptions import *

class Undefined:
    """
    Represents an undefined or non-existent value.

    This class is used to represent the concept of an undefined value or a non-existent quantity.
    It behaves differently from typical numeric or boolean types in that operations involving
    instances of Undefined generally return Undefined itself or False, and comparisons always return False.
    """

    def __str__(self):
        return "Undefined()"

    def __repr__(self):
        return "Undefined()"

    def __add__(self, other):
        return Undefined()

    def __radd__(self, other):
        return Undefined()

    def __iadd__(self, other):
        return self

    def __sub__(self, other):
        return Undefined()

    def __rsub__(self, other):
        return Undefined()

    def __isub__(self, other):
        return self

    def __mul__(self, other):
        return Undefined()

    def __rmul__(self, other):
        return Undefined()

    def __imul__(self, other):
        return self

    def __truediv__(self, other):
        return Undefined()

    def __floordiv__(self, other):
        return Undefined()

    def __rtruediv__(self, other):
        return Undefined()

    def __rfloordiv__(self, other):
        return Undefined()

    def __idiv__(self, other):
        return self

    def __ifloordiv__(self, other):
        return self

    def __neg__(self):
        return self

    def __invert__(self):
        return self

    # Normally boolean resulting operations are defined as "False"
    def __eq__(self, other):
        return False

    # nothing is equal to undefined
    def __ne__(self, other):
        return True

    def __gt__(self, other):
        return False

    def __ge__(self, other):
        return False

    def __lt__(self, other):
        return False

    def __le__(self):
        return False

    def __and__(self, other):
        return False

    def __rand__(self, other):
        return False

    def __iand__(self, other):
        return False

    # This is important
    def __or__(self, other):
        return other

    def __ror__(self, other):
        return other

    def __ior__(self, other):
        return other

    def __xor__(self, other):
        return False

    def __rxor__(self, other):
        return False

    def __ixor__(self, other):
        return False