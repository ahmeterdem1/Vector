from ctypes import (
    c_int8 as int8,
    c_uint8 as uint8,
    c_int16 as int16,
    c_uint16 as uint16,
    c_int32 as int32,
    c_uint32 as uint32,
    c_int64 as int64,
    c_uint64 as uint64,
    c_bool as _bool,
    c_float as float32,
    c_double as float64,
    # No complex defined
)

from typing import Union, Type

INT_TYPE = Union[int, int8, uint8, int16, uint16, int32, uint32, int64, uint64]
FLOAT_TYPE = Union[float, float32, float64]
BOOL_TYPE = Union[bool, _bool]
BASIC_ITERABLE = Union[list, tuple]


def promotion(dtype1: Type, dtype2: Type) -> Type:
    pass
