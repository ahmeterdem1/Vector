from ..ndarray import Array
from typing import Tuple
from collections import OrderedDict

def unique_all(x: Array) -> Tuple[Array, Array, Array, Array]:
    values_ = list(OrderedDict.fromkeys(x.values))  # So that the relative order is preserved
    indices_ = [-1] * len(values_)
    counts_ = dict.fromkeys(values_, 0)
    inverse_indices_ = [0] * x.size

    # Find all required fields in O(size) time
    value_index = 0
    for i in range(x.size):
        counts_[x.values[i]] += 1
        inverse_indices_[i] = values_.index(x.values[i])
        if x.values[i] == values_[value_index]:
            indices_[value_index] = i  # Instead of appending, initialize the list in its full size and just index/assign here
            value_index += 1

    values = Array(values_)
    indices = Array(indices_)
    inverse_indices = Array(inverse_indices_)
    inverse_indices.reshape(x.shape)
    counts = Array(list(counts_.values()))

    return values, indices, inverse_indices, counts

def unique_counts(x: Array) -> Tuple[Array, Array]:
    values_ = list(set(x.values))
    counts_ = [x.values.count(v) for v in values_]

    values = Array(values_)
    counts = Array(counts_)

    return values, counts

def unique_inverse(x: Array):
    values_ = list(set(x.values))
    inverse_indices_ = [values_.index(x_val) for x_val in x.values]

    values = Array(values_)
    inverse_indices = Array(inverse_indices_)
    inverse_indices.reshape(x.shape)

    return values, inverse_indices

def unique_values(x: Array) -> Array:
    return Array(list(set(x.values)))
