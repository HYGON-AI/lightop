import collections
from itertools import repeat
from typing import Any, Union
from torch import Size

def _ntuple(n, name="parse"):
    def parse(x):
        if isinstance(x, collections.abc.Iterable):
            return tuple(x)
        return tuple(repeat(x, n))

    parse.__name__ = name
    return parse

_pair = _ntuple(2, "_pair")


_shape_t = Union[int, list[int], Size]