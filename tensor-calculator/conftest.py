import os
import sys
from fractions import Fraction

import pytest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from recur_tensor import RecurTensor


def build(p, q, n, data):
    t = RecurTensor(p, q, n)

    def fill(node, src):
        if node.p + node.q == 0:
            node.data = Fraction(src)
        else:
            for i in range(n):
                fill(node.data[i], src[i])

    fill(t, data)
    return t


def dump(t):
    if t.p + t.q == 0:
        return Fraction(t.data)
    return [dump(child) for child in t.data]


def F(*args):
    return Fraction(*args)


def frac_list(data):
    if isinstance(data, (list, tuple)):
        return [frac_list(x) for x in data]
    return Fraction(data)


@pytest.fixture
def m22():
    return build(2, 0, 2, [[1, 2], [3, 4]])


@pytest.fixture
def t3():
    return build(3, 0, 2, [[[100 * i + 10 * j + k for k in range(2)]
                           for j in range(2)] for i in range(2)])
