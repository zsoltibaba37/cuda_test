#!/usr/bin/env python

"""
from numba import cuda
print(cuda.is_available())

import cupy as cp
x = cp.arange(10)
print(x * 2)
"""

import cupy as cp
x = cp.arange(10**7, dtype=cp.float32)
y = cp.sqrt(x)
print(y[:10])



