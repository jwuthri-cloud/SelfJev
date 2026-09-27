"""Paired statistics."""

import math


def mcnemar(a, b):
    oa, ob = sum(a[i] and not b[i] for i in a), sum(b[i] and not a[i] for i in a)
    n, k = oa + ob, min(oa, ob)
    return oa, ob, (min(1.0, 2 * sum(math.comb(n, j) for j in range(k + 1)) / 2**n) if n else 1.0)
