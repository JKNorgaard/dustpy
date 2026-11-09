"""Shared direct solver for dust and solid-tracer transport."""

import numpy as np
import scipy.sparse as sp


class _ReorderedLU:
    """Keep the permutation with its factorization for repeated RHS solves."""

    def __init__(self, lu, order):
        self._lu = lu
        self._order = order

    def solve(self, rhs):
        y = self._lu.solve(np.asarray(rhs)[self._order])
        x = np.empty_like(y)
        x[self._order] = y
        return x


def factorize(A):
    """Factor the current matrix, selecting the ordering by stored density.

    The threshold is empirical: full-grid dust/ice benchmarks favored MMD
    for early sparse states and fresh RCM for denser evolved states. Numeric
    factors and the RCM permutation must be rebuilt each transport step.
    """
    options = dict(diag_pivot_thresh=0.0, options=dict(SymmetricMode=True))
    if A.nnz < 8 * A.shape[0]:
        return sp.linalg.splu(A, permc_spec="MMD_AT_PLUS_A", **options)

    # These matrices are nonsymmetric; form the ordering from both directions.
    order = sp.csgraph.reverse_cuthill_mckee(A, symmetric_mode=False)
    B = A[order, :][:, order].tocsc()
    lu = sp.linalg.splu(B, permc_spec="NATURAL", **options)
    return _ReorderedLU(lu, order)
