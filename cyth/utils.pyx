# cython: nonecheck=False
# cython: boundscheck=False
# cython: wraparound=False
# cython: cdivision=True
# cython: language_level=3
# cython: infer_types=False
# cython: embedsignature=True

cdef extern from "math.h" nogil:
    cdef:
        double M_PI
        double cos(double)

import numpy as np
cimport numpy as np


cpdef np.ndarray get_phs_corrs(const double[:] phs_spec) except +:

    cdef:
        Py_ssize_t i, j, n_phss, n_corrs, phss_ctr
        double[:] corrs

    assert phs_spec.ndim == 1, phs_spec.ndim

    n_phss = phs_spec.shape[0]

    n_corrs = (n_phss * (n_phss - 1)) // 2

    corrs = np.empty(n_corrs, dtype=np.float64)

    phss_ctr = 0
    for i in range(n_phss):
        for j in range(i + 1, n_phss):

            corrs[phss_ctr] = cos(phs_spec[i] - phs_spec[j])

            phss_ctr += 1

    assert phss_ctr == n_corrs, (phss_ctr, n_corrs)

    return np.asarray(corrs)


cpdef void adjust_phss_range(
        const double[:, :] phss, double[:, :] phss_adj) except +:

    cdef:
        Py_ssize_t i, j
        double phs, ratio

    assert phss.shape[0] == phss_adj.shape[0], (
        phss.shape, phss_adj.shape)

    assert phss.shape[1] == phss_adj.shape[1], (
        phss.shape, phss_adj.shape)

    for i in range(phss.shape[0]):
        for j in range(phss.shape[1]):

            phs = phss[i, j]

            if phs > +M_PI:
                ratio = (phs / +M_PI) - 1
                phs = -M_PI * (1 - ratio)

            elif phs < -M_PI:
                ratio = (phs / -M_PI) - 1
                phs = +M_PI * (1 - ratio)

            phss_adj[i, j] = phs

    return